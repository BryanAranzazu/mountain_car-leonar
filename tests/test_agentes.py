"""Pruebas rapidas de los dos agentes. Corren en segundos: no entrenan hasta converger."""
from itertools import pairwise
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from mountain_car.agents import DQNAgent, QLearningAgent
from mountain_car.cli import _run_episode

ENV_ID = "MountainCar-v0"
SAVES = Path(__file__).resolve().parents[1] / "saves"


# ── Q-Learning tabular ───────────────────────────────────────────────


def test_discretize_da_indices_validos_y_hashables():
    agent = QLearningAgent(ENV_ID, n_bins=20)
    for obs in ([-1.2, -0.07], [0.6, 0.07], [-0.5, 0.0], [5.0, -5.0]):
        key = agent.discretize(np.array(obs, dtype=np.float32))
        assert isinstance(key, tuple) and len(key) == 2
        assert all(isinstance(i, int) and 0 <= i < 20 for i in key)
        hash(key)


def test_discretize_separa_observaciones_lejanas():
    agent = QLearningAgent(ENV_ID)
    assert agent.discretize(np.array([-1.0, -0.05])) != agent.discretize(np.array([0.4, 0.05]))


def test_modo_determinista_nunca_explora():
    agent = QLearningAgent(ENV_ID, epsilon_start=1.0)
    state = (3, 4)
    agent.q_table[state] = np.array([-5.0, -1.0, -9.0])
    assert {agent.select_action(state, deterministic=True) for _ in range(200)} == {1}


def test_actualizacion_td_coincide_con_la_formula():
    agent = QLearningAgent(ENV_ID, lr=0.5, gamma=0.9)
    s, s2 = (0, 0), (1, 1)
    agent.q_table[s2] = np.array([-2.0, -4.0, -3.0])
    agent._update(s, 2, -1.0, s2, terminated=False)
    assert agent.q_table[s][2] == pytest.approx(0.5 * (-1.0 + 0.9 * -2.0))


def test_estado_terminal_no_hace_bootstrap():
    agent = QLearningAgent(ENV_ID, lr=1.0)
    s, s2 = (0, 0), (1, 1)
    agent.q_table[s2] = np.array([-50.0, -50.0, -50.0])
    agent._update(s, 0, -1.0, s2, terminated=True)
    assert agent.q_table[s][0] == pytest.approx(-1.0)


# ── DQN ──────────────────────────────────────────────────────────────


def test_red_q_devuelve_un_valor_por_accion():
    agent = DQNAgent(ENV_ID, hidden=16)
    out = agent.q_net(torch.zeros(7, agent.state_dim))
    assert out.shape == (7, agent.action_dim)


def test_paso_de_aprendizaje_cambia_solo_la_red_en_linea():
    torch.manual_seed(0)
    agent = DQNAgent(ENV_ID, hidden=16, batch_size=8)
    env = gym.make(ENV_ID)
    obs, _ = env.reset(seed=0)
    for _ in range(32):
        a = env.action_space.sample()
        nxt, r, term, _trunc, _ = env.step(a)
        agent.buffer.push(obs, a, float(r), nxt, term)
        obs = nxt
    env.close()
    antes_linea = [p.clone() for p in agent.q_net.parameters()]
    antes_objetivo = [p.clone() for p in agent.target_net.parameters()]
    loss = agent._learn()
    assert np.isfinite(loss) and loss > 0
    assert any(not torch.equal(a, b) for a, b in zip(antes_linea, agent.q_net.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(antes_objetivo, agent.target_net.parameters()))


def test_exploracion_persistente_produce_rachas_largas():
    import random

    random.seed(0)
    obs = np.zeros(2, dtype=np.float32)

    def racha_media(p: float) -> float:
        agent = DQNAgent(ENV_ID, hidden=8, explore_repeat=p, epsilon_start=1.0)
        acciones = [agent.select_action(obs) for _ in range(4000)]
        cambios = sum(a != b for a, b in pairwise(acciones))
        return len(acciones) / (cambios + 1)

    assert racha_media(0.0) < 2.0  # epsilon-greedy de libro: ~1.5 pasos
    assert racha_media(0.95) > 15.0  # correccion del Ejercicio 3: ~25-30 pasos


def test_explore_repeat_sobrevive_a_guardar_y_cargar(tmp_path):
    agent = DQNAgent(ENV_ID, hidden=8, explore_repeat=0.8)
    path = tmp_path / "dqn.pt"
    agent.save(path)
    assert DQNAgent.load(path).explore_repeat == 0.8


# ── agentes entrenados que acompanan al repositorio ──────────────────


@pytest.mark.parametrize(
    ("cls", "archivo", "umbral"),
    [(QLearningAgent, "qlearning_mountaincar.pkl", -150.0), (DQNAgent, "dqn_mountaincar.pt", -110.0)],
)
def test_agentes_guardados_alcanzan_la_bandera(cls, archivo, umbral):
    path = SAVES / archivo
    if not path.exists():
        pytest.skip(f"no hay agente guardado en {path}")
    agent = cls.load(path)
    env = gym.make(ENV_ID)
    env.reset(seed=1007)
    resultados = [_run_episode(env, agent) for _ in range(20)]
    env.close()
    media = np.mean([r for r, _, _ in resultados])
    assert media > umbral, f"media voraz {media:.1f} por debajo de {umbral}"
