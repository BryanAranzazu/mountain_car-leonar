"""Entrena un agente con semilla fija, lo evalua por tramos y conserva el mejor.

Uso:
    uv run python scripts/experimento.py qlearning --episodes 20000 --chunk 500
    uv run python scripts/experimento.py dqn --episodes 2500 --chunk 50

El entrenamiento se corta en tramos de `--chunk` episodios. Al cerrar cada tramo se
juegan `--check-episodes` episodios voraces (sin exploracion) sobre semillas fijas,
y si la media supera a la mejor vista hasta entonces se guarda esa version del agente
en saves/, con el nombre que usa la CLI. Asi el archivo guardado corresponde al mejor
punto del entrenamiento y no al ultimo, que en MountainCar puede ser bastante peor.

La cifra que se reporta sale de una evaluacion final aparte, con semillas que no se
usaron ni para entrenar ni para elegir el punto de control.

Archivos que deja en resultados/metricas/:
    <agente>_historial.csv     recompensa de cada episodio de entrenamiento
    <agente>_controles.csv     media voraz al cierre de cada tramo
    <agente>_evaluacion.json   resumen numerico
"""
import argparse
import json
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from mountain_car.cli import AGENTS, ENV_ID, _run_episode

OUT = Path("resultados") / "metricas"


def evaluar(agent, episodios: int, seed: int) -> tuple[np.ndarray, int]:
    env = gym.make(ENV_ID)
    env.reset(seed=seed)
    results = [_run_episode(env, agent) for _ in range(episodios)]
    env.close()
    return np.array([r for r, _, _ in results]), int(sum(ok for _, _, ok in results))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("agent", choices=tuple(AGENTS))
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--check-episodes", type=int, default=20)
    ap.add_argument("--eval-episodes", type=int, default=100)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cls, save_path = AGENTS[args.agent]
    agent = cls(ENV_ID)

    history: list[float] = []
    controles: list[tuple[int, float, int]] = []
    best_mean, best_at = -np.inf, 0
    t0 = time.perf_counter()

    for start in range(0, args.episodes, args.chunk):
        n = min(args.chunk, args.episodes - start)
        history += agent.train(total_episodes=n, seed=args.seed if start == 0 else None)
        rewards, flags = evaluar(agent, args.check_episodes, seed=args.seed + 500)
        controles.append((start + n, float(rewards.mean()), flags))
        if rewards.mean() > best_mean:
            best_mean, best_at = float(rewards.mean()), start + n
            agent.save(save_path)
        print(f"  control @ {start + n}: media voraz {rewards.mean():.2f}, banderas {flags}/{args.check_episodes}")

    minutes = (time.perf_counter() - t0) / 60
    final_rewards, final_flags = evaluar(agent, args.eval_episodes, seed=args.seed + 1000)

    best = cls.load(save_path)
    rewards, flags = evaluar(best, args.eval_episodes, seed=args.seed + 1000)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savetxt(OUT / f"{args.agent}_historial.csv", history, fmt="%.0f", header="recompensa", comments="")
    np.savetxt(
        OUT / f"{args.agent}_controles.csv", controles, fmt=["%d", "%.2f", "%d"],
        delimiter=",", header="episodio,media_voraz,banderas", comments="",
    )

    hist = np.array(history)
    moving = np.convolve(hist, np.ones(100) / 100, mode="valid")
    summary = {
        "agente": args.agent,
        "semilla": args.seed,
        "episodios_entrenamiento": args.episodes,
        "minutos_entrenamiento": round(minutes, 2),
        "primer_episodio_con_bandera": int(np.argmax(hist > -200)) + 1 if (hist > -200).any() else None,
        "mejor_episodio_entrenamiento": float(hist.max()),
        "mejor_media_movil_100_entrenamiento": round(float(moving.max()), 2),
        "punto_de_control_elegido": best_at,
        "eval_episodios": args.eval_episodes,
        "eval_media": round(float(rewards.mean()), 2),
        "eval_desviacion": round(float(rewards.std()), 2),
        "eval_mejor": float(rewards.max()),
        "eval_peor": float(rewards.min()),
        "eval_banderas": flags,
        "eval_media_ultimo_modelo": round(float(final_rewards.mean()), 2),
        "eval_banderas_ultimo_modelo": final_flags,
    }
    (OUT / f"{args.agent}_evaluacion.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
