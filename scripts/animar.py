"""Graba un episodio voraz de cada agente como GIF.

    uv run --with pillow python scripts/animar.py

Usa el render `rgb_array` de Gymnasium, asi que no abre ninguna ventana.
"""
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gymnasium as gym
import torch
from PIL import Image, ImageDraw

from mountain_car.agents import DQNAgent, QLearningAgent

FIG = Path("resultados") / "figuras"
ENV_ID = "MountainCar-v0"
AGENTES = {
    "qlearning": (QLearningAgent, Path("saves/qlearning_mountaincar.pkl"), "Q-Learning tabular"),
    "dqn": (DQNAgent, Path("saves/dqn_mountaincar.pt"), "DQN"),
}
NOMBRE_ACCION = {0: "<< empuja a la izquierda", 1: "-- no empuja", 2: "empuja a la derecha >>"}


def grabar(nombre: str, seed: int = 1007) -> int:
    cls, path, titulo = AGENTES[nombre]
    agente = cls.load(path)
    env = gym.make(ENV_ID, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    cuadros, paso, fin = [], 0, False
    while not fin:
        accion, _ = agente.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(accion))
        paso += 1
        fin = terminated or truncated
        if paso % 2 and not fin:
            continue  # un cuadro de cada dos: el GIF pesa la mitad
        img = Image.fromarray(env.render()).resize((480, 320))
        d = ImageDraw.Draw(img)
        d.text((10, 8), f"{titulo}  |  paso {paso}  |  recompensa {-paso}", fill=(20, 20, 20))
        d.text((10, 24), NOMBRE_ACCION[int(accion)], fill=(20, 20, 20))
        cuadros.append(img.quantize(64))
    env.close()
    cuadros += [cuadros[-1]] * 12  # pausa al llegar
    cuadros[0].save(FIG / f"agente_{nombre}.gif", save_all=True, append_images=cuadros[1:], duration=50,
                    loop=0, optimize=True)
    return paso


if __name__ == "__main__":
    torch.set_num_threads(1)
    FIG.mkdir(parents=True, exist_ok=True)
    for nombre in AGENTES:
        print(f"{nombre}: episodio de {grabar(nombre)} pasos -> resultados/figuras/agente_{nombre}.gif")
