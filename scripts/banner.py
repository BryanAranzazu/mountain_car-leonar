"""Dibuja el banner del repositorio a partir de los datos del propio taller.

    uv run --with matplotlib python scripts/banner.py

Izquierda: el valle de MountainCar con el carro y la bandera. Derecha: la politica que
aprendio DQN y la espiral de un episodio voraz en el plano posicion-velocidad.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from graficar import SAVES, TEMAS, rejilla_estados, trayectoria, valores_q
from matplotlib.colors import ListedColormap

from mountain_car.agents import DQNAgent

FIG = Path("resultados") / "figuras"


def banner(t: dict, tema: str, agente) -> None:
    fig = plt.figure(figsize=(12.8, 3.2), facecolor=t["fondo"])

    # Politica y espiral, a la derecha y atenuadas para que el titulo mande.
    ax = fig.add_axes([0.64, 0.0, 0.36, 1.0])
    pos, vel, obs = rejilla_estados(220)
    q, _ = valores_q(agente, "dqn", obs)
    ax.imshow(q.argmax(1).reshape(len(vel), len(pos)), origin="lower", aspect="auto",
              extent=[pos[0], pos[-1], vel[0], vel[-1]], cmap=ListedColormap(t["acciones"]), alpha=0.55,
              interpolation="nearest")
    tr = trayectoria(agente)
    ax.plot(tr[:, 0], tr[:, 1], color=t["texto"], lw=2)
    ax.plot(*tr[0], marker="o", ms=8, mfc=t["fondo"], mec=t["texto"], mew=1.8)
    ax.axis("off")

    # El valle, el carro y la bandera.
    ax2 = fig.add_axes([0.0, 0.0, 0.64, 0.44])
    x = np.linspace(-1.2, 0.6, 400)
    y = np.sin(3 * x)
    ax2.fill_between(x, y, -1.4, color=t["rejilla"])
    ax2.plot(x, y, color=t["tenue"], lw=2)
    ax2.plot(tr[:, 0], np.sin(3 * tr[:, 0]) + 0.06, color=t["serie"]["dqn"], lw=0, marker="o", ms=2.2, alpha=0.5)
    cx = -0.5
    ax2.plot([cx], [np.sin(3 * cx) + 0.13], marker="s", ms=13, color=t["texto"])
    ax2.plot([0.5, 0.5], [np.sin(1.5), np.sin(1.5) + 0.55], color=t["texto"], lw=2)
    ax2.plot([0.545], [np.sin(1.5) + 0.47], marker=">", ms=12, color=t["serie"]["dqn"])
    ax2.set_xlim(-1.2, 0.6)
    ax2.set_ylim(-1.4, 1.7)
    ax2.axis("off")

    fig.text(0.035, 0.80, "MountainCar-v0", fontsize=30, color=t["texto"], va="center")
    fig.text(0.035, 0.62, "Q-Learning tabular y DQN  ·  exploración persistente con recompensa plana",
             fontsize=12.5, color=t["texto2"], va="center")
    fig.text(0.035, 0.50, "Maestría en Inteligencia Artificial  ·  Universidad de La Sabana", fontsize=10.5,
             color=t["tenue"], va="center")
    fig.savefig(FIG / f"banner_{tema}.png", dpi=100, facecolor=t["fondo"])
    plt.close(fig)


if __name__ == "__main__":
    torch.set_num_threads(1)
    ag = DQNAgent.load(SAVES / "dqn_mountaincar.pt")
    for tema, t in TEMAS.items():
        banner(t, tema, ag)
    print("Banner guardado en resultados/figuras/banner_claro.png y banner_oscuro.png")
