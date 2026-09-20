"""Dibuja las curvas de entrenamiento a partir de los CSV de resultados/.

    uv run --with matplotlib python scripts/graficar.py

matplotlib no es dependencia del paquete; `--with` lo instala solo para esta orden.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("resultados")
NOMBRES = {"qlearning": "Q-Learning tabular", "dqn": "DQN"}
COLOR = {"qlearning": "#1f6f8b", "dqn": "#b5542a"}


def cargar(agente: str):
    hist = np.loadtxt(OUT / f"{agente}_historial.csv", skiprows=1)
    ctrl = np.loadtxt(OUT / f"{agente}_controles.csv", skiprows=1, delimiter=",")
    resumen = json.loads((OUT / f"{agente}_evaluacion.json").read_text())
    return hist, ctrl, resumen


def curva(agente: str) -> None:
    hist, ctrl, res = cargar(agente)
    media = np.convolve(hist, np.ones(100) / 100, mode="valid")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(np.arange(1, len(hist) + 1), hist, color=COLOR[agente], alpha=0.18, lw=0.6,
            label="recompensa por episodio (con exploracion)")
    ax.plot(np.arange(100, len(hist) + 1), media, color=COLOR[agente], lw=1.8,
            label="media movil de 100 episodios")
    ax.plot(ctrl[:, 0], ctrl[:, 1], "o-", color="black", ms=3, lw=0.8,
            label="control voraz (20 episodios, sin exploracion)")
    ax.axvline(res["punto_de_control_elegido"], color="gray", ls=":", lw=1)
    ax.axhline(-110, color="green", ls="--", lw=1, label="umbral convencional de resuelto (-110)")
    ax.set_ylim(-205, -80)
    ax.set_xlabel("episodio de entrenamiento")
    ax.set_ylabel("recompensa total (= -pasos)")
    ax.set_title(
        f"{NOMBRES[agente]} en MountainCar-v0 | evaluacion final: {res['eval_media']:.1f} "
        f"+/- {res['eval_desviacion']:.1f}, bandera en {res['eval_banderas']}/{res['eval_episodios']}"
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / f"{agente}_curva.png", dpi=140)
    plt.close(fig)


def comparacion() -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for agente in ("qlearning", "dqn"):
        _, ctrl, _ = cargar(agente)
        ax.plot(ctrl[:, 0], ctrl[:, 1], "o-", ms=3, lw=1.2, color=COLOR[agente], label=NOMBRES[agente])
    ax.axhline(-110, color="green", ls="--", lw=1, label="umbral -110")
    ax.set_xscale("log")
    ax.set_xlabel("episodios de entrenamiento (escala logaritmica)")
    ax.set_ylabel("media voraz en el control")
    ax.set_title("Desempeno sin exploracion a lo largo del entrenamiento")
    ax.legend()
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "comparacion_controles.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    for a in ("qlearning", "dqn"):
        curva(a)
    comparacion()
    print("Graficas guardadas en resultados/")
