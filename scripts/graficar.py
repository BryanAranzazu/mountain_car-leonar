"""Genera todas las figuras del repositorio, en version clara y oscura.

    uv run --with matplotlib python scripts/graficar.py

Lee resultados/metricas/, resultados/registros/ y los agentes de saves/.
Escribe en resultados/figuras/ un PNG por figura y tema (sufijos _claro y _oscuro),
para que el README muestre la variante que corresponde al tema del lector.
matplotlib no es dependencia del paquete; `--with` lo instala solo para esta orden.
"""
import json
import re
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from mountain_car.agents import DQNAgent, QLearningAgent

MET = Path("resultados") / "metricas"
REG = Path("resultados") / "registros"
FIG = Path("resultados") / "figuras"
SAVES = Path("saves")
ENV_ID = "MountainCar-v0"

NOMBRES = {"qlearning": "Q-Learning tabular", "dqn": "DQN"}

# Paleta validada (contraste, separacion bajo daltonismo) para fondo claro y oscuro.
TEMAS = {
    "claro": {
        "fondo": "#fcfcfb", "texto": "#0b0b0b", "texto2": "#52514e", "tenue": "#898781", "rejilla": "#e1e0d9", "eje": "#c3c2b7",
        "serie": {"qlearning": "#2a78d6", "dqn": "#eb6834"},
        "acciones": ["#2a78d6", "#c3c2b7", "#eb6834"],
        "rampa": ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"], "vacio": "#f0efec",
    },
    "oscuro": {
        "fondo": "#1a1a19", "texto": "#ffffff", "texto2": "#c3c2b7", "tenue": "#898781", "rejilla": "#2c2c2a", "eje": "#383835",
        "serie": {"qlearning": "#3987e5", "dqn": "#d95926"},
        "acciones": ["#3987e5", "#52514e", "#d95926"],
        "rampa": ["#0d366b", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"], "vacio": "#262625",
    },
}
ACCIONES = ["empujar a la izquierda", "no empujar", "empujar a la derecha"]


# ── utilidades ───────────────────────────────────────────────────────


def estilo(t: dict) -> None:
    plt.rcParams.update({
        "figure.facecolor": t["fondo"], "axes.facecolor": t["fondo"], "savefig.facecolor": t["fondo"],
        "text.color": t["texto"], "axes.labelcolor": t["texto2"], "axes.titlecolor": t["texto"],
        "xtick.color": t["tenue"], "ytick.color": t["tenue"], "xtick.labelcolor": t["texto2"],
        "ytick.labelcolor": t["texto2"], "axes.edgecolor": t["eje"], "grid.color": t["rejilla"],
        "grid.linewidth": 0.8, "axes.grid": True, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 12,
        "axes.titlelocation": "left", "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.frameon": False, "legend.fontsize": 9,
        "font.family": "DejaVu Sans", "figure.dpi": 100,
    })


def guardar(fig, nombre: str, tema: str) -> None:
    fig.savefig(FIG / f"{nombre}_{tema}.png", dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def cargar(agente: str):
    hist = np.loadtxt(MET / f"{agente}_historial.csv", skiprows=1)
    ctrl = np.loadtxt(MET / f"{agente}_controles.csv", skiprows=1, delimiter=",")
    resumen = json.loads((MET / f"{agente}_evaluacion.json").read_text())
    return hist, ctrl, resumen


def agentes():
    return {
        "qlearning": QLearningAgent.load(SAVES / "qlearning_mountaincar.pkl"),
        "dqn": DQNAgent.load(SAVES / "dqn_mountaincar.pt"),
    }


def rejilla_estados(n: int = 160):
    pos = np.linspace(-1.2, 0.6, n)
    vel = np.linspace(-0.07, 0.07, n)
    P, V = np.meshgrid(pos, vel)
    return pos, vel, np.stack([P.ravel(), V.ravel()], axis=1).astype(np.float32)


def valores_q(agente, nombre: str, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (Q de cada estado, mascara de estados que el agente visito)."""
    if nombre == "dqn":
        with torch.no_grad():
            return agente.q_net(torch.as_tensor(obs)).numpy(), np.ones(len(obs), dtype=bool)
    claves = [agente.discretize(o) for o in obs]
    visto = np.array([k in agente.q_table for k in claves])
    q = np.array([agente.q_table[k] if v else np.zeros(3) for k, v in zip(claves, visto)])
    return q, visto


def trayectoria(agente, seed: int = 1007) -> np.ndarray:
    env = gym.make(ENV_ID)
    obs, _ = env.reset(seed=seed)
    puntos, fin = [obs], False
    while not fin:
        accion, _ = agente.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(accion))
        puntos.append(obs)
        fin = terminated or truncated
    env.close()
    return np.array(puntos)


# ── figuras ──────────────────────────────────────────────────────────


def curvas(t: dict, tema: str) -> None:
    """Multiples pequenos: un panel por agente, mismo eje vertical."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, agente in zip(axes, ("qlearning", "dqn")):
        hist, ctrl, res = cargar(agente)
        color = t["serie"][agente]
        x = np.arange(1, len(hist) + 1)
        media = np.convolve(hist, np.ones(100) / 100, mode="valid")
        ax.plot(x, hist, color=color, alpha=0.16, lw=0.5)
        ax.plot(x[99:], media, color=color, lw=2, label="entrenamiento, media móvil de 100 (con exploración)")
        ax.plot(ctrl[:, 0], ctrl[:, 1], color=t["texto"], lw=1, marker="o", ms=3.5,
                mfc=t["fondo"], mew=1, label="control voraz de 20 episodios (sin exploración)")
        ax.axhline(-110, color=t["tenue"], lw=1, ls=(0, (4, 3)))
        ax.annotate("umbral de resuelto: −110", (0, -110), xytext=(6, -12), textcoords="offset points",
                    ha="left", fontsize=8.5, color=t["texto2"])
        cx = res["punto_de_control_elegido"]
        cy = ctrl[ctrl[:, 0] == cx][0, 1]
        ax.plot([cx], [cy], marker="o", ms=9, mfc="none", mec=t["texto"], mew=1.6)
        ax.annotate(f"punto de control elegido\nevaluación final: {res['eval_media']:.1f}",
                    (cx, cy), xytext=(-14, 46 if agente == "qlearning" else 22), textcoords="offset points", ha="right", fontsize=8.5,
                    color=t["texto"], arrowprops={"arrowstyle": "-", "color": t["tenue"], "lw": 0.8})
        ax.set_title(f"{NOMBRES[agente]}  ·  {res['eval_media']:.1f} ± {res['eval_desviacion']:.1f}  ·  "
                     f"bandera en {res['eval_banderas']}/{res['eval_episodios']}")
        ax.set_xlabel("episodio de entrenamiento")
        ax.set_ylim(-204, -78)
        ax.set_xlim(0, len(hist) * 1.01)
    axes[0].set_ylabel("recompensa del episodio (= −pasos)")
    h = [Line2D([], [], color=t["texto2"], lw=2),
         Line2D([], [], color=t["texto"], lw=1, marker="o", ms=3.5, mfc=t["fondo"], mew=1)]
    lab = ["entrenamiento, media móvil de 100 (con exploración)", "control voraz de 20 episodios (sin exploración)"]
    fig.legend(h, lab, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    guardar(fig, "curvas_entrenamiento", tema)


def eficiencia(t: dict, tema: str) -> None:
    """Desempeno voraz contra experiencia consumida, en un eje comun."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for agente in ("qlearning", "dqn"):
        hist, ctrl, _ = cargar(agente)
        pasos = np.cumsum(-hist)[ctrl[:, 0].astype(int) - 1]
        color = t["serie"][agente]
        ax.plot(pasos, ctrl[:, 1], color=color, lw=2, marker="o", ms=4, mfc=t["fondo"], mew=1.2)
        i = int(np.argmax(ctrl[:, 1]))
        ax.annotate(NOMBRES[agente], (pasos[i], ctrl[i, 1]), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=10, color=t["texto"])
    ax.axhline(-110, color=t["tenue"], lw=1, ls=(0, (4, 3)))
    ax.set_xscale("log")
    ax.set_ylim(-204, -90)
    ax.annotate("umbral de resuelto: −110", (1e4, -110), xytext=(0, 5), textcoords="offset points",
                fontsize=8.5, color=t["texto2"])
    ax.set_xlabel("pasos de entorno consumidos (escala logarítmica)")
    ax.set_ylabel("media voraz en el control")
    ax.set_title("DQN supera el umbral con una fracción de la experiencia que consume la tabla")
    fig.tight_layout()
    guardar(fig, "eficiencia_en_muestras", tema)


def exploracion(t: dict, tema: str) -> None:
    filas = []
    for linea in (REG / "diagnostico_exploracion.txt").read_text().splitlines():
        m = re.match(r"\s*([\d.]+)\s*\|\s*(\d+)/(\d+)\s*\|\s*([\d.]+)", linea)
        if m:
            filas.append((float(m[1]), int(m[2]), int(m[3]), float(m[4])))
    p, banderas, total, racha = map(np.array, zip(*filas))
    fig, ax = plt.subplots(figsize=(9, 4.4))
    color = t["serie"]["dqn"]
    barras = ax.bar(range(len(p)), banderas, width=0.62, color=color)
    for b, n, r in zip(barras, banderas, racha):
        ax.annotate(f"{n}", (b.get_x() + b.get_width() / 2, n), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=10, color=t["texto"])
    ax.set_xticks(range(len(p)))
    ax.set_xticklabels([f"p = {v:.2f}\nracha ≈ {r:.1f} pasos" for v, r in zip(p, racha)])
    ax.annotate("epsilon-greedy de libro", (0, 0), xytext=(0, 26), textcoords="offset points", ha="center",
                fontsize=8.5, color=t["texto2"])
    ax.annotate("valor elegido", (4, banderas[4]), xytext=(0, 20), textcoords="offset points", ha="center",
                fontsize=8.5, color=t["texto2"])
    ax.set_ylim(0, banderas.max() * 1.3)
    ax.set_ylabel(f"episodios con bandera, de {total[0]}")
    ax.set_xlabel("probabilidad de repetir la acción exploratoria anterior")
    ax.set_title("La exploración uniforme nunca alcanza la bandera; la persistente sí")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    guardar(fig, "diagnostico_exploracion", tema)


def mapas(t: dict, tema: str, ags: dict) -> None:
    pos, vel, obs = rejilla_estados()
    ext = [pos[0], pos[-1], vel[0], vel[-1]]
    n = len(pos)

    # Politica voraz: que accion elige cada agente en cada estado, con una trayectoria encima.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    cmap = ListedColormap(t["acciones"])
    for ax, nombre in zip(axes, ("qlearning", "dqn")):
        q, visto = valores_q(ags[nombre], nombre, obs)
        accion = np.where(visto, q.argmax(1), np.nan).reshape(n, n)
        ax.set_facecolor(t["vacio"])
        ax.imshow(accion, origin="lower", extent=ext, aspect="auto", cmap=cmap, vmin=0, vmax=2,
                  interpolation="nearest")
        tr = trayectoria(ags[nombre])
        ax.plot(tr[:, 0], tr[:, 1], color=t["texto"], lw=1.6)
        ax.plot(*tr[0], marker="o", ms=7, mfc=t["fondo"], mec=t["texto"], mew=1.5)
        ax.annotate("inicio", tr[0], xytext=(8, -14), textcoords="offset points", fontsize=8.5, color=t["texto"])
        ax.axvline(0.5, color=t["texto"], lw=1, ls=(0, (2, 2)))
        ax.annotate("bandera", (0.5, vel[-1]), xytext=(-4, -12), textcoords="offset points", ha="right",
                    fontsize=8.5, color=t["texto"])
        ax.set_title(f"{NOMBRES[nombre]}  ·  un episodio voraz de {len(tr) - 1} pasos")
        ax.set_xlabel("posición")
        ax.grid(False)
    axes[0].set_ylabel("velocidad")
    marcas = [Patch(color=c, label=a) for c, a in zip(t["acciones"], ACCIONES)]
    marcas.append(Patch(facecolor=t["vacio"], edgecolor=t["eje"], label="celda nunca visitada (solo en la tabla)"))
    fig.legend(handles=marcas, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Política aprendida: empujar en el sentido del movimiento", x=0.01, ha="left",
                 fontsize=13)
    fig.tight_layout()
    guardar(fig, "mapa_politica", tema)

    # Costo restante: pasos que el agente estima que le faltan, -max_a Q(s, a).
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    rampa = LinearSegmentedColormap.from_list("rampa", t["rampa"])
    rampa.set_bad(t["vacio"])
    for ax, nombre in zip(axes, ("qlearning", "dqn")):
        q, visto = valores_q(ags[nombre], nombre, obs)
        costo = np.where(visto, -q.max(1), np.nan).reshape(n, n)
        im = ax.imshow(costo, origin="lower", extent=ext, aspect="auto", cmap=rampa, interpolation="nearest",
                       vmin=0, vmax=100)
        ax.axvline(0.5, color=t["texto"], lw=1, ls=(0, (2, 2)))
        ax.set_title(NOMBRES[nombre])
        ax.set_xlabel("posición")
        ax.grid(False)
    axes[0].set_ylabel("velocidad")
    barra = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    barra.set_label("costo restante estimado, −max Q(s, a)")
    barra.outline.set_visible(False)
    fig.suptitle("Función de valor: la tabla es un mosaico, la red es una superficie continua", x=0.01,
                 ha="left", fontsize=13)
    guardar(fig, "mapa_valor", tema)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    ags = agentes()
    for tema, t in TEMAS.items():
        estilo(t)
        curvas(t, tema)
        eficiencia(t, tema)
        exploracion(t, tema)
        mapas(t, tema, ags)
    print("Figuras guardadas en resultados/figuras/ (variantes _claro y _oscuro)")
