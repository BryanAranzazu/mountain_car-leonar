"""Ejercicio 3, paso 3: medir en vez de suponer.

Cuenta cuantos episodios alcanzan la bandera con dos formas de explorar:
  - uniforme: una accion nueva e independiente en cada paso (epsilon-greedy clasico)
  - persistente: se repite la accion anterior con probabilidad p
"""
import argparse

import gymnasium as gym
import numpy as np


def contar(p_repetir: float, episodios: int, seed: int) -> tuple[int, float]:
    rng = np.random.default_rng(seed)
    env = gym.make("MountainCar-v0")
    env.reset(seed=seed)
    banderas, rachas = 0, []
    for _ in range(episodios):
        env.reset()
        accion, racha, fin = int(rng.integers(3)), 1, False
        while not fin:
            if rng.random() >= p_repetir:
                nueva = int(rng.integers(3))
                if nueva != accion:
                    rachas.append(racha)
                    racha = 0
                accion = nueva
            racha += 1
            _, _, terminated, truncated, _ = env.step(accion)
            fin = terminated or truncated
        banderas += int(terminated)
    env.close()
    return banderas, float(np.mean(rachas))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    print(f"{'p_repetir':>10} | {'banderas':>10} | racha media (pasos)")
    for p in (0.0, 0.5, 0.8, 0.9, 0.95, 0.98):
        b, r = contar(p, args.episodes, args.seed)
        print(f"{p:>10.2f} | {b:>4}/{args.episodes:<5} | {r:6.1f}")
