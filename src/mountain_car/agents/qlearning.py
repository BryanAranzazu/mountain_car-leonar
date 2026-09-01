"""
Tabular Q-Learning for MountainCar-v0.

MountainCar has a 2-D continuous observation (position, velocity) with hard
bounds published by the environment itself, so the whole state space can be
discretised into a simple `n_bins x n_bins` grid -- no hand-tuned bounds and
no special-casing needed.
"""
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Self

import gymnasium as gym
import numpy as np


class QLearningAgent:
    def __init__(
        self,
        env_id: str,
        *,
        n_bins: int = 20,
        lr: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.9995,
    ) -> None:
        self.env_id = env_id
        self.n_bins = n_bins
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.training_episodes = 0

        env = gym.make(env_id)
        low, high = env.observation_space.low, env.observation_space.high
        self.n_actions = int(env.action_space.n)  # type: ignore[attr-defined]
        env.close()

        # Bin edges per dimension (interior edges only, as np.digitize wants).
        self._bins = [np.linspace(lo, hi, n_bins + 1)[1:-1] for lo, hi in zip(low, high)]
        self.q_table: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(self.n_actions))

    # ── helpers ───────────────────────────────────────────────────────

    def discretize(self, obs: np.ndarray) -> tuple:
        """EXERCISE 1a: map a continuous observation to a discrete table key.

        `obs` is a 2-element array (position, velocity). `self._bins[i]` holds
        the bin edges for dimension i, already built for you in __init__.
        Return a hashable key -- a tuple of one bin index per dimension.

        Tip: np.digitize(value, edges) returns the index of the bin a value
        falls into. Tip: the key must be hashable, so build a tuple of ints.
        """
        raise NotImplementedError("EXERCISE 1a: implement discretize()")

    def select_action(self, state: tuple, *, deterministic: bool = False) -> int:
        """EXERCISE 1b: epsilon-greedy action selection.

        With probability `self.epsilon`, explore: pick a uniformly random
        action in [0, self.n_actions). Otherwise exploit: pick the action with
        the highest value in `self.q_table[state]`.

        When `deterministic` is True, never explore -- always exploit. That is
        the mode used for evaluation and rendering.

        Tip: self.q_table is a defaultdict, so indexing an unseen state is safe
        and returns a zero vector. Tip: np.argmax gives you the best action.
        """
        raise NotImplementedError("EXERCISE 1b: implement select_action()")

    def predict(self, obs: np.ndarray, *, deterministic: bool = True) -> tuple[int, None]:
        return self.select_action(self.discretize(obs), deterministic=deterministic), None

    # ── core RL ───────────────────────────────────────────────────────

    def _update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        terminated: bool,
    ) -> None:
        """EXERCISE 1c: the Q-Learning update.

        Move Q(state, action) toward the TD target:

            target   = reward + gamma * max_a' Q(next_state, a')
            Q(s, a) += lr * (target - Q(s, a))

        One catch: if `terminated` is True there is no next state to bootstrap
        from, so the target is just `reward`.

        Note that `terminated` is NOT the same as "the episode ended" -- see
        the training loop below for why that distinction matters here.
        """
        raise NotImplementedError("EXERCISE 1c: implement the Q-Learning update")

    def train(self, total_episodes: int = 10_000, log_interval: int = 100) -> list[float]:
        env = gym.make(self.env_id)
        rewards_history: list[float] = []

        for episode in range(1, total_episodes + 1):
            obs, _ = env.reset()
            state = self.discretize(obs)
            total_reward = 0.0
            done = False

            while not done:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                next_state = self.discretize(next_obs)
                self._update(state, action, float(reward), next_state, terminated)

                state = next_state
                total_reward += reward

            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            self.training_episodes += 1
            rewards_history.append(total_reward)

            if episode % log_interval == 0:
                avg = np.mean(rewards_history[-log_interval:])
                print(
                    f"Episode {episode}/{total_episodes} | "
                    f"Avg Reward: {avg:.2f} | "
                    f"Epsilon: {self.epsilon:.4f} | "
                    f"States visited: {len(self.q_table)}"
                )

        env.close()
        return rewards_history

    # ── persistence ───────────────────────────────────────────────────

    _HPARAMS = ("env_id", "n_bins", "lr", "gamma", "epsilon_end", "epsilon_decay")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: getattr(self, k) for k in self._HPARAMS}
        data["q_table"] = dict(self.q_table)
        data["epsilon"] = self.epsilon
        data["training_episodes"] = self.training_episodes
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved Q-Learning agent to {path}")

    @classmethod
    def load(cls, path: Path) -> Self:
        with open(path, "rb") as f:
            data = pickle.load(f)

        agent = cls(
            data["env_id"],
            epsilon_start=data["epsilon"],
            **{k: data[k] for k in cls._HPARAMS if k != "env_id"},
        )
        agent.q_table = defaultdict(lambda: np.zeros(agent.n_actions), data["q_table"])
        agent.training_episodes = data["training_episodes"]
        return agent

    def info(self) -> str:
        return (
            f"Q-Learning agent for {self.env_id}\n"
            f"  Episodes trained : {self.training_episodes}\n"
            f"  States visited   : {len(self.q_table)} / {self.n_bins ** 2}\n"
            f"  Epsilon          : {self.epsilon:.4f}\n"
            f"  LR / Gamma       : {self.lr} / {self.gamma}"
        )
