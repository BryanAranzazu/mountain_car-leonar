import argparse
from importlib.metadata import version
from pathlib import Path

import gymnasium as gym
import numpy as np

from mountain_car.agents import DQNAgent, QLearningAgent

ENV_ID = "MountainCar-v0"
SAVE_DIR = Path("saves")
VERSION = version("mountain_car")

# agent name -> (class, save file)
AGENTS = {
    "qlearning": (QLearningAgent, SAVE_DIR / "qlearning_mountaincar.pkl"),
    "dqn": (DQNAgent, SAVE_DIR / "dqn_mountaincar.pt"),
}

ACTION_NAMES = {0: "push left", 1: "no push", 2: "push right"}


def _resolve(name: str):
    """Return (agent_class, save_path) for an agent name."""
    return AGENTS[name]


def _load(name: str):
    """Load a saved agent, or return None (after printing) if there is none."""
    cls, path = _resolve(name)
    if not path.exists():
        print(f"No save found at {path}")
        return None
    return cls.load(path)


def _run_episode(env, agent) -> tuple[float, int, bool]:
    """Play one greedy episode. Returns (total reward, steps, reached_flag)."""
    obs, _ = env.reset()
    total, steps, terminated = 0.0, 0, False
    while True:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        total += reward
        steps += 1
        if terminated or truncated:
            return total, steps, terminated


# ── commands ─────────────────────────────────────────────────────────


def cmd_inspect(args: argparse.Namespace) -> None:
    env_id = args.env or ENV_ID
    env = gym.make(env_id)

    print(f"Environment: {env_id}\n")
    print(f"Observation space : {env.observation_space}")
    print(f"  shape           : {env.observation_space.shape}")
    print(f"  low             : {env.observation_space.low}")
    print(f"  high            : {env.observation_space.high}")
    print(f"\nAction space      : {env.action_space}")
    print(f"  n actions       : {env.action_space.n}")
    print(f"Max episode steps : {env.spec.max_episode_steps if env.spec else 'N/A'}")

    print(f"\n-- Sample transitions ({args.steps} steps, random policy) --\n")
    obs, _ = env.reset()
    print(f"  Initial state: {np.array2string(obs, precision=3)}\n")

    for step in range(1, args.steps + 1):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        print(f"  step {step:>3} | action={action} | reward={reward:+.3f} | done={done}")
        print(f"           state -> {np.array2string(obs, precision=3)}")
        if done:
            obs, _ = env.reset()
            print("           [episode ended, resetting]")
            print(f"           state -> {np.array2string(obs, precision=3)}")
        print()

    env.close()


def cmd_init(args: argparse.Namespace) -> None:
    cls, path = _resolve(args.agent)
    if path.exists():
        print(f"Save already exists at {path}. Run 'mountaincar delete {args.agent}' first.")
        return
    cls(ENV_ID).save(path)
    print(f"Initialized {args.agent} agent.")


def cmd_train(args: argparse.Namespace) -> None:
    cls, path = _resolve(args.agent)
    agent = cls.load(path) if path.exists() else cls(ENV_ID)
    agent.train(total_episodes=args.episodes)
    agent.save(path)
    print("Training complete.")


def cmd_delete(args: argparse.Namespace) -> None:
    _, path = _resolve(args.agent)
    if path.exists():
        path.unlink()
        print(f"Deleted {path}")
    else:
        print(f"No save found at {path}")


def cmd_load(args: argparse.Namespace) -> None:
    agent = _load(args.agent)
    if agent is None:
        return
    print(agent.info())

    if args.eval:
        print("\nEvaluating (10 episodes) ...")
        env = gym.make(ENV_ID)
        results = [_run_episode(env, agent) for _ in range(10)]
        env.close()
        rewards = [r for r, _, _ in results]
        solved = sum(reached for _, _, reached in results)
        print(f"  Mean reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
        print(f"  Reached the flag: {solved}/10 episodes")


def cmd_sim(args: argparse.Namespace) -> None:
    agent = _load(args.agent)
    if agent is None:
        return

    env = gym.make(ENV_ID)
    all_rewards: list[float] = []

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        total_reward, step = 0.0, 0
        terminated = truncated = False

        print(f"== Episode {ep}/{args.episodes} ==\n")
        print(f"  initial state: {np.array2string(obs, precision=3)}\n")

        while not (terminated or truncated):
            step += 1
            action, _ = agent.predict(obs, deterministic=True)
            action = int(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            # args.steps is None -> show every step
            if args.steps is None or step <= args.steps:
                name = ACTION_NAMES.get(action, "?")
                print(
                    f"  step {step:>4} | action={f'{action} ({name})':>16} | "
                    f"reward={reward:+8.3f} | total={total_reward:+9.2f}"
                )
                if args.verbose:
                    print(f"           state -> {np.array2string(obs, precision=3)}")

        if args.steps is not None and step > args.steps:
            print(f"  ... ({step - args.steps} more steps) ...")

        outcome = "REACHED THE FLAG" if terminated else "TIMED OUT (step limit)"
        print(f"\n  Result: {outcome} | Steps: {step} | Total reward: {total_reward:+.2f}\n")
        all_rewards.append(total_reward)

    env.close()

    if len(all_rewards) > 1:
        print(
            f"Summary over {len(all_rewards)} episodes: "
            f"mean={np.mean(all_rewards):+.2f} +/- {np.std(all_rewards):.2f}"
        )


def cmd_render(args: argparse.Namespace) -> None:
    agent = _load(args.agent)
    if agent is None:
        return

    env = gym.make(ENV_ID, render_mode="human")
    for ep in range(1, args.episodes + 1):
        total, steps, reached = _run_episode(env, agent)
        status = "reached the flag" if reached else "timed out"
        print(f"Episode {ep}/{args.episodes} | Reward: {total:.2f} | Steps: {steps} | {status}")
    env.close()


def cmd_version(_args: argparse.Namespace) -> None:
    print(f"mountain_car {VERSION}")


def cmd_list(_args: argparse.Namespace) -> None:
    print("Available agents:\n")
    for name, (_, path) in AGENTS.items():
        status = "saved" if path.exists() else "no save"
        print(f"  {name:<14} [{status}]  {path}")


# ── argument parser ──────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mountaincar",
        description="Train and evaluate RL agents on MountainCar-v0",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help_: str, func) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_)
        p.set_defaults(func=func)
        return p

    def add_agent(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("agent", choices=tuple(AGENTS))
        return p

    add("version", "Show the package version", cmd_version)
    add("list", "List available agents and their save status", cmd_list)

    p = add("inspect", "Inspect an environment: spaces and sample transitions", cmd_inspect)
    p.add_argument("--env", default=None, help=f"Gymnasium env ID (default: {ENV_ID})")
    p.add_argument("--steps", type=int, default=5, help="Random steps to sample (default: 5)")

    add_agent(add("init", "Initialize a new (untrained) agent and save it", cmd_init))
    add_agent(add("delete", "Delete a saved agent", cmd_delete))

    p = add_agent(add("train", "Train an agent and save the result", cmd_train))
    p.add_argument("--episodes", type=int, default=10_000, help="Training episodes (default: 10k)")

    p = add_agent(add("load", "Load a saved agent and display info", cmd_load))
    p.add_argument("--eval", action="store_true", help="Run a quick 10-episode evaluation")

    p = add_agent(add("sim", "Simulate episodes with a trained agent (text output)", cmd_sim))
    p.add_argument("--episodes", type=int, default=1, help="Episodes to simulate (default: 1)")
    p.add_argument("--steps", type=int, default=None, help="Only print the first N steps per episode")
    p.add_argument("--verbose", action="store_true", help="Also print the state vector each step")

    p = add_agent(add("render", "Render episodes using a saved agent (graphical window)", cmd_render))
    p.add_argument("--episodes", type=int, default=1, help="Episodes to render (default: 1)")

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    args.func(args)
