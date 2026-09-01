# Exercises

Three exercises, in order. Each one runs on its own, and each one ends with a
command you can run to check yourself.

Everything outside the marked `EXERCISE` blocks is already written for you: the
CLI, the training loops, saving/loading, logging. You are filling in the parts
that *are* the algorithm.

```bash
uv sync
uv run mountaincar inspect        # look at the environment before you start
```

Before anything else, read the environment description in the README. The
reward structure is the single most important fact about this task, and
Exercise 3 will not make sense without it.

---

## Exercise 1 — Tabular Q-Learning

File: `src/mountain_car/agents/qlearning.py`

MountainCar's observation is 2 continuous numbers. A Q-table needs discrete
keys, so we chop each dimension into `n_bins` bins and treat each cell of the
resulting grid as one state.

| | what to write |
|---|---|
| **1a** `discretize` | continuous observation → a hashable table key |
| **1b** `select_action` | epsilon-greedy, with a `deterministic` escape hatch |
| **1c** `_update` | the Q-Learning TD update |

Each stub's docstring has the specifics and the tips. Some general ones:

- **1a** — `self._bins` is already built in `__init__`; you only need to use it.
  Ask yourself what happens to a value outside the bin edges, and whether that
  is a problem here.
- **1b** — `deterministic=True` must *never* explore. Evaluation and rendering
  both rely on this; if you ignore the flag, your agent will look worse than it
  is and the results will be noisy.
- **1c** — Write the target first, then the update. Keep them as two lines
  until it works; you can compress later.
- Watch the `States visited` counter in the training log. If it stays tiny,
  your `discretize` is collapsing distinct observations onto the same key. If
  it climbs to the thousands, it is too fine-grained. With `n_bins=20` and 2
  dimensions there are 400 possible cells, and a healthy run visits a few
  hundred of them.
- If the score never moves off `-200`, check in this order: is `epsilon`
  actually non-zero early on? does `deterministic=True` change the behaviour?
  is `_update` writing back into `self.q_table` rather than into a copy?

**Check yourself:**

```bash
uv run mountaincar train qlearning --episodes 20000
uv run mountaincar load qlearning --eval
```

Expect the average reward to sit at `-200` for a while and then start climbing.
A correct implementation reaches the flag in 10/10 evaluation episodes and
scores roughly `-133`. Then watch it drive:

```bash
uv run mountaincar render qlearning
```

---

## Exercise 2 — Deep Q-Network

File: `src/mountain_car/agents/dqn.py`

Same algorithm, but the Q-table becomes a neural network, which means no
discretisation and two new machines: a replay buffer (given) and a target
network (given).

| | what to write |
|---|---|
| **2a** `QNetwork` | the `state_dim → hidden → hidden → action_dim` MLP |
| **2b** `_learn` | the mini-batch Bellman update and gradient step |

For **2b**, the mini-batch is already sampled and converted to tensors for you.
Four things are yours: `current_q`, `next_q`, `target_q`, and the gradient step.
Things worth getting right:

- Keep track of tensor **shapes**. `self.q_net(states_t)` is
  `(batch, action_dim)`, but the loss compares one number per transition, so
  both sides should end up `(batch, 1)`. A silent broadcast from a stray shape
  is the most common bug here and it will not raise — it will just quietly
  train on nonsense.
- `next_q` must come from `self.target_net`, not `self.q_net`, and must not
  carry gradients. If you find yourself asking why there are two networks at
  all, try it with one and watch what happens to the loss.
- Read the training loop and notice what gets pushed into the replay buffer as
  the "done" flag. It is `terminated`, not `terminated or truncated`. Work out
  why that distinction matters in an environment with a step limit — this one
  is worth a minute of thought, because getting it wrong is subtle and quiet.
  (Clue: what would the target become for a transition marked done? And what
  fraction of early MountainCar episodes end by *timing out* rather than by
  reaching the flag?)
- Sanity check while debugging: print `current_q.shape` and `target_q.shape`.
  If they are not identical you have a broadcast bug, even though it runs.

**Check yourself:** a quick shape/plumbing test before spending real time —

```bash
uv run mountaincar train dqn --episodes 20
```

If that runs without raising, your shapes are at least consistent. Now train
properly:

```bash
uv run mountaincar train dqn --episodes 1000
```

...and read the next exercise, because this will not work.

---

## Exercise 3 — Why won't it learn?

File: `src/mountain_car/agents/dqn.py`, method `select_action`

With Exercise 2 correct, DQN on MountainCar reports a **completely flat score,
forever**. Not unstable, not slowly improving — flat, at exactly the worst
possible value, for as many episodes as you care to run.

The learning code is fine. `select_action` is textbook epsilon-greedy and it is
also fine, as *code*. Your job is to find out what is actually wrong, and the
point of this exercise is the diagnosis, not the patch.

### Work it out

Resist jumping straight to a fix. Gather evidence first -- then take as many of
the clues below as you need. They are deliberately ordered from gentle to
nearly-the-answer, so stop reading as soon as you have it.

1. **Confirm the learning code is not the problem.** You have a second
   environment available and the agent is not specialised to MountainCar:

   ```bash
   uv run python -c "
   from mountain_car.agents.dqn import DQNAgent
   DQNAgent('CartPole-v1', epsilon_decay=0.98).train(total_episodes=200, log_interval=25)"
   ```

   If that improves, your `_learn` and `QNetwork` are correct and the problem is
   specific to MountainCar. Keep this habit: before debugging an algorithm,
   check it against a task you know it can solve.

2. **Ask what the agent has actually seen.** Re-read the reward table in the
   README. Write down the sequence of rewards for an episode that does not
   reach the flag. Across a thousand such episodes, what is there for gradient
   descent to *distinguish*?

3. **Measure, don't assume.** How often does an agent behaving randomly reach
   the flag at all? Don't reason about it -- count it. Write a short loop that
   plays episodes with `env.action_space.sample()` and tallies how many
   *terminate* rather than hitting the step limit.

   > **What you should find:** zero. Not "rarely" -- **0 out of 300**. Sit with
   > that number for a moment, because it says the agent has never once
   > observed the event it is supposed to be learning to cause.

4. **Watch what random actually does.** Render an episode of random actions and
   watch the car, then compare it against what the car must do to get up the
   hill.

### Clues

<details>
<summary><b>Clue 1</b> — what does the successful driving pattern look like?</summary>

Read the first paragraph of the README's environment description again. The
engine is too weak to drive straight up, so the car has to *rock back and
forth* and build momentum -- push right while moving right, push left while
moving left, in long sustained runs. It is a pumping motion, like a child on a
swing.

Now look at your exploration and ask whether it can produce that shape of
behaviour at all.
</details>

<details>
<summary><b>Clue 2</b> — put a number on it</summary>

Suppose a sustained run of about 20 pushes in the same direction is what it
takes to escape the valley. Your exploration draws a fresh uniform action from
3 choices at every single step.

What is the probability of drawing the same action 20 times in a row?

    (1/3)^20  ~  3 in 10 billion

That is why the measurement in step 3 came back as exactly zero. The
exploration is not *unlucky*; the behaviour it needs is effectively impossible
for it to emit. Adding more episodes will never fix this.
</details>

<details>
<summary><b>Clue 3</b> — what the flat score is telling you</summary>

Every reward is `-1`, so if the agent never reaches the flag then *every* state
is worth the same thing: the discounted sum of `-1` forever, which for
`gamma=0.99` is `-1/(1-0.99) = -100`.

Check it. Print the network's Q-values for a batch of random states and look at
the **spread between the three actions within each state**:

```python
import numpy as np, torch, gymnasium as gym
env = gym.make("MountainCar-v0")
states = torch.as_tensor(
    np.array([env.observation_space.sample() for _ in range(200)]), dtype=torch.float32)
q = agent.q_net(states).detach().numpy()
print("mean Q:", q.mean(), "| spread across actions:", np.abs(q.max(1) - q.min(1)).mean())
```

Training a broken run and measuring gives:

| episodes | mean Q | spread across actions |
|---:|---:|---:|
| 250 | -22.3 | 0.013 |
| 750 | -53.0 | 0.004 |
| 1500 | -77.9 | 0.062 |

Two things to read off this. The mean is crawling toward that `-100` fixed
point, so the network is doing exactly what the Bellman equation asks. And the
spread is *approximately zero*: the network says all three actions are equally
good in every state.

Your network has learned correctly. It has learned that nothing it does
matters -- which, given the data it was shown, is true. The failure is upstream
of learning; it is in how the data is collected.
</details>

<details>
<summary><b>Clue 4</b> — the property you need (this one is nearly the answer)</summary>

Per-step uniform sampling makes consecutive actions statistically
**independent**, and independent zero-mean pushes average out to a random walk
that leaves the car sitting in the valley.

What you need is exploration whose consecutive actions are **temporally
correlated** -- where choosing an action makes you more likely to keep choosing
it, so that exploration produces sustained runs instead of jitter.

You do not need a fancy scheme. The simplest thing that has this property will
do, and it takes only a few lines plus a small piece of per-episode state.
</details>

5. **Now fix it.** Change how *exploratory* actions are chosen so the behaviour
   from Clue 1 becomes reachable. You are not changing the learning rule, the
   reward, or the environment -- only exploration.

### Notes on your fix

- Whatever you add, `deterministic=True` must still be pure greedy. Exploration
  belongs to training only, or your evaluation numbers are meaningless.
- If you introduce a new hyperparameter, add it to the class's `_HPARAMS` tuple
  so `save`/`load` round-trips it. Check with
  `uv run mountaincar load dqn` after a save.
- Reset any per-episode state you add at the top of each episode in `train`.

**Check yourself:**

```bash
uv run mountaincar train dqn --episodes 2500
uv run mountaincar load dqn --eval
```

A working fix reaches the flag in 10/10 evaluation episodes at roughly `-106`,
which beats your tabular agent from Exercise 1 and clears the conventional
"solved" bar of `-110`. Expect the *training* log to look worse than the
evaluation — if you understand your own fix, you can explain why.

---

## If you get stuck

Ask in this order:

1. Does it raise, or does it run and do nothing useful? Those are very
   different problems.
2. If it raises: print the shape of every tensor going into the loss.
3. If it runs and does nothing: what would the numbers look like if it *were*
   working? Compare that to what you see, and let the gap point you at which
   component to check.
4. Only then change something — one thing at a time, and re-run.
