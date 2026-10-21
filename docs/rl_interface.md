# Reinforcement-learning interface

`hocca_sim.HoCCAEnv` is a **parallel multi-agent** environment.  This page is
the complete contract; `examples/marl_template.py` is a runnable skeleton.

---

## 1. The step loop

```python
from hocca_sim import make_env

env = make_env(seed=0)                    # SimConfig() + reference parameters
obs, info = env.reset(seed=0)

while True:
    actions = policy(obs)                              # (N,) ints in {0,1,2}
    options = ...,                                     # optional (N,) ints or -1
    obs, rewards, terminated, truncated, info = env.step(actions, options=options)
    if terminated[0] or truncated[0]:
        break
```

`env.step` accepts either a bare array or a mapping `{agent_index: action}`.
`options` is optional bookkeeping for option-critic learners (see §5).

## 2. Actions

| Attribute | Value |
|---|---|
| `env.action_space` | `Discrete(3)` (one shared space for all UAVs) |
| meaning | `0 -> -30°`, `1 -> 0°`, `2 -> +30°` |
| `env.uav_cfg.turn_actions_deg` | `(-30.0, 0.0, 30.0)` |

Every action is legal in every state under the default boundary rule, so no
action mask is required.  If you enable a constrained scenario (blocked cells),
supply `action_mask_fn` to `RLPolicy` and mask the logits in your learner.

## 3. Observations

`obs` is a dict-like `Observation`:

| Key | Shape | Meaning |
|---|---|---|
| `obs["agents"]` | `(N, obs_dim)` float32 | the decentralised observation of every UAV (flat) |
| `obs["agent_dicts"][u]` | dict | the same observation split into named channels |
| `obs["state"]` | `(state_dim,)` float32 | the centralised training state (critics only) |
| `obs["step"]` | int | current environment step |

`obs_dim = W*W*C + 12 + N` with `W = observation.window = 15` and
`C = 3` channels:

```
channel 0  local occupancy belief (probability) over a W x W window
channel 1  the UAV's own sensed mask over the same window
channel 2  magnitude of the evidence it received this step, same window
scalars    [x/G, y/G, cos psi, sin psi, energy, option/(L-1),
            option_age/K_L, t/T, coverage, confirmed/K, belief at own cell,
            received_items/(m (N-1))]
neighbours  received item count from each sender, normalised by m
```

Ground-truth target positions are **never** part of an observation.  They are
available only through `env.evaluator` (`target_positions`, `target_cells`),
which exists for metrics and figures and must never be read by a policy.

`env.share_observation_space` (`env.state_space`) describes `obs["state"]`:
per-UAV scalars plus 12 x 12 block averages of the team belief and the visited
mask, plus eight global counters.

## 4. Rewards, terminations, truncations

* `rewards` — `(N,)` float array: the per-agent reward of Eq. (reward),
  `r_team/N + lambda_A (G_i - mean G)`.  `info["team_reward"]` is the joint
  reward whose sum the team actually optimises.
* `terminated` — `(N,)` bool: all targets confirmed (`terminate_on_all_confirmed`).
* `truncated` — `(N,)` bool: the 260-step horizon was reached.

`info` also carries `coverage`, `recall`, `precision`, `confirmed`, `entropy`,
`moves`, `packets_sent`, `items_attempted`, `messages_per_uav_step`,
`reward_breakdown`, `credits`, `option_expired` and a per-agent `agents` list.

## 5. Temporal abstraction (options)

The option-critic layer needs three things from the environment, and
only those three:

```python
env.option_space            # Discrete(4)   -- |Omega| = 4, unlabeled slots
env.option_age              # (N,) int      -- steps the current option has run
env.option_expired_mask()   # (N,) bool     -- True when age >= K_L = 8 or unset
```

Pass the option you are executing as `options=`; the environment tracks the age,
resets it when the option index changes, and forces a reselect at `K_L = 8`.
Option termination does **not** zero the TD target — only `terminated` does, as
specified in the design.

## 6. Centralised critics

`obs["state"]` is built for `Q_U(s, omega, a)` and `Q_Omega(s, omega)`.  It is
available during training only; do not feed it to an actor that claims
decentralised execution.  `env.state_dim` gives its size.

## 7. Spaces without Gymnasium

`hocca_sim.spaces` re-exports the real `gymnasium.spaces` classes when Gymnasium
is installed and otherwise provides duck-typed `Box`/`Discrete`/`MultiBinary`/
`Dict` with `sample`, `contains` and `shape`.  The simulator therefore runs on a
bare NumPy stack, and still registers as a first-class Gymnasium environment:

```python
from hocca_sim import register_with_gymnasium
register_with_gymnasium()
import gymnasium as gym
env = gym.make("HOCCA-CooperativeSearch-v0")
```

## 8. Adapters

| Adapter | Use |
|---|---|
| `hocca_sim.wrappers.GymnasiumMultiAgentWrapper` | dict-in / dict-out with `agent_0..N-1` names and a `state` key |
| `hocca_sim.wrappers.PettingZooParallelWrapper` | PettingZoo parallel API (`agents`, `observe`, `last`) |
| `hocca_sim.wrappers.SingleAgentVectorWrapper` | joint action for a quick single-policy PPO/SAC run (destroys decentralisation) |
| `hocca_sim.wrappers.EpisodeStatsWrapper` | return/length/metric logging |
| `hocca_sim.policies.RLPolicy` | wrap any object exposing `act(obs, mask=None) -> int` |

## 9. Writing a controller

```python
from hocca_sim.policies import ActionOutput, Policy, register_policy

@register_policy("mine")
class MyPolicy(Policy):
    uses_options = True

    def reset(self, env):
        ...                     # clear recurrent state, allocate buffers

    def act(self, env, observation):
        actions = ...           # (N,) ints
        options = ...           # (N,) ints, or None
        return ActionOutput(actions=actions, options=options)

    def observe(self, env, obs, actions, rewards, next_obs, done):
        ...                     # push the transition into your buffer
```

Then `python scripts/run_episode.py --policy mine`.  Import the module holding
the class before calling `get_policy`, or use `RLPolicy(agent=...)` directly.

## 10. Where HOCCA's own components attach

| Component | Environment hook | Your code |
|---|---|---|
| Local belief window | `obs["agent_dicts"][u]["local_belief"]` | encoder input |
| Received-packet features | `obs["agent_dicts"][u]["received"]`, `neighbour_items` | message GAT |
| VoI scheduler | already inside the environment (`comm.schedule`) | none — it is deterministic |
| Option actor / termination | `env.option_expired_mask()`, `options=` | `mu_i`, `beta_i` |
| Intra-option actor | `env.action_space` | `pi_omega` |
| Centralised critics | `obs["state"]` | `Q_U`, `Q_Omega` |
| Differential credit | `info["credits"]` (the realised `G_i` of one random permutation) | reward normalisation |

Because the scheduler, the provenance filter and the credit estimator are inside
the environment, an external learner gets *identical task semantics* no matter
which training algorithm is used — which is exactly the "shared inputs" clause of
the baseline protocol.
