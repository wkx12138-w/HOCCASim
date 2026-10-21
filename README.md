# HOCCA-Simulator

Reference simulation environment for **HOCCA**: Hierarchical Option-Critic
reinforcement learning with communication-constrained evidence sharing for
cooperative multi-UAV search.

The simulation environment is now open source. The learning algorithm (option
actors, message-restricted graph attention, centralised critics and the training
pipeline) will be released progressively in this repository.

```bash
python -m pip install numpy matplotlib
python scripts/run_episode.py
python scripts/make_figures.py
```

## Environment

| Item | Value |
|---|---|
| Grid | 48 × 48 = 2304 cells |
| Targets | 16 = 12 static + 4 moving |
| Target hop | `T_m = 20` steps, `p_m = 0.25`, eight neighbours, renormalised at the border |
| Fleet | `N = 4`, unit speed |
| Actions | `{-30°, 0°, +30°}` on 12 headings |
| Sensor | disk `R_s = 3`, `p_D = 0.90`, `p_FA = 0.02` |
| Prior occupancy | `b_c(0) = 16/2304` |
| Communication | `B = 2` messages × `m = 8` evidence items per UAV-step |
| Confirmation | cumulative log-odds `tau = 5` |
| Reward | `(1, 25, 1.5, 0.3, 0.02, 0.2)`, `gamma = 0.99` |
| Horizon | 260 steps |

Implementation covers the footprint sensing model, the log-odds occupancy
filter, provenance-tagged evidence with once-only fusion, the budgeted
value-of-information scheduler, potential shaping, differential information
credit, and the evaluation metrics (`C(t)`, `R_det(t)`, precision, `T80`,
moves, messages per UAV-step). Optional impairments: packet loss, fixed delay,
link outages and variable bandwidth.

## Controllers and RL interface

This repository is a **demonstration release**: it ships one example controller
so that the environment can be exercised end to end without training. The
episodes it produces are illustrative runs of the simulator, not benchmark
results.

```bash
python scripts/run_episode.py --policy rule --trace
python examples/rule_agent.py
```

Any reinforcement-learning algorithm can be attached:

```python
from hocca_sim import make_env
from hocca_sim.policies import RLPolicy

env = make_env(seed=0)
policy = RLPolicy(agent=my_marl_agent)

obs, info = env.reset(seed=0)
obs, rewards, terminated, truncated, info = env.step(actions, options=options)
```

Interfaces: `env.single_observation_space`, `env.state_space` (critics only),
`env.action_space`, `env.option_space`, `env.option_expired_mask()`, and
`info["credits"]`. Gymnasium, PettingZoo and single-agent adapters are in
`hocca_sim.wrappers`; see `docs/rl_interface.md` and
`examples/marl_template.py`.

## Figures

```bash
python scripts/make_figures.py
python scripts/render_scene.py --save figures/scene.png --step 137
python scripts/render_scene.py --animate
```

| Output | Content |
|---|---|
| `figures/fig_setup.png` | reference environment schematic |
| `figures/scene.png` | one rollout frame with the UAV/timestamp header |
| `figures/case_study.png` | `N = 4, 6, 8` × `t = 80, 180, 260` on one target realisation |
| `figures/coverage_recall.png` | coverage and recall against environment step |

The renderer reproduces the intended visual language: blocky light-grey ground,
fixed-wing top-view glyphs, dashed sensing footprints, the eight-colour UAV
ramp, and the cumulative sensed area shaded grey against a white unobserved
area. Style constants are listed in `docs/figures.md`.

Every figure in `figures/` is produced by the demo controller described above
and is included for illustration only.

## Layout

```
hocca_sim/      environment, beliefs, communication, rewards, metrics, renderer
configs/        reference_48x48.yaml
scripts/        run_episode, evaluate, make_figures, render_scene, calibrate_reward
examples/       random_agent, rule_agent, custom_policy, marl_template, gymnasium_usage
docs/           environment.md, rl_interface.md, figures.md, publish.md
tests/          35 unit tests
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Scope

This is a demonstration release. It contains the simulation environment, its
evaluation utilities and one example controller; it does not ship any trained
agent, any benchmark result or any learning curve. The learning components of
HOCCA are not part of it and are being prepared for a later release. Values that
the design leaves unspecified are configuration parameters with documented
defaults, listed in `docs/environment.md`.

## License

MIT.
