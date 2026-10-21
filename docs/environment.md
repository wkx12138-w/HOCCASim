# Environment specification

This document is the authoritative description of what `hocca_sim` simulates.
Every numbered statement maps to a section of the reference design; the section
in brackets is the source. Values that the design does not fix are marked
**RECONSTRUCTED** together with the reason — they are configuration fields, not
hidden constants.

---

## 1. Grid and coordinate system

| Item | Value | Source |
|---|---|---|
| Grid | `G x G = 48 x 48`, 2304 cells | Table 1 |
| Internal indexing | `0 .. 47` on both axes, `array[y, x]` | — |
| Display indexing | `1 .. 48` (cells are labelled from 1) | [Problem Formulation] |
| Boundary rule | `clamp` — the aircraft stops on the border line (**RECONSTRUCTED**: unspecified by the design) | — |

`hocca_sim.render.SceneStyle(cell_offset=1.0)` draws the 1-based convention used
by the figures; `cell_offset=0.0` reproduces the 0-based convention of
`figures/fig_setup.pdf`.

## 2. Targets [Eq. (occupancy), Target hop process]

* `K = 16` targets: `12` static and `4` moving.
* Initial placement: **uniform without replacement over grid cells**, so the
  initial cells are distinct.
* Static targets use the identity transition.
* Moving targets hop on the schedule `t % T_m == 0`, `T_m = 20` steps. At a hop
  a moving target stays in place with probability `1 - p_m` and otherwise moves
  uniformly to a valid eight-neighbour; `p_m = 0.25`. Boundary probabilities are
  renormalised automatically because the candidate list only contains in-grid
  cells.
* Two targets may occupy the same cell after a hop (`allow_target_overlap`).

## 3. Occupancy prior [Eq. (marginalbelief)]

For a uniform without-replacement prior the occupancy marginal is

```
b_c(0) = K / G^2 = 16 / 2304 = 0.006944   (NOT 0.5)
```

`logit(b_c(0)) = -4.9618` nats. This single number is the starting point of every
log-odds update and is asserted in `tests/test_simulator.py`.

## 4. UAVs [Table 1]

* `N = 4` by default; unit speed, i.e. **1 cell per step**.
* Heading lives on a 12-bin lattice (`30` degree resolution).
* Primitive action: a turn command `a_u(t) ∈ {-30°, 0°, +30°}`; the heading is
  re-quantised onto the lattice after every command, then the aircraft advances
  `speed` cells along the new heading in continuous coordinates.
* The launch poses of the reference scenario are the four corners
  `(4,4), (44,4), (4,44), (44,44)` and every aircraft faces the grid centre
  (**RECONSTRUCTED**: unspecified by the design).
* Energy: normalised budget, linear consumption with distance, disabled by
  default (`energy_per_step = 0`). (**RECONSTRUCTED**)

## 5. Sensing [Eq. (sensor), Eq. (likelihood)]

* Footprint `F_u(t) = {c : ||p_u(t) - c||_2 <= R_s}` with `R_s = 3`, a Euclidean
  disk rasterised onto the cell lattice — 29 cells in the interior.
* For every observed cell an independent binary measurement is drawn:

```
Pr(Z = 1 | X = 1) = p_D  = 0.90
Pr(Z = 1 | X = 0) = p_FA = 0.02
```

* Measurements are conditionally independent given the target state and their
  provenance identifiers.

## 6. Belief filter [Eq. (logodds), exact belief]

Two occupancy maps run in parallel:

`agent_beliefs[u]`
: UAV `u`'s own log-odds map, updated with **its own measurements** and with
  **delivered evidence items**. This is the only belief the decentralised policy
  and the observation use.
`team_belief`
: the full-sharing fusion of every measurement; it defines the potential `Φ`,
  the differential credit, and the entropy trace. The critics may use it during
  centralised training; actors may not.

Log-odds update (exactly Eq. (logodds)):

```
l_c <- l_c + z log(p_D/p_FA) + (1 - z) log((1-p_D)/(1-p_FA))
     = l_c + 3.8067        if z = 1
     = l_c - 2.2824        if z = 0
```

Confirmation: a cell is confirmed in agent `u`'s belief when
`l_{u,c} >= tau`, with `tau = 5.0` nats (**RECONSTRUCTED**;
this threshold is unspecified by the design). Three positive looks give
`-4.962 + 3 * 3.807 = 6.46 >= 5`; two give `2.65 < 5`, and three false alarms
occur with probability `8e-6`, so the threshold is the natural "3 detections"
rule.

The team's confirmed set is the **union over agents**, which is why the
communication budget matters for recall.

Motion predictor (**single-map approximation** of Eq. (ck)):

```
b-_c = (1-p_m) b_c + p_m * sum_{c' ~ c} b_{c'} / n_valid(c')
```

applied only on hop steps, clipped to `[0, 1]`. Total occupancy mass is exactly
conserved (asserted in the tests). An optional
`belief.mass_normalisation` renormalises the mass to `K` as a controller
heuristic.

## 7. Communication [Communication Model, Eq. (innovation), (voi), (select), (fusion)]

* Hard budget: `B = 2` packets per sender per step, `m = 8` evidence items per
  packet, so `Q = B m = 16` directed items per UAV-step. Since all-to-all would
  need `N - 1 = 3` packets, the budget binds.
* An item `e = (u, v, c, t_e, iota_e, delta_l_e)` carries a unique provenance
  identifier and the quantised likelihood innovation `delta_l_e ∈ {+3.8067,
  -2.2824}`.
* Sender-side score, Eq. (voi): `s_e = D_KL(Bern(b_c^{v,e}) || Bern(b_c^v))`,
  where `b_c^v` is the **acknowledged receiver belief** — the sender's own model
  of what the receiver knows (`comm.ack_ell[sender, receiver]`).
* Scheduler: the modular problem of Eq. (select) is solved exactly by the
  documented top-`Q` rule, because packets are independent per receiver: the
  content of packet `v` is the `m` highest-scoring items addressed to `v`, and
  the `B` packets are the `B` receivers with the largest positive totals. Items
  with non-positive net score `s_e - lambda_bit * c_e` are never sent
  (`bit_cost_lambda = 0.05`, **RECONSTRUCTED** — at `1.0` the radio is silent).
* Fusion, Eq. (fusion): every receiver applies each identifier at most once
  (`comm.applied_ids`), so evidence is never double counted.
* Retention: an item stays in the sender's ledger until it has been addressed to
  every receiver or is older than `replay_window = 8` steps (**RECONSTRUCTED**).
* Channel: an item sent at step `t` is fused at the beginning of step `t + 1`
  (this is Algorithm 1's order), plus `comm.delay_steps` extra hops. Optional
  impairments: `packet_loss`, fixed `delay_steps`, link `outage_period` /
  `outage_length`, and a per-step `bandwidth_schedule`.
* Application-layer accounting: 8-byte header + 16-byte item, 8-byte
  acknowledgement header + 8-byte identifiers. A saturated fleet of four
  therefore spends at most 416 application bytes per UAV-step; the simulator
  reports the realised total (`env.comm.total_bytes`).

## 8. Reward [Eq. (shaping), (marginal), (reward)]

```
Phi(xi) = -lambda_H * sum_c h(b_c)                (team belief, nats)
r_team  = r_task + gamma * Phi(xi') - Phi(xi)
r_i     = r_team / N + lambda_A * (G_i - mean_j G_j)
```

`r_task` is the weighted sum of the six named components plus the two
penalties. The full formula is not fixed by the design, so each normalisation is
an explicit flag:

| Term | Default form | Coefficient |
|---|---|---|
| confirmed detections | `lambda_D * (#new confirmed targets)` | 25 |
| coverage | `coverage_coef * (#newly sensed cells)` | 0.09 (**RECONSTRUCTED**) |
| revisits | `-lambda_R * repeats / looks` | 1.5 |
| communication | `-lambda_C * items / (N Q)` | 0.3 |
| energy | `-lambda_E * energy_fraction` | 0.02 |
| spatial balance | `-lambda_F * Var_u(new cells)/mean^2` | 0.2 |
| boundary | `-boundary_penalty * blocked / N` | 0.05 (**RECONSTRUCTED**) |
| collision | `-collision_penalty * pairs closer than 1 cell` | 0.5 (**RECONSTRUCTED**) |

A *revisit* is a look at a cell that was last sensed more than
`revisit_window = 12` steps ago; consecutive overlapping footprints of the same
sweep are not charged. `coverage_coef = 0.09` is chosen so that a successful
260-step rollout lands in the intended order of magnitude for the episode
return; `scripts/calibrate_reward.py` prints the live decomposition so the value
can be re-tuned for your own setup.

Differential credit: for a random permutation `sigma` of the fleet, the
same-step likelihood batches are assimilated sequentially on the team belief and
`G_{sigma(k)} = H~(q^{k-1}) - H~(q^k)`. Because likelihood products commute the
final posterior is order independent and `sum_u G_u = H~(q^-) - H~(q^+)`
(asserted in the tests).

## 9. Episode and metrics [Metrics and Statistical Protocol]

* Horizon `T = 260`.
* Coverage `C(t) = |union_{tau<=t} union_u F_u(tau)| / G^2` — footprints, never
  UAV centres.
* Recall `R_det(t) = K_conf(t) / K`; precision `TP / (TP + FP)` over confirmed
  cells; `T80_cov`, `T80_det`, `T_all`.
* Early termination as soon as every target is confirmed
  this keeps the reported `Moves` count consistent with the mean success time
  instead of always equalling `N` times the horizon.

## 10. Randomness

Each episode derives six independent streams from the episode seed
(`numpy.random.SeedSequence.spawn`): `reset`, `targets`, `sensing`, `channel`,
`credit`, `layout`. Re-running with the same seed reproduces the episode bit for
bit; `tests/test_simulator.py::test_determinism` asserts it.

---

## What this release contains

* The simulation environment described above, exactly as specified.
* The evaluation metrics, the reward decomposition and the figure renderer.
* A rule-based controller used as a runnable reference.

The learning components of HOCCA — the local encoder, the message-restricted
graph attention module, the option actor, the termination head and the
centralised critics — are not part of this release and are being prepared for a
later one. `docs/rl_interface.md` documents where they attach.
