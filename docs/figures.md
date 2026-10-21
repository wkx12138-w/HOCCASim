# Figure style

The figures follow one consistent visual language.  The renderer is
`hocca_sim.render`; the reference implementation it reproduces was validated
pixel by pixel (98.4 % of pixels within ±3/255, mean absolute difference 0.198).

## Palette

| Slot | Colour | Where |
|---|---|---|
| UAV-1..UAV-8 | `#c74247 #d38923 #3167bf #39865a #8754ab #159ba5 #9d6b3f #c55594` | case-study caption |
| Reference scene (4 slots) | `#c34450 #d29427 #3976b5 #408760` | stand-alone scene |
| Ground texture | `#edf0f2` family, blocky, seed 7 | everywhere |
| Target markers | `#263743` dark, `#ffffff` halo, amber `#c98a1a` for "encountered" | everywhere |
| Method curves | `#777777 #56B4E9 #E69F00 #009E73 #8C7BC9 #0067B1` | IQL, QMIX, MADDPG, MAPPO, DTH-MADDPG, HOCCA |
| Grid lines | `#bfc8cf` at 45 % | everywhere |
| Spines / ticks | `#7e8b95` / `#52616c` | everywhere |

## Components

* **Ground** — `make_background()`: a `24 x 24` field of block means in
  `[0.92, 0.96]` upsampled by 8, faint blue lift, 90 near-white speckles.  Fixed
  seed, so every run is identical.
* **Aircraft** — fixed-wing top view, `+x` is the nose: narrow fuselage, long
  main wing, tailplane, scaled by 0.80.  The cockpit is a short stroke in the
  UAV colour.
* **Sensing footprint** — dashed circle, radius equal to `R_s` in cells,
  `linestyle=(0, (3, 2))`, `linewidth=0.85`, `alpha=0.75`.
* **Trails** — full trajectory at `lw=1.25, alpha=0.93` plus a 10-step
  high-contrast tail at `lw=2.6, alpha=0.95`.
* **Targets** — circles (static) and squares (moving), `s=105`, white halo edge
  of 1.7 pt.
* **Frame** — 1-based cells, extent `0.5 .. 48.5`, ticks `1, 8, 16, 24, 32, 40,
  48`, `Grid x` / `Grid y` labels, white 2 pt inner border.
* **Header / footer** — bold left-aligned `UAV = 4 | t = 137` and centred
  `Coverage 78.8% | Confirmed 8/16`, plus a small `Illustrative scene` note.

## Background modes

| `SceneStyle.background_mode` | Appearance | Figure |
|---|---|---|
| `"texture"` | whole search area textured | `scene.png`, `fig_setup.png` |
| `"coverage"` | white page, textured **only where the fleet has sensed** | `case_study.png` |
| `"plain"` | flat `#edf0f2` | quick previews |

The case study's caption states it directly: *"Light-grey shading represents the
cumulative area sensed by the fleet, whereas white regions remain unobserved."*

## Generating the figures

```bash
python scripts/make_figures.py            # everything
python scripts/make_figures.py --only case
python scripts/render_scene.py --save figures/scene.png --step 137
python scripts/render_scene.py --animate  # or --gif figures/scene.gif
```

| Output | Content |
|---|---|
| `fig_setup.png` | 48 x 48 grid, four corner-launched UAVs with footprints, 12 static targets (circles), 4 moving targets (squares), dashed motion corridors |
| `scene.png` | one rollout frame with title and footer |
| `case_study.png` | 3 x 3: `N = 4, 6, 8` x `t = 80, 180, 260` on the same target realisation, with confirmed / encountered / unobserved markers |
| `coverage_recall.png` | footprint coverage and recall against environment step |
| `training_curves.png` | reward template — feed your own logs with `--reward-json` |

## Honesty rules built into the plots

* External numbers, if you enable them with `--guides`, are drawn only as
  labelled guides, never as data series.
* The rule-based controller is labelled `HOCCA (rule-based replica)`, not
  `HOCCA`.
* `figures/` is a demonstration output; the curves come from the example
  controller and are illustrative only.
* `make_figures.py` refuses to invent a training curve: without
  `--reward-json` it prints that the figure was skipped.
