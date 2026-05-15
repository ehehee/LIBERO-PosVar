# libero_popcorn_production

Single-task multi-stage benchmark. The agent must, in order:

1. Place `chefmate_8_frypan_1` on `flat_stove_1`'s `cook_region`.
2. Turn the stove on (`Turnon flat_stove_1`).
3. Turn the stove off (`Turnoff flat_stove_1`).
4. Move the frypan off the stove (`Not (On chefmate_8_frypan_1 flat_stove_1_cook_region)`).

Stages 2 and 3 are mutually exclusive at any single frame, so the standard
LIBERO `_check_success` (flat `And` of predicates) can't represent the
task. The custom problem class
`libero/envs/problems/libero_kitchen_popcorn_production.py:Libero_Kitchen_Popcorn_Production`
walks the BDDL `(Sequence ...)` goal block with a per-episode cursor
that advances when each stage's predicate becomes true and never
regresses, so transient violations of an earlier stage don't disqualify
the trajectory.

## Checking success at evaluation time

The problem class exposes three things any rollout/eval harness can
read directly off the env:

| API | Returns |
| --- | --- |
| `env.env._check_success()` | `True` only after the cursor has walked **all** stages |
| `env.env.popcorn_stage_idx` | `0..num_stages` — how many stages have been observed so far (monotonic) |
| `env.env.popcorn_num_stages` | `4` |

The cursor is also wired into the standard LIBERO `done` flag —
`Libero_Kitchen_Tabletop_Manipulation._post_action` already does
`done = self._check_success()`, so any rollout loop that watches `done`
terminates on full success without changes.

### Minimal eval pattern

```python
import os, sys, numpy as np, torch
sys.path.insert(0, "/path/to/LIBERO-PosVar")
sys.path.insert(0, "/path/to/LIBERO-PosVar/libero")

from libero import benchmark, get_libero_path
from libero.envs import OffScreenRenderEnv

bench = benchmark.get_benchmark_dict()["libero_popcorn_production"]()
task = bench.tasks[0]
init_rows = torch.load(
    os.path.join(get_libero_path("init_states"),
                 task.problem_folder, task.init_states_file),
    weights_only=False,
)
env = OffScreenRenderEnv(bddl_file_name=bench.get_task_bddl_file_path(0))

MAX_STEPS = 600
final_stage = np.zeros(len(init_rows), dtype=int)   # max stage reached per trial
success     = np.zeros(len(init_rows), dtype=bool)

for t in range(len(init_rows)):
    env.reset()
    env.set_init_state(init_rows[t])
    for _ in range(MAX_STEPS):
        action = policy(obs)            # 7-D OSC_POSE delta + gripper
        obs, _, done, _ = env.step(action)
        if done:                        # full success: all stages observed
            break
    final_stage[t] = env.env.popcorn_stage_idx
    success[t]    = env.env._check_success()

print(f"full success rate: {success.mean():.1%}  ({success.sum()}/{len(success)})")
for s in range(env.env.popcorn_num_stages + 1):
    reached = (final_stage >= s).mean()
    print(f"  reached stage >= {s}: {reached:.1%}")
```

The per-stage attrition table is usually more useful than the single
fully-complete-or-not number — it tells you *where* policies fail
(e.g. "70% land the frypan on the stove, 55% turn it on, 40% turn it
off, 35% complete everything").

### Optional: surface stage progress through `info`

If you want the stage cursor in your training/logging pipeline without
reaching into `env.env`, add a 4-line `_post_action` override to
`libero_kitchen_popcorn_production.py`:

```python
def _post_action(self, action):
    reward, done, info = super()._post_action(action)
    info["popcorn_stage_idx"] = self._popcorn_stage_idx
    info["popcorn_num_stages"] = len(self._popcorn_stages)
    return reward, done, info
```

Then every gym-style `obs, reward, done, info = env.step(action)` yields
the cursor in `info["popcorn_stage_idx"]`, so any harness that already
records `info` (training, video overlay, etc.) picks it up
automatically.

## Files

| | |
| --- | --- |
| BDDL | `libero/libero/bddl_files/libero_popcorn_production/KITCHEN_SCENE9_popcorn_production.bddl` |
| Problem class | `libero/libero/envs/problems/libero_kitchen_popcorn_production.py` |
| Init states (50 trials) | `libero/libero/init_files/libero_popcorn_production/KITCHEN_SCENE9_popcorn_production.{init,pruned_init}` |
| Bake driver | `scripts/bake_first_frame_init.py --benchmark libero_popcorn_production --variance popcorn` |
| Variance generator | `scripts/generate_stove_pot_xy_variance.py` (used with `table_only=True`) |
