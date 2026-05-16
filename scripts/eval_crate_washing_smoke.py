"""Smoke test for the bimanual ``libero_crate_washing`` benchmark.

Loads the suite, instantiates an off-screen render env, optionally seeds the
state from a baked init row, and steps a small number of zero actions just
to confirm the wiring works end-to-end (BDDL parse, problem-class lookup,
arena load, two-Panda merge, OSC controllers, observation collection, the
stage-cursor success check). Stage progress is printed each iteration.

Run from the LIBERO-PRO repo root with offscreen rendering enabled:

    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \\
        python3 scripts/eval_crate_washing_smoke.py
"""

import argparse
import os
import sys

import numpy as np
import yaml

# Allow the script to import the LIBERO package even when LIBERO-PosVar is
# not pip-installed in the current interpreter, and point ``get_libero_path``
# at *this* checkout's asset / bddl / init dirs without clobbering any
# pre-existing ``~/.libero/config.yaml`` the user has for a different fork.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
for path in (_REPO_ROOT, os.path.join(_REPO_ROOT, "libero")):
    if path not in sys.path:
        sys.path.insert(0, path)

_LOCAL_CONFIG_DIR = os.path.join(_REPO_ROOT, ".libero_config")
_LOCAL_CONFIG_FILE = os.path.join(_LOCAL_CONFIG_DIR, "config.yaml")
_LIBERO_PKG_ROOT = os.path.join(_REPO_ROOT, "libero", "libero")
os.makedirs(_LOCAL_CONFIG_DIR, exist_ok=True)
if not os.path.exists(_LOCAL_CONFIG_FILE):
    with open(_LOCAL_CONFIG_FILE, "w") as _f:
        yaml.dump(
            {
                "benchmark_root": _LIBERO_PKG_ROOT,
                "bddl_files": os.path.join(_LIBERO_PKG_ROOT, "bddl_files"),
                "init_states": os.path.join(_LIBERO_PKG_ROOT, "init_files"),
                "datasets": os.path.join(_LIBERO_PKG_ROOT, "..", "datasets"),
                "assets": os.path.join(_LIBERO_PKG_ROOT, "assets"),
            },
            _f,
        )
os.environ.setdefault("LIBERO_CONFIG_PATH", _LOCAL_CONFIG_DIR)

try:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
except ModuleNotFoundError:
    from libero import benchmark, get_libero_path
    from libero.envs import OffScreenRenderEnv


BENCHMARK_NAME = "libero_crate_washing"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Number of zero-action env.step calls (default: %(default)s).",
    )
    parser.add_argument(
        "--use-init",
        action="store_true",
        help="If set, also load the first row from the baked init file.",
    )
    args = parser.parse_args()

    bench = benchmark.get_benchmark_dict()[BENCHMARK_NAME]()
    task = bench.tasks[0]
    bddl_path = bench.get_task_bddl_file_path(0)
    print(f"[smoke] benchmark={BENCHMARK_NAME}")
    print(f"[smoke] task={task.name}  language={task.language!r}")
    print(f"[smoke] bddl={bddl_path}")

    env = OffScreenRenderEnv(bddl_file_name=bddl_path)
    env.reset()

    if args.use_init:
        init_path = os.path.join(
            get_libero_path("init_states"),
            task.problem_folder,
            task.init_states_file,
        )
        if os.path.exists(init_path):
            import torch

            rows = torch.load(init_path, weights_only=False)
            env.set_init_state(np.asarray(rows[0]))
            print(f"[smoke] loaded init row 0 from {init_path}")
        else:
            print(f"[smoke] WARNING: --use-init requested but {init_path} missing")

    action_dim = env.env.action_dim
    print(f"[smoke] action_dim={action_dim}")
    print(
        f"[smoke] stage_idx (post-reset)={env.env.crate_stage_idx} "
        f"/ {env.env.crate_num_stages}"
    )

    noop = np.zeros(action_dim, dtype=np.float64)
    for step_i in range(args.max_steps):
        obs, reward, done, info = env.step(noop)
        if step_i == 0 or step_i == args.max_steps - 1:
            print(
                f"[smoke] step={step_i:03d} reward={reward:.3f} done={done} "
                f"stage_idx={env.env.crate_stage_idx}"
            )
        if done:
            print(f"[smoke] task succeeded at step {step_i}")
            break

    print(
        f"[smoke] final stage_idx={env.env.crate_stage_idx} "
        f"/ {env.env.crate_num_stages}  "
        f"check_success={env.env._check_success()}"
    )

    env.close()


if __name__ == "__main__":
    main()
