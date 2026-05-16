"""Bake init states for the bimanual ``libero_crate_washing`` benchmark.

The crate-washing scene is fully deterministic — the only free body
(``crate_box_11``) starts at a fixed pose loaded from the MJCF home
keyframe and there is no LIBERO region-sampler randomisation involved.
A single saved init row is therefore enough, but we save ``N_TRIALS``
copies of the same flattened state so harnesses that iterate over rows
(see the popcorn ``README.md`` minimal eval pattern) can run multiple
trials and aggregate per-stage attrition exactly as they do for
single-task suites with per-trial variance.

Run from the LIBERO-PRO repo root:

    MUJOCO_GL=egl python3 scripts/bake_crate_washing_init.py
"""

import argparse
import os
import sys

import numpy as np
import torch
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
N_TRIALS = 20


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-trials",
        type=int,
        default=N_TRIALS,
        help="Number of init rows to save (default: %(default)s).",
    )
    parser.add_argument(
        "--task-index",
        type=int,
        default=0,
        help="Which task in the benchmark to bake (default: 0).",
    )
    args = parser.parse_args()

    bench = benchmark.get_benchmark_dict()[BENCHMARK_NAME]()
    task = bench.tasks[args.task_index]

    bddl_path = bench.get_task_bddl_file_path(args.task_index)
    print(f"[bake] using BDDL: {bddl_path}")

    env = OffScreenRenderEnv(bddl_file_name=bddl_path)
    env.reset()

    state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
    rows = torch.tensor(np.tile(state, (args.n_trials, 1)))

    out_dir = os.path.join(
        get_libero_path("init_states"), task.problem_folder
    )
    os.makedirs(out_dir, exist_ok=True)
    pruned_path = os.path.join(out_dir, task.init_states_file)
    init_path_ = os.path.join(
        out_dir, task.init_states_file.replace(".pruned_init", ".init")
    )

    torch.save(rows, pruned_path)
    torch.save(rows, init_path_)
    print(f"[bake] wrote {args.n_trials} rows to:")
    print(f"         {pruned_path}")
    print(f"         {init_path_}")

    env.close()


if __name__ == "__main__":
    main()
