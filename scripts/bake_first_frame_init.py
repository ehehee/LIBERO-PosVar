"""Bake first-frame init states for LIBERO variance suites.

Combines:
  * the suite-routing BDDL perturbation logic from
    `verifiable-OS/services/sim_bridge/env/loader.py:_perturb_bddl`, which
    picks the right `generate_*` perturbation for each variance suite, and
  * the per-task fanout / save pattern from
    `scripts/permute_distractor_init.py` (Ray remote per task, write to
    both `<task>.init` and `<task>.pruned_init`).

Unlike `permute_distractor_init.py`, this script does NOT step the
simulator at all — no `DUMMY_ACTION`, no settle loop, no tilt check.
Each saved row is exactly the state right after `env.reset()`, i.e. the
first frame the renderer would see. Objects therefore sit at the
sampler's small `z_offset` above the floor (mid-drop). Use this when
the consumer is fine with that — e.g. the runtime variance path where
the rollout's `num_steps_wait` absorbs the drop anyway.

Run from the LIBERO-PRO repo root:
    MUJOCO_GL=egl python scripts/bake_first_frame_init.py
    MUJOCO_GL=egl python scripts/bake_first_frame_init.py --benchmark libero_object_target_xy_variance
    MUJOCO_GL=egl python scripts/bake_first_frame_init.py --no-ray
"""

import argparse
import os
import random
import sys
import tempfile
import time

import init_path  # noqa: F401  -- prepends LIBERO-PRO repo root to sys.path

# The variance generators (generate_target_xy_variance.py, etc.) live
# alongside this script. Ray workers don't inherit the launcher's sys.path,
# so add the script's own dir explicitly here.
_GENERATORS_DIR = os.path.dirname(os.path.realpath(__file__))
if _GENERATORS_DIR not in sys.path:
    sys.path.insert(0, _GENERATORS_DIR)

import numpy as np
import ray
import torch

try:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
except ModuleNotFoundError:
    from libero import benchmark, get_libero_path
    from libero.envs import OffScreenRenderEnv


BENCHMARK_NAME = "libero_object_all_variance"
N_TRIALS = 50
SEED = 0
STAGGER_SECS = 1.0

# Mirrors loader.py:_is_variance_suite — these suites get runtime BDDL
# perturbation. For other suites the perturbation step is a no-op and
# variance comes purely from the per-trial seed driving LIBERO's region
# sampler.
VARIANCE_SUITES = {
    "libero_object_target_pos_var20x20",
    "libero_object_target_xy_variance",
    "libero_object_permutation",
    "libero_object_all_variance",
    "libero_object_target_basket_swap_variance",
    "libero_object_target_combined_variance",
    "libero_object_target_permutation_variance",
}


def perturb_bddl_for_suite(suite_name, content, rng):
    """Apply the same BDDL perturbation loader.py applies at runtime."""
    if suite_name in (
        "libero_object_target_pos_var20x20",
        "libero_object_target_xy_variance",
    ):
        from generate_target_xy_variance import perturb_bddl_content
        content, _ = perturb_bddl_content(content, rng)

    elif suite_name in (
        "libero_object_permutation",
        "libero_object_target_permutation_variance",
    ):
        from generate_target_permutation_variance import permute_bddl_content
        content, _ = permute_bddl_content(content, rng)

    elif suite_name == "libero_object_target_basket_swap_variance":
        from generate_target_basket_swap_variance import swap_basket_target
        content, _ = swap_basket_target(content, rng)

    elif suite_name == "libero_object_target_combined_variance":
        from generate_target_combined_variance import perturb_combined
        content, _ = perturb_combined(content, rng)

    elif suite_name == "libero_object_all_variance":
        from generate_all_variance import basket_swap
        from generate_target_xy_variance import perturb_bddl_content
        from generate_target_permutation_variance import permute_bddl_content
        content, _ = basket_swap(content)
        content, _ = perturb_bddl_content(content, rng)
        content, _ = permute_bddl_content(content, rng)

    return content


def _read_base_bddl(bench, task_idx):
    """Read the unperturbed source BDDL.

    Variance suites ship pre-perturbed BDDLs under their own folder, so
    re-perturbing those would compound the changes. loader.py works
    around this by reading the original from libero_object/<task>.bddl
    and we mirror that fallback here.
    """
    bddl_path = bench.get_task_bddl_file_path(task_idx)
    base_dir = os.path.join(get_libero_path("bddl_files"), "libero_object")
    base_path = os.path.join(base_dir, os.path.basename(bddl_path))
    if os.path.exists(base_path):
        with open(base_path) as f:
            return f.read()
    with open(bddl_path) as f:
        return f.read()


def _resolve_save_paths(task):
    """Return (init_path_out, pruned_path_out) under init_states/<folder>/."""
    folder = os.path.join(
        get_libero_path("init_states"), task.problem_folder
    )
    os.makedirs(folder, exist_ok=True)
    fname = task.init_states_file
    if fname.endswith(".pruned_init"):
        stem = fname[: -len(".pruned_init")]
    elif fname.endswith(".init"):
        stem = fname[: -len(".init")]
    else:
        stem = fname
    return (
        os.path.join(folder, stem + ".init"),
        os.path.join(folder, stem + ".pruned_init"),
    )


def bake_init_for_task(bench, suite_name, task_idx, base_seed, n_trials):
    task = bench.tasks[task_idx]
    init_path_out, pruned_path_out = _resolve_save_paths(task)

    base_content = _read_base_bddl(bench, task_idx)

    rows = None
    for t in range(n_trials):
        trial_seed = base_seed + t
        rng = random.Random(trial_seed)
        new_content = perturb_bddl_for_suite(suite_name, base_content, rng)

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".bddl", delete=False, dir="/tmp"
        )
        tmp.write(new_content)
        tmp.close()
        tmp_path = tmp.name

        env = None
        try:
            env = OffScreenRenderEnv(
                bddl_file_name=tmp_path, camera_heights=128, camera_widths=128
            )
            nq, nv = env.sim.model.nq, env.sim.model.nv
            row_len = 1 + nq + nv
            if rows is None:
                rows = np.zeros((n_trials, row_len), dtype=np.float64)

            env.seed(trial_seed)
            env.reset()
            # robosuite's hard_reset rebuilds the MjSim, so re-fetch.
            sim = env.sim

            # Pure first frame: no env.step, no sim.step, no settle.
            # Just snapshot whatever MuJoCo computed during reset() —
            # robot at home pose, objects at sampler placement (with the
            # small z_offset still above the floor).
            state = np.asarray(sim.get_state().flatten(), dtype=np.float64)
            rows[t] = state[:row_len].copy()
        finally:
            if env is not None:
                env.close()
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    n_unique = len(np.unique(rows, axis=0))
    print(f"[{task.name}] unique_rows={n_unique}/{n_trials}")

    torch.save(rows, pruned_path_out)
    torch.save(rows, init_path_out)
    return task.name


@ray.remote(num_cpus=1)
def _bake_task_remote(benchmark_name, task_idx, base_seed, n_trials):
    # Ray pins CUDA_VISIBLE_DEVICES="" on CPU-only workers, which crashes
    # robosuite's EGL selector. Drop the empty value and pin EGL device 0.
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

    try:
        from libero.libero import benchmark as _bench_mod
    except ModuleNotFoundError:
        from libero import benchmark as _bench_mod

    bench = _bench_mod.get_benchmark_dict()[benchmark_name]()
    return bake_init_for_task(bench, benchmark_name, task_idx, base_seed, n_trials)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default=BENCHMARK_NAME)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--n-trials", type=int, default=N_TRIALS)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--stagger", type=float, default=STAGGER_SECS)
    p.add_argument("--no-ray", action="store_true")
    args = p.parse_args()

    if args.benchmark not in VARIANCE_SUITES:
        print(
            f"warning: {args.benchmark} is not a known variance suite; "
            f"BDDL perturbation will be a no-op (per-trial seed only)"
        )

    bench = benchmark.get_benchmark_dict()[args.benchmark]()

    if args.no_ray:
        for i in range(bench.n_tasks):
            bake_init_for_task(
                bench, args.benchmark, i,
                args.seed + i * args.n_trials, args.n_trials,
            )
        return

    workers = args.workers if args.workers is not None else bench.n_tasks
    ray.init(ignore_reinit_error=True, num_cpus=workers)

    futures = []
    for i in range(bench.n_tasks):
        fut = _bake_task_remote.remote(
            args.benchmark, i,
            args.seed + i * args.n_trials, args.n_trials,
        )
        futures.append(fut)
        if i < bench.n_tasks - 1 and args.stagger > 0:
            time.sleep(args.stagger)

    pending = list(futures)
    done_count = 0
    while pending:
        done, pending = ray.wait(pending, num_returns=1)
        try:
            name = ray.get(done[0])
            done_count += 1
            print(f"  done {done_count}/{len(futures)}  {name}")
        except Exception as e:
            done_count += 1
            print(f"  FAIL {done_count}/{len(futures)}  {e}")


if __name__ == "__main__":
    main()
