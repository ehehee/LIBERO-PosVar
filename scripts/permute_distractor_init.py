"""Bake per-trial object permutations into the .init / .pruned_init files
using the SAME placement path the runtime uses for variance suites.

Why a rewrite: the previous version edited qpos directly, swapping each
distractor's xy to another slot while keeping its own z+quat. That bakes
the *old* contact height into a *new* slot, so MuJoCo ejects/drops the
object at frame 0. The runtime variance path doesn't touch qpos — it
perturbs the BDDL `(On obj region)` clauses and lets LIBERO's
ObjectPositionSampler.sample(on_top=True) compute a collision-resolved,
physics-aware z per object on its assigned region.

This script does the same thing offline:
  1. Read the original BDDL.
  2. Pick the permutation set (target stays; basket optional) from the
     `(On obj region)` clauses in `(:init)`.
  3. Generate N_TRIALS unique permutations of distractor → region.
  4. For each permutation: write a temp BDDL with the rewritten `(On ...)`
     lines, build OffScreenRenderEnv, env.reset(), and capture
     env.sim.get_state().flatten() as one baked row. Each row is a
     valid, settled MuJoCo state — no z-from-old-slot leakage.
  5. Save the (N_TRIALS, row_len) array to both <task>.init and
     <task>.pruned_init.

Tasks fan out over Ray (`@ray.remote`, mirroring the pattern in
`tools/visualize_skills.py`). One Ray task per LIBERO task → all
`bench.n_tasks` bake in parallel. Ray task launches are staggered by
`--stagger` seconds (default 1.0) so concurrent EGL/MuJoCo
initialisations don't stampede the GPU — same lesson as
`vos/compose/parallel.py`.

Run from the LIBERO-PRO repo root:
    MUJOCO_GL=egl python scripts/permute_distractor_init.py
    MUJOCO_GL=egl python scripts/permute_distractor_init.py --workers 10
    MUJOCO_GL=egl python scripts/permute_distractor_init.py --no-ray  # serial
"""

import argparse
import math
import os
import re
import tempfile
import time

import init_path  # noqa: F401  -- prepends repo root to sys.path

import numpy as np
import ray
import torch

try:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
except ModuleNotFoundError:
    # In some installs the inner libero/libero/ package is installed as the
    # top-level `libero` module, so `libero.libero` doesn't exist.
    from libero import benchmark, get_libero_path
    from libero.envs import OffScreenRenderEnv


BENCHMARK_NAME = "libero_object_all_variance"
N_TRIALS = 50
SEED = 0
STAGGER_SECS = 1.0  # gap between Ray task submissions to avoid EGL stampede

# Object-settle params. After env.reset() LIBERO's sampler places objects
# with a small `z_offset` above the floor, so frame 0 is mid-drop. Runtime
# rollouts absorb this by running `num_steps_wait` dummy env.step() calls
# at the start of each episode (openpi's main.py does ~10). We mirror that
# here: drive env.step(DUMMY_ACTION) so the OSC controller gravity-comps
# the robot while objects fall onto the floor, then early-stop on qvel.
# This is more reliable than raw sim.step() + manual robot pinning, which
# lets the gripper droop a fraction during each step before being restored
# — enough to punt nearby distractors away.
DUMMY_ACTION = [0.0] * 6 + [-1.0]  # 6-DOF OSC_POSE delta + gripper open
MAX_SETTLE_STEPS = 300             # env.steps (≈ 25 sim sub-steps each)
MIN_SETTLE_STEPS = 10              # match openpi num_steps_wait floor
SETTLE_QVEL_TOL = 1e-3
# Require N consecutive sub-tolerance readings before declaring "settled".
# A single check can fire at the apex of a bounce, where vz=0 momentarily.
SETTLE_CONSEC = 5
# Velocity-only convergence is not enough: a bottle that lands leaning on a
# neighbor reaches qvel=0 (friction) but its local +z is well off world +z.
# Reject any settled state whose worst free-joint object exceeds this tilt.
SETTLE_TILT_DEG = 20.0
# If settle doesn't converge, re-seed and try the same permutation again
# — sometimes the sampler picks an unlucky in-region (x,y) where an object
# perches on a neighbor instead of finding its way to the floor.
SETTLE_MAX_RETRIES = 3
# Some permutations are *structurally* unrecoverable: every (x,y) inside the
# assigned region lands the bottle leaning on the basket wall (or a tall
# neighbor). Re-seeding the sampler doesn't help — we have to draw a
# different permutation. We pre-generate N_TRIALS * PERM_POOL_MULT unique
# permutations (capped by n_objects!) and pop a fresh one each time the
# current one fails all SETTLE_MAX_RETRIES placement seeds.
PERM_POOL_MULT = 4

_ON_RE = re.compile(r"\(On\s+(\S+)\s+(\S+)\)")
_OBJ_OF_INTEREST_RE = re.compile(r"\(:obj_of_interest\s+([^)]*)\)")


def parse_obj_of_interest(content):
    m = _OBJ_OF_INTEREST_RE.search(content)
    if m is None:
        raise RuntimeError("no :obj_of_interest block")
    return m.group(1).split()


def identify_permutables(content, include_basket):
    """Return (target, basket, permute_objs, permute_regions).

    `permute_objs[i]` originally sits on `permute_regions[i]`. The target
    is always excluded; the basket is included only if `include_basket`.
    All `(On ...)` clauses live inside `(:init)` in LIBERO BDDLs, so a
    flat regex pass is safe — `(:goal)` uses `In`/`And`, not `On`.
    """
    of_interest = parse_obj_of_interest(content)
    target = next(o for o in of_interest if not o.startswith("basket"))
    basket = next((o for o in of_interest if o.startswith("basket")), None)

    permute_objs, permute_regions = [], []
    for obj, region in _ON_RE.findall(content):
        if obj == target:
            continue
        if obj == basket and not include_basket:
            continue
        permute_objs.append(obj)
        permute_regions.append(region)
    return target, basket, permute_objs, permute_regions


def perturb_bddl_permutation(content, permute_objs, permute_regions, perm_idx):
    """Rewrite (On obj region) clauses so permute_objs[i] lands on
    permute_regions[perm_idx[i]]. Other (On ...) clauses are untouched.
    """
    new_obj_to_region = {
        permute_objs[i]: permute_regions[perm_idx[i]]
        for i in range(len(permute_objs))
    }

    def _sub(match):
        obj = match.group(1)
        if obj in new_obj_to_region:
            return f"(On {obj} {new_obj_to_region[obj]})"
        return match.group(0)

    return _ON_RE.sub(_sub, content)


def unique_permutations(n_items, n, rng):
    """Yield n distinct permutations of range(n_items) (n <= n_items!)."""
    seen = set()
    out = []
    while len(out) < n:
        perm = tuple(rng.permutation(n_items).tolist())
        if perm in seen:
            continue
        seen.add(perm)
        out.append(perm)
    return out


def _read_quats(qpos, quat_adrs):
    """Snapshot each free-joint object's quaternion (wxyz) into a list."""
    return [
        (
            float(qpos[a]),
            float(qpos[a + 1]),
            float(qpos[a + 2]),
            float(qpos[a + 3]),
        )
        for a in quat_adrs
    ]


def _max_tilt_delta_deg(now_quats, init_quats):
    """Largest tilt (deg) any object accumulated relative to its init pose.

    LIBERO's placement sampler stands each object up in its canonical pose
    plus a random yaw, so init_quats are already "upright + yaw". A pure
    yaw rotation between init and now should NOT count as tilt; only
    rotations whose axis has a horizontal component should.

    We compute q_delta = q_now * q_init^{-1} (left-multiplied delta in the
    world frame) and ask how much q_delta rotates world +z away from
    itself. Pure yaw leaves +z fixed (tilt = 0); tipping over rotates +z
    away (tilt up to 180). For a unit quaternion q_delta = (w, x, y, z),
    R(q_delta) @ (0,0,1) has its z-component equal to 1 - 2(x^2 + y^2),
    so tilt = arccos(1 - 2(x^2 + y^2)).
    """
    if not now_quats or not init_quats:
        return 0.0
    max_deg = 0.0
    for (wn, xn, yn, zn), (wi, xi, yi, zi) in zip(now_quats, init_quats):
        # q_init^{-1} = conjugate for unit quat: (wi, -xi, -yi, -zi).
        # q_delta = q_now * q_init_conj  (Hamilton product).
        wd = wn * wi + xn * xi + yn * yi + zn * zi
        xd = -wn * xi + xn * wi - yn * zi + zn * yi
        yd = -wn * yi + xn * zi + yn * wi - zn * xi
        # zd is unused: tilt depends only on xd, yd.
        zz = 1.0 - 2.0 * (xd * xd + yd * yd)
        if zz > 1.0:
            zz = 1.0
        elif zz < -1.0:
            zz = -1.0
        deg = float(np.degrees(np.arccos(zz)))
        if deg > max_deg:
            max_deg = deg
        # Suppress unused-variable warning while keeping the math obvious.
        _ = wd
    return max_deg


def bake_init_for_task(bench, task_idx, seed, include_basket):
    task = bench.tasks[task_idx]
    bddl_path = bench.get_task_bddl_file_path(task_idx)
    pruned_path = os.path.join(
        get_libero_path("init_states"), task.problem_folder, task.init_states_file
    )
    init_path_out = pruned_path.replace(".pruned_init", ".init")

    with open(bddl_path) as f:
        base_content = f.read()

    target, basket, permute_objs, permute_regions = identify_permutables(
        base_content, include_basket
    )
    expected = 6 if include_basket else 5
    if len(permute_objs) != expected:
        raise RuntimeError(
            f"[{task.name}] expected {expected} permutable objects, got {permute_objs}"
        )

    # Match the row length used by existing .init files: [time(1), qpos, qvel].
    # We allocate from the first env we build below.
    rng = np.random.default_rng(seed)
    # Some permutations are structurally bad (every (x,y) inside the region
    # leans the bottle on a wall). Pre-generate a larger pool — capped by
    # n_objects! — and pop a fresh perm whenever the current one fails all
    # placement retries. Pool order is already randomized inside
    # unique_permutations, so we just iterate.
    n_perms_max = math.factorial(len(permute_objs))
    n_pool = min(N_TRIALS * PERM_POOL_MULT, n_perms_max)
    perm_pool = unique_permutations(len(permute_objs), n_pool, rng)
    perm_iter = iter(perm_pool)

    rows = None
    perms_used = []
    t = 0
    while t < N_TRIALS:
        try:
            perm_idx = next(perm_iter)
        except StopIteration:
            raise RuntimeError(
                f"[{task.name}] exhausted permutation pool of {n_pool} "
                f"while filling slot {t}/{N_TRIALS}; raise PERM_POOL_MULT "
                f"(or relax SETTLE_TILT_DEG / increase SETTLE_MAX_RETRIES)"
            )

        new_content = perturb_bddl_permutation(
            base_content, permute_objs, permute_regions, perm_idx
        )
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

            # Read nq/nv from the constructor-built sim so we can size `rows`,
            # but DO NOT cache the sim handle across env.reset(): robosuite
            # has hard_reset=True by default, so each reset destroys the old
            # MjSim and rebinds env.sim to a fresh one. We always re-fetch
            # `sim = env.sim` after `env.reset()` below.
            nq, nv = env.sim.model.nq, env.sim.model.nv

            row_len = 1 + nq + nv
            if rows is None:
                rows = np.zeros((N_TRIALS, row_len), dtype=np.float64)

            converged = False
            last_max_qvel = float("inf")
            last_tilt_deg = 0.0
            sim = None
            for retry in range(SETTLE_MAX_RETRIES):
                # Per-trial seed so the placement sampler is deterministic
                # but different across trials/retries.
                env.seed(seed + t + retry * 100003)
                env.reset()
                sim = env.sim  # fresh handle — old one is invalid after reset

                # Object-only qvel + qpos-quat indices for the early-stop check.
                object_qvel_idx = []
                object_qpos_quat_adrs = []
                for jname in sim.model.joint_names:
                    jid = sim.model.joint_name2id(jname)
                    if sim.model.jnt_type[jid] != 0:  # mjJNT_FREE == 0
                        continue
                    ds = sim.model.jnt_dofadr[jid]
                    object_qvel_idx.extend(range(ds, ds + 6))
                    object_qpos_quat_adrs.append(
                        sim.model.jnt_qposadr[jid] + 3
                    )
                obj_qvel_arr = np.array(sorted(object_qvel_idx), dtype=np.int64)

                # Snapshot per-object quaternions BEFORE the settle drop so
                # we can measure tilt as delta-from-canonical (yaw-invariant)
                # rather than absolute world-z alignment, which is wrong for
                # bottles whose local +z is authored along their cylindrical
                # axis (alphabet_soup / salad_dressing / milk / tomato_sauce).
                init_quats = _read_quats(sim.data.qpos, object_qpos_quat_adrs)

                # Settle: env.step holds the robot via OSC gravity-comp
                # while objects fall to the floor. Require N consecutive
                # sub-tolerance readings so we don't exit at a bounce apex.
                low_streak = 0
                last_max_qvel = float("inf")
                velocity_settled = False
                for step_i in range(MAX_SETTLE_STEPS):
                    env.step(DUMMY_ACTION)
                    if step_i + 1 < MIN_SETTLE_STEPS:
                        continue
                    if obj_qvel_arr.size == 0:
                        low_streak = SETTLE_CONSEC
                    else:
                        last_max_qvel = float(
                            np.max(np.abs(sim.data.qvel[obj_qvel_arr]))
                        )
                        if last_max_qvel < SETTLE_QVEL_TOL:
                            low_streak += 1
                        else:
                            low_streak = 0
                    if low_streak >= SETTLE_CONSEC:
                        velocity_settled = True
                        break

                now_quats = _read_quats(sim.data.qpos, object_qpos_quat_adrs)
                last_tilt_deg = _max_tilt_delta_deg(now_quats, init_quats)

                if velocity_settled and last_tilt_deg <= SETTLE_TILT_DEG:
                    converged = True
                    break

                print(
                    f"  [{task.name}] trial {t} retry {retry + 1}: "
                    f"max|qvel|={last_max_qvel:.4f} tilt={last_tilt_deg:.1f}° "
                    f"after {MAX_SETTLE_STEPS} steps — re-seeding"
                )

            if not converged:
                # All placement retries leaned the same way — this
                # permutation is structurally bad. Drop it and pull a
                # fresh one from the pool without advancing t.
                print(
                    f"  [{task.name}] perm {perm_idx} unrecoverable for "
                    f"trial slot {t} (last tilt={last_tilt_deg:.1f}°); "
                    f"drawing next permutation"
                )
                continue  # while-loop: don't increment t, don't save

            state = np.asarray(sim.get_state().flatten(), dtype=np.float64)
            row = state[:row_len].copy()
            # Zero residual qvel so the saved state is fully at-rest.
            row[1 + nq :] = 0.0
            rows[t] = row
            perms_used.append(perm_idx)
            t += 1
        finally:
            if env is not None:
                env.close()
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    n_unique = len(np.unique(rows, axis=0))
    print(
        f"[{task.name}] target={target} basket={basket} "
        f"permuted={permute_objs} unique_rows={n_unique}/{N_TRIALS} "
        f"perms_drawn={len(perms_used)} pool={n_pool}"
    )

    torch.save(rows, pruned_path)
    torch.save(rows, init_path_out)
    return task.name


@ray.remote(num_cpus=1)
def _bake_task_remote(benchmark_name, task_idx, seed, include_basket):
    """Ray-side entry: re-resolve libero in the worker process and bake."""
    # Ray sets CUDA_VISIBLE_DEVICES="" on CPU-only workers (no num_gpus
    # claim), which crashes robosuite's EGL selector — it does
    # int(CUDA_VISIBLE_DEVICES.split(",")[0]) and chokes on the empty
    # string. Either drop the empty value or pin an explicit EGL device.
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

    # Re-import in the worker — Ray ships this module by source, so we
    # re-run the libero resolution there too. Imports are cheap relative
    # to the 50× env builds that follow.
    try:
        from libero.libero import benchmark as _bench_mod
    except ModuleNotFoundError:
        from libero import benchmark as _bench_mod

    bench = _bench_mod.get_benchmark_dict()[benchmark_name]()
    return bake_init_for_task(bench, task_idx, seed, include_basket)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default=BENCHMARK_NAME)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument(
        "--no-basket",
        action="store_true",
        help="Permute only the 5 distractors (basket stays fixed). Default: basket is permuted too.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Max concurrent Ray workers (default: bench.n_tasks). "
        "Each worker builds its own EGL/MuJoCo context.",
    )
    p.add_argument(
        "--stagger",
        type=float,
        default=STAGGER_SECS,
        help=f"Seconds between Ray task submissions to avoid EGL init "
        f"stampede (default: {STAGGER_SECS}).",
    )
    p.add_argument(
        "--no-ray",
        action="store_true",
        help="Run sequentially in this process (skip Ray fanout).",
    )
    args = p.parse_args()

    bench = benchmark.get_benchmark_dict()[args.benchmark]()

    if args.no_ray:
        for i in range(bench.n_tasks):
            # Per-task seed offset keeps trials within a task deterministic
            # while differing across tasks.
            bake_init_for_task(
                bench, i, args.seed + i * N_TRIALS,
                include_basket=not args.no_basket,
            )
        return

    workers = args.workers if args.workers is not None else bench.n_tasks
    ray.init(ignore_reinit_error=True, num_cpus=workers)

    futures = []
    for i in range(bench.n_tasks):
        fut = _bake_task_remote.remote(
            args.benchmark,
            i,
            args.seed + i * N_TRIALS,
            not args.no_basket,
        )
        futures.append(fut)
        if i < bench.n_tasks - 1 and args.stagger > 0:
            time.sleep(args.stagger)

    # Surface results as they finish so the user sees per-task progress
    # rather than a long silent wait.
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
