"""Interactive MuJoCo viewer for the ``libero_crate_washing`` benchmark.

Loads the LIBERO env exactly the same way ``eval_crate_washing_smoke.py``
does (so the BDDL parse / problem class / two-Panda merge are all
exercised end-to-end), but pops up the standard ``mujoco.viewer``
passive-viewer window so you can actually see the bimanual scene.

By default the script just holds the post-reset pose; the two arms sit in
their canonical "ready" qpos and zero OSC actions are sent every control
step so the controllers hold them there. Optionally pass ``--use-init``
to seed the sim from the first row of the baked init file (currently the
same pose, but mirrors the popcorn eval pattern).

Requirements:
    * A display reachable from the process (``DISPLAY`` set, X / VNC
      forwarding, etc.). The viewer uses GLFW — it will NOT work over
      plain SSH without forwarding. Do NOT set ``MUJOCO_GL=egl`` for
      this script (that disables GLFW).

Run from the LIBERO-PosVar repo root::

    /home/pschalde/GaP/graph-as-policy/.venv/bin/python3 \\
        scripts/view_crate_washing.py
"""

import argparse
import os
import sys
import time

import numpy as np
import yaml

import mujoco
import mujoco.viewer

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
        "--use-init",
        action="store_true",
        help="If set, load the first row from the baked init file before viewing.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="If set, stop after this many sim steps; otherwise hold forever.",
    )
    parser.add_argument(
        "--slowdown",
        type=float,
        default=1.0,
        help="Wall-clock playback factor. 1.0 = real-time; 2.0 = half speed.",
    )
    parser.add_argument(
        "--collision_mesh",
        "--collision-mesh",
        action="store_true",
        help=(
            "Render the active decomposed collision meshes for the crate and "
            "washing machine in the viewer."
        ),
    )
    args = parser.parse_args()

    bench = benchmark.get_benchmark_dict()[BENCHMARK_NAME]()
    task = bench.tasks[0]
    bddl_path = bench.get_task_bddl_file_path(0)
    print(f"[view] benchmark={BENCHMARK_NAME}")
    print(f"[view] task={task.name}  language={task.language!r}")
    print(f"[view] bddl={bddl_path}")

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        render_collision_mesh=args.collision_mesh,
        render_visual_mesh=True,
    )
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
            print(f"[view] loaded init row 0 from {init_path}")
        else:
            print(f"[view] WARNING: --use-init but {init_path} missing")

    # The robosuite MjSim wraps the raw mujoco.MjModel / mujoco.MjData at
    # `_model` / `_data`. The passive viewer wants the raw objects.
    mj_model = env.env.sim.model._model
    mj_data = env.env.sim.data._data
    crate_collision_count = 0
    machine_collision_count = 0
    if args.collision_mesh:
        crate_collision_count, machine_collision_count = show_contact_surfaces(mj_model)

    action_dim = env.env.action_dim
    print(f"[view] action_dim={action_dim}")
    print(
        f"[view] stage_idx (post-reset)={env.env.crate_stage_idx} "
        f"/ {env.env.crate_num_stages}"
    )
    if args.collision_mesh:
        print(
            "[view] collision mesh overlay enabled "
            f"(crate={crate_collision_count}, machine={machine_collision_count})"
        )

    noop = np.zeros(action_dim, dtype=np.float64)
    sim_dt = float(mj_model.opt.timestep)
    print(f"[view] sim dt={sim_dt:.4f}s  (close the viewer window to exit)")

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        # Frame the bimanual workspace. Mirrors live_demo.py's camera.
        viewer.cam.lookat[:] = (1.0, 0.0, 0.6)
        viewer.cam.distance = 2.6
        viewer.cam.azimuth = 130.0
        viewer.cam.elevation = -22.0
        viewer.sync()

        step_i = 0
        start_wall = time.time()
        start_sim = float(mj_data.time)
        while viewer.is_running():
            if args.max_steps is not None and step_i >= args.max_steps:
                break
            # Driving via env.step keeps the controllers / observables in
            # sync, but it can be slow (it advances control_freq sim steps
            # per call). Stepping mj directly is faster for pure viewing.
            env.step(noop)
            viewer.sync()
            step_i += 1
            if step_i % 50 == 0:
                print(
                    f"[view] step={step_i:04d}  "
                    f"stage_idx={env.env.crate_stage_idx}/{env.env.crate_num_stages}  "
                    f"success={env.env._check_success()}"
                )
            # Throttle to wall-clock so motion is watchable.
            elapsed_real = time.time() - start_wall
            elapsed_sim = (float(mj_data.time) - start_sim) * args.slowdown
            sleep_for = elapsed_sim - elapsed_real
            if sleep_for > 0.001:
                time.sleep(sleep_for)

        # Hold the final pose so the user can inspect the scene before
        # closing the window.
        print(
            f"[view] final stage_idx={env.env.crate_stage_idx} "
            f"/ {env.env.crate_num_stages}  "
            f"check_success={env.env._check_success()}"
        )
        while viewer.is_running():
            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()
            time.sleep(0.01)

    env.close()


def show_contact_surfaces(mj_model):
    """Make crate and washing-machine contact geoms visible in the passive viewer."""
    crate_count = 0
    machine_count = 0
    for gid in range(mj_model.ngeom):
        gname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if gname.startswith("crate_box_11_col_") or gname.startswith(
            "crate_box_11_handle_"
        ):
            crate_count += 1
            mj_model.geom_group[gid] = 2
            mj_model.geom_rgba[gid] = (1.0, 0.45, 0.0, 0.45)
        elif gname.startswith("crate_machine_col_"):
            machine_count += 1
            mj_model.geom_group[gid] = 2
            mj_model.geom_rgba[gid] = (0.1, 0.7, 1.0, 0.35)
    return crate_count, machine_count


if __name__ == "__main__":
    main()
