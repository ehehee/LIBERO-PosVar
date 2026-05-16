"""Bake per-arm joint AND Cartesian waypoints for the LIBERO
``libero_crate_washing`` task.

Strategy
--------
The Franka Panda kinematics in the raw ``crate-washing-sim/`` MJCF and in
LIBERO-PosVar's ``libero_crate_washing`` scene are identical (same joint
chain, same link lengths) and both scenes place the robot bases at world
(1.74, ±0.2, 0.76) with orientation ``(0, 0, π)``. The crate body
``crate_box_11`` sits at the same world location in both scenes.

Stage 1: joint targets. Run the proven Jacobian IK in
``graph-as-policy/examples/crate_washing/scripts/live_demo_bimanual_lift.py``
against the raw MJCF env, but pre-warp the IK seed to LIBERO's home
qpos so the joints solved are reachable from LIBERO's start state.

Stage 2: Cartesian targets. Boot a LIBERO env, apply each joint
waypoint to the appropriate arm, FK the resulting ``robot{0,1}_right_hand``
world pose, and record it. The LIBERO env's hand body orientation
convention differs from the raw MJCF's, so we cannot reuse the IK's
``_HANDLE_GRASP_QUAT`` directly — FK in LIBERO is the only way to
get the world-frame ``panda_hand`` pose that pyroki must hit to
reproduce the joint configuration the joint-transfer route reaches.

Outputs
-------
1. ``LIBERO-PosVar/scripts/baked_waypoints.json`` — each phase carries
   ``{left: {joints, hand_pose_world}, right: {joints, hand_pose_world}}``.
   ``hand_pose_world`` is ``{position: [x,y,z], rotation_wxyz: [w,x,y,z]}``.
2. ``stdout`` — Python literal constants (both joint arrays AND
   Cartesian pose tuples) ready to paste into
   ``graph-as-policy/examples/libero_crate_washing/graph/scripts/bimanual_lift/run.py``.

Run from the repo root (need ``MUJOCO_GL=egl`` for offscreen rendering):

    cd /home/pschalde/GaP/graph-as-policy
    MUJOCO_GL=egl /home/pschalde/GaP/graph-as-policy/.venv/bin/python \\
        /home/pschalde/GaP/LIBERO-PosVar/scripts/bake_libero_crate_waypoints.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

_GAP_ROOT = Path("/home/pschalde/GaP/graph-as-policy")
sys.path.insert(0, str(_GAP_ROOT / "services" / "sim_bridge"))
sys.path.insert(0, str(_GAP_ROOT / "examples" / "crate_washing" / "scripts"))

from env.crate_washing_env import CrateWashingEnv  # noqa: E402
from env.libero_bimanual_env import LiberoBimanualEnv  # noqa: E402
from live_demo_bimanual_lift import _compute_waypoints  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("bake")


# LIBERO Panda home qpos (matches robosuite Panda's ``init_qpos`` and is
# what the LIBERO env starts at after ``env.reset()``).
_LIBERO_HOME = np.array(
    [0.0, -0.161, 0.0, -2.445, 0.0, 2.227, 0.7854], dtype=np.float64
)


def _set_home(env: CrateWashingEnv) -> None:
    """Overwrite the original env's qpos with LIBERO's home pose for
    both arms before running IK. This puts the IK seed on the same arm
    configuration LIBERO will start from, ensuring the resulting joint
    targets are reachable via straight-line joint interpolation."""
    for a, q in zip(env._left_joint_addrs, _LIBERO_HOME):
        env.data.qpos[a] = q
    for a, q in zip(env._right_joint_addrs, _LIBERO_HOME):
        env.data.qpos[a] = q
    mujoco.mj_kinematics(env.model, env.data)


def _format_array(name: str, arr: np.ndarray) -> str:
    inner = ", ".join(f"{x:+.6f}" for x in arr)
    return f"{name} = np.array([{inner}], dtype=np.float64)"


def _fk_hand_pose_libero(
    libero_env: LiberoBimanualEnv,
    left_q: np.ndarray,
    right_q: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Apply ``left_q`` / ``right_q`` to the LIBERO env's qpos and return
    each arm's ``robot{N}_right_hand`` world pose as
    ``(position xyz, rotation wxyz)``.

    Robosuite's MjSim ``forward`` propagates qpos through the kinematic
    tree so ``xpos`` / ``xquat`` of every body reflect the requested
    joint configuration without stepping physics.
    """
    sim = libero_env.handle_env.env.sim
    saved_qpos = sim.data.qpos.copy()

    for a, q in zip(libero_env._left_qpos_addrs, left_q):
        sim.data.qpos[a] = q
    for a, q in zip(libero_env._right_qpos_addrs, right_q):
        sim.data.qpos[a] = q
    sim.forward()

    l_hand = sim.model.body_name2id("robot0_right_hand")
    r_hand = sim.model.body_name2id("robot1_right_hand")
    left_pose = (sim.data.xpos[l_hand].copy(), sim.data.xquat[l_hand].copy())
    right_pose = (sim.data.xpos[r_hand].copy(), sim.data.xquat[r_hand].copy())

    sim.data.qpos[:] = saved_qpos
    sim.forward()
    return left_pose, right_pose


def main() -> None:
    if "DISPLAY" not in os.environ:
        os.environ.setdefault("MUJOCO_GL", "egl")

    log.info("Building CrateWashingEnv (vendored MJCF)")
    env = CrateWashingEnv(enable_render=False)

    log.info("Pre-warping IK seed to LIBERO home qpos")
    _set_home(env)

    crate_bid = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_BODY, "crate_box_11"
    )
    z0 = float(env.data.xpos[crate_bid, 2])
    log.info("crate_box_11 starts at world Z = %.4f", z0)

    log.info("Running Jacobian IK for all phases…")
    wp = _compute_waypoints(env)

    # Pack joint waypoints first. Include HOME so the workflow can
    # return both arms to a neutral pose at the end.
    out: dict[str, dict[str, dict]] = {
        "HOME": {
            "left": {"joints": _LIBERO_HOME.tolist()},
            "right": {"joints": _LIBERO_HOME.tolist()},
        }
    }
    for label, (ql, qr) in wp.items():
        out[label] = {
            "left": {"joints": ql.tolist()},
            "right": {"joints": qr.tolist()},
        }

    log.info("Booting LiberoBimanualEnv to FK joint targets → world hand poses")
    libero_env = LiberoBimanualEnv(task_id=0, max_steps=200, cam_w=128, cam_h=128)
    try:
        for label, blocks in out.items():
            l_q = np.asarray(blocks["left"]["joints"])
            r_q = np.asarray(blocks["right"]["joints"])
            (l_p, l_quat), (r_p, r_quat) = _fk_hand_pose_libero(
                libero_env, l_q, r_q
            )
            blocks["left"]["hand_pose_world"] = {
                "position": l_p.tolist(),
                "rotation_wxyz": l_quat.tolist(),
            }
            blocks["right"]["hand_pose_world"] = {
                "position": r_p.tolist(),
                "rotation_wxyz": r_quat.tolist(),
            }
            log.info(
                "FK[%s] left hand world pos=%s quat=%s",
                label, np.round(l_p, 4), np.round(l_quat, 4),
            )
            log.info(
                "FK[%s] right hand world pos=%s quat=%s",
                label, np.round(r_p, 4), np.round(r_quat, 4),
            )
    finally:
        libero_env.close()

    json_path = Path(__file__).with_name("baked_waypoints.json")
    json_path.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", json_path)

    print("\n# === Paste into bimanual_lift/run.py ===")
    print("import numpy as np\n")
    for label in ("HOME", "APPROACH_SIDE", "INSERT_TO_HANDLE", "LIFT"):
        if label not in out:
            continue
        ql = np.array(out[label]["left"]["joints"])
        qr = np.array(out[label]["right"]["joints"])
        print(_format_array(f"_LEFT_{label}", ql))
        print(_format_array(f"_RIGHT_{label}", qr))
        for arm in ("left", "right"):
            pose = out[label][arm]["hand_pose_world"]
            pos = pose["position"]
            quat = pose["rotation_wxyz"]
            print(
                f"_{arm.upper()}_{label}_HAND_POS = np.array([{pos[0]:+.6f}, "
                f"{pos[1]:+.6f}, {pos[2]:+.6f}], dtype=np.float64)"
            )
            print(
                f"_{arm.upper()}_{label}_HAND_QUAT_WXYZ = np.array([{quat[0]:+.6f}, "
                f"{quat[1]:+.6f}, {quat[2]:+.6f}, {quat[3]:+.6f}], dtype=np.float64)"
            )
        print()


if __name__ == "__main__":
    main()
