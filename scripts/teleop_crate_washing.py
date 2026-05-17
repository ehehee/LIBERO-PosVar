"""Interactive MuJoCo viewer + keyboard teleop for libero_crate_washing.

Same scene/load path as ``view_crate_washing.py``, but adds simple keyboard
control of the two Franka arms via per-tick IK so you can drive the
grippers into the crate handles and test whether the new CoACD collision
mesh actually behaves like real handle cutouts.

Run from the LIBERO-PosVar repo root (needs a reachable X display)::

    python scripts/teleop_crate_washing.py

Do NOT set MUJOCO_GL=egl (the passive viewer requires GLFW with a window).

Mouse + keyboard teleop (deterministic, no drift):
    Ctrl + right-drag an end-effector body in the MuJoCo viewer to translate XYZ
                    (dragging never changes the orientation target)
    TAB             switch active arm (left <-> right)
    I / K           active arm:  +Y  /  -Y     (world frame, 1 cm per press)
    J / L           active arm:  -X  /  +X
    U / O           active arm:  +Z  /  -Z
    N / M           active arm:  yaw -  /  yaw +   (3 deg per press)
    B / V           active arm:  pitch - / pitch +
    , / .           active arm:  roll - / roll +
    Z / X           close / open active gripper
    C               toggle active gripper open/closed
    1 / 2           make left / right arm active
    5 / 6           close / open left gripper
    3 / 4           close / open right gripper
    E / D           close / open both grippers
    R               re-seat IK target at current EE pose (recover divergence)
    T               toggle: crate + washing machine contact-surface overlay
    H               print current EE pose + targets to console
    BACKSPACE       reset env  (built-in viewer key)
    SPACE           pause/unpause physics  (built-in viewer key)

Dragging or keys update each arm's Cartesian target, then IK latches a joint
target. Between updates the arms hold those joint targets with gravity
compensation, which is much more stable than re-solving IK every sim tick.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import yaml

import mujoco
import mujoco.viewer

# --- LIBERO setup (mirrors view_crate_washing.py) ---------------------------

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
for _path in (_REPO_ROOT, os.path.join(_REPO_ROOT, "libero")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

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
    from libero.libero import benchmark as _benchmark
    from libero.libero.envs import OffScreenRenderEnv
except ModuleNotFoundError:
    from libero import benchmark as _benchmark
    from libero.envs import OffScreenRenderEnv

BENCHMARK_NAME = "libero_crate_washing"

# --- DOF / actuator layout --------------------------------------------------

_ARM_DOF = 7
_EE_BODY_NAMES = ("gripper0_eef", "gripper1_eef")
# Actuator name templates (robosuite Panda + standard gripper).
_ARM_ACT_NAMES = ("robot0_torq_j{}", "robot1_torq_j{}")
_GRIP_ACT_NAMES = (
    ("gripper0_gripper_finger_joint1", "gripper0_gripper_finger_joint2"),
    ("gripper1_gripper_finger_joint1", "gripper1_gripper_finger_joint2"),
)

# Per-joint PD gains (Nm/rad and Nm*s/rad). Critically damped against the
# Panda joint inertias; high enough to overpower gravity firmly.
_KP = np.array([400.0, 400.0, 400.0, 400.0, 200.0, 100.0, 50.0])
_KD = 2.0 * np.sqrt(_KP) * 1.0  # damping_ratio = 1
# Torque clip per joint to match actuator ctrlrange (Nm).
_TAU_LIMIT = np.array([80.0, 80.0, 80.0, 80.0, 80.0, 12.0, 12.0])

# GLFW key codes (we avoid importing glfw to keep deps minimal).
_K_TAB, _K_SPACE, _K_BACKSPACE = 258, 32, 259
_K_I, _K_J, _K_K, _K_L, _K_U, _K_O = 73, 74, 75, 76, 85, 79
_K_N, _K_M, _K_Z, _K_X = 78, 77, 90, 88
_K_C = 67
_K_E, _K_D = 69, 68
_K_1, _K_2, _K_3, _K_4, _K_5, _K_6 = 49, 50, 51, 52, 53, 54
_K_R, _K_T, _K_H, _K_P = 82, 84, 72, 80
_K_B, _K_V = 66, 86
_K_COMMA, _K_PERIOD = 44, 46

_TRANS_STEP = 0.01   # meters per key press
_ROT_STEP = np.deg2rad(3.0)  # radians per key press
_MAX_DRAG_STEP = 0.015  # meters accepted from the viewer per control tick


def _first(maybe_tuple):
    if isinstance(maybe_tuple, tuple):
        return maybe_tuple[0]
    return maybe_tuple


# --- IK ---------------------------------------------------------------------


class ArmIK:
    """DLS 6-DOF IK for one Panda arm on a private MjData."""

    def __init__(
        self,
        model: mujoco.MjModel,
        ee_body_id: int,
        qpos_addrs: list[int],
        dof_addrs: list[int],
        joint_ranges: np.ndarray,
    ):
        self.model = model
        self.ik_data = mujoco.MjData(model)
        self.ee_body_id = ee_body_id
        self.qpos_addrs = np.array(qpos_addrs, dtype=np.int32)
        self.dof_addrs = np.array(dof_addrs, dtype=np.int32)
        self.joint_ranges = np.array(joint_ranges, dtype=np.float64)
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self._lambda_sq = 0.05**2
        self._max_iters = 12
        self._pos_tol = 1e-3
        self._rot_tol = 5e-3
        self._step_clip = 0.06
        self._total_delta_clip = 0.18

    def solve(
        self,
        full_qpos: np.ndarray,
        target_pos: np.ndarray,
        target_quat_wxyz: np.ndarray,
        *,
        rot_weight: float = 1.0,
        posture_weight: float = 0.0,
        posture_target: np.ndarray | None = None,
    ) -> np.ndarray:
        self.ik_data.qpos[:] = full_qpos
        seed_q = np.array(self.ik_data.qpos[self.qpos_addrs], dtype=np.float64)
        if posture_target is None:
            posture_target = seed_q
        for _ in range(self._max_iters):
            mujoco.mj_forward(self.model, self.ik_data)
            cur_pos = self.ik_data.xpos[self.ee_body_id]
            cur_quat = self.ik_data.xquat[self.ee_body_id]
            err_pos = target_pos - cur_pos
            err_rot = np.zeros(3)
            mujoco.mju_subQuat(err_rot, target_quat_wxyz, cur_quat)
            err_rot *= rot_weight
            if (
                np.linalg.norm(err_pos) < self._pos_tol
                and (rot_weight <= 0.0 or np.linalg.norm(err_rot) < self._rot_tol)
            ):
                break
            mujoco.mj_jacBody(
                self.model, self.ik_data, self._jacp, self._jacr, self.ee_body_id
            )
            J = np.vstack(
                [
                    self._jacp[:, self.dof_addrs],
                    rot_weight * self._jacr[:, self.dof_addrs],
                ]
            )
            err = np.concatenate([err_pos, err_rot])
            j_hash = J.T @ np.linalg.solve(
                J @ J.T + self._lambda_sq * np.eye(J.shape[0]), np.eye(J.shape[0])
            )
            dq = j_hash @ err
            if posture_weight > 0.0:
                nullspace = np.eye(_ARM_DOF) - j_hash @ J
                dq += nullspace @ (
                    posture_weight
                    * (posture_target - self.ik_data.qpos[self.qpos_addrs])
                )
            n = float(np.linalg.norm(dq))
            if n > self._step_clip:
                dq *= self._step_clip / n
            self.ik_data.qpos[self.qpos_addrs] += dq
            total_delta = self.ik_data.qpos[self.qpos_addrs] - seed_q
            total_delta_norm = float(np.linalg.norm(total_delta))
            if total_delta_norm > self._total_delta_clip:
                self.ik_data.qpos[self.qpos_addrs] = (
                    seed_q + total_delta * (self._total_delta_clip / total_delta_norm)
                )
                break
            self.ik_data.qpos[self.qpos_addrs] = np.clip(
                self.ik_data.qpos[self.qpos_addrs],
                self.joint_ranges[:, 0],
                self.joint_ranges[:, 1],
            )
        mujoco.mj_forward(self.model, self.ik_data)
        return np.array(self.ik_data.qpos[self.qpos_addrs], dtype=np.float64)


# --- Main -------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-hz", type=int, default=20)
    args = parser.parse_args()

    bench = _benchmark.get_benchmark_dict()[BENCHMARK_NAME]()
    task = bench.tasks[0]
    bddl_path = bench.get_task_bddl_file_path(0)
    print(f"[teleop] benchmark={BENCHMARK_NAME}  task={task.name}")
    print(f"[teleop] bddl={bddl_path}")

    # We only use ``OffScreenRenderEnv`` for scene loading + reset; the main
    # loop bypasses ``env.step`` and runs raw ``mj_step`` so we can drive
    # actuators with a stiff PD + gravity-comp controller (robosuite's
    # JOINT_POSITION resets goal=current_qpos every tick, leaving no
    # restoring force against gravity = the "momentum drift" we observed).
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        controller="JOINT_POSITION",
        control_freq=args.control_hz,
        camera_widths=64,
        camera_heights=48,
        camera_names=["agentview"],
        camera_depths=False,
        horizon=100_000,
    )
    env.reset()

    sim = env.env.sim
    model = sim.model._model
    data = sim.data._data
    mujoco.mj_forward(model, data)

    # Robosuite keeps visual meshes in group 1 and collision meshes in group 0.
    # The latter are the green/blue Franka shapes that clutter the passive
    # viewer. Alpha-hiding them does not disable contacts.
    for gid in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        bname = (
            mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])
            )
            or ""
        )
        if (
            gname.endswith("_collision")
            and bname.startswith(("robot0_", "robot1_", "gripper0_", "gripper1_"))
        ):
            model.geom_rgba[gid, 3] = 0.0

    left_qpos_addrs = [
        int(_first(sim.model.get_joint_qpos_addr(f"robot0_joint{i}"))) for i in range(1, 8)
    ]
    right_qpos_addrs = [
        int(_first(sim.model.get_joint_qpos_addr(f"robot1_joint{i}"))) for i in range(1, 8)
    ]
    left_dof_addrs = [
        int(_first(sim.model.get_joint_qvel_addr(f"robot0_joint{i}"))) for i in range(1, 8)
    ]
    right_dof_addrs = [
        int(_first(sim.model.get_joint_qvel_addr(f"robot1_joint{i}"))) for i in range(1, 8)
    ]
    left_joint_ranges = np.array([
        model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{i}")]
        for i in range(1, 8)
    ], dtype=np.float64)
    right_joint_ranges = np.array([
        model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot1_joint{i}")]
        for i in range(1, 8)
    ], dtype=np.float64)

    # Actuator IDs (motor torques for the arm; position actuators for grip).
    arm_act_ids = [
        np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, _ARM_ACT_NAMES[arm_id].format(i))
            for i in range(1, 8)
        ], dtype=np.int32)
        for arm_id in range(2)
    ]
    grip_act_ids = [
        np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in _GRIP_ACT_NAMES[arm_id]
        ], dtype=np.int32)
        for arm_id in range(2)
    ]
    grip_ctrl_ranges = [
        np.array([model.actuator_ctrlrange[a] for a in grip_act_ids[arm_id]])
        for arm_id in range(2)
    ]
    # Sim substeps per teleop tick (e.g., 100 at sim_dt=0.5ms with control_hz=20).
    sim_substeps = max(1, int(round((1.0 / args.control_hz) / model.opt.timestep)))

    ee_body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in _EE_BODY_NAMES
    ]
    iks = [
        ArmIK(model, ee_body_ids[0], left_qpos_addrs, left_dof_addrs, left_joint_ranges),
        ArmIK(model, ee_body_ids[1], right_qpos_addrs, right_dof_addrs, right_joint_ranges),
    ]

    # --- Geom rgba snapshot for the visual/collision toggle ---------------
    # The passive viewer may have geom group 3 disabled, so render collision
    # overlays in group 2 alongside the visual scene. We keep visual meshes
    # visible and overlay contact surfaces on `T`.
    crate_visual_geom_ids: list[int] = []
    crate_collision_geom_ids: list[int] = []
    machine_collision_geom_ids: list[int] = []
    for gid in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if gname == "crate_box_11_visual":
            crate_visual_geom_ids.append(gid)
        elif gname.startswith("crate_box_11_col_") or gname.startswith(
            "crate_box_11_handle_"
        ):
            crate_collision_geom_ids.append(gid)
            model.geom_group[gid] = 2
        elif gname.startswith("crate_machine_col_"):
            machine_collision_geom_ids.append(gid)
            model.geom_group[gid] = 2
    visual_rgba_default = {
        gid: np.array(model.geom_rgba[gid]) for gid in crate_visual_geom_ids
    }
    collision_rgba_default = {
        gid: np.array(model.geom_rgba[gid]) for gid in crate_collision_geom_ids
    }
    machine_collision_rgba_default = {
        gid: np.array(model.geom_rgba[gid]) for gid in machine_collision_geom_ids
    }
    # Default state: visual visible, collision hidden.
    for gid in crate_collision_geom_ids:
        model.geom_rgba[gid] = (1.0, 0.5, 0.0, 0.0)  # invisible-orange
    for gid in machine_collision_geom_ids:
        model.geom_rgba[gid] = (0.1, 0.7, 1.0, 0.0)  # invisible-blue
    show_collision = [False]

    def apply_show_collision(show: bool) -> None:
        if show:
            for gid in crate_visual_geom_ids:
                model.geom_rgba[gid] = visual_rgba_default[gid]
            for gid in crate_collision_geom_ids:
                model.geom_rgba[gid] = (1.0, 0.45, 0.0, 0.45)
            for gid in machine_collision_geom_ids:
                model.geom_rgba[gid] = (0.1, 0.7, 1.0, 0.35)
        else:
            for gid in crate_visual_geom_ids:
                model.geom_rgba[gid] = visual_rgba_default[gid]
            for gid in crate_collision_geom_ids:
                model.geom_rgba[gid] = (*collision_rgba_default[gid][:3], 0.0)
            for gid in machine_collision_geom_ids:
                model.geom_rgba[gid] = (*machine_collision_rgba_default[gid][:3], 0.0)

    # --- Teleop state -----------------------------------------------------
    active_arm = [0]                       # mutable so closures can rebind
    gripper_fraction = [1.0, 1.0]          # 0 closed, 1 open
    dragging_arm = [None]                  # arm whose viewer perturb is active
    dragging_body = [None]                 # selected body being dragged
    dragging_body_quat = [np.array([1.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])]
    dragging_last_refpos = [np.zeros(3), np.zeros(3)]
    dragging_posture = [np.zeros(_ARM_DOF), np.zeros(_ARM_DOF)]
    # Per-arm Cartesian target (world frame). Recomputed only on user input.
    target_pos = [np.array(data.xpos[ee_body_ids[i]], dtype=np.float64) for i in range(2)]
    target_quat = [np.array(data.xquat[ee_body_ids[i]], dtype=np.float64) for i in range(2)]
    # Per-arm joint target (latched). Driven to by the JOINT_POSITION
    # controller every tick. Recomputed via IK ONLY when the user nudges the
    # Cartesian target; this prevents the per-tick IK-drift the user reported.
    joint_targets = [
        np.array([data.qpos[a] for a in left_qpos_addrs], dtype=np.float64),
        np.array([data.qpos[a] for a in right_qpos_addrs], dtype=np.float64),
    ]

    def resolve_ik(
        arm_id: int,
        *,
        rot_weight: float = 1.0,
        posture_weight: float = 0.0,
        posture_target: np.ndarray | None = None,
        max_predicted_rot_err: float | None = None,
    ) -> bool:
        full_qpos = np.array(data.qpos, dtype=np.float64)
        tq = target_quat[arm_id]
        tq = tq / max(float(np.linalg.norm(tq)), 1e-9)
        candidate = iks[arm_id].solve(
            full_qpos,
            target_pos[arm_id],
            tq,
            rot_weight=rot_weight,
            posture_weight=posture_weight,
            posture_target=posture_target,
        )
        if max_predicted_rot_err is not None:
            err = np.zeros(3)
            mujoco.mju_subQuat(
                err, tq, iks[arm_id].ik_data.xquat[ee_body_ids[arm_id]]
            )
            if float(np.linalg.norm(err)) > max_predicted_rot_err:
                return False
        joint_targets[arm_id] = candidate
        return True

    def seat_target_at_current(arm_id: int) -> None:
        target_pos[arm_id] = np.array(data.xpos[ee_body_ids[arm_id]], dtype=np.float64)
        target_quat[arm_id] = np.array(data.xquat[ee_body_ids[arm_id]], dtype=np.float64)
        addrs = left_qpos_addrs if arm_id == 0 else right_qpos_addrs
        joint_targets[arm_id] = np.array([data.qpos[a] for a in addrs], dtype=np.float64)

    def nudge_rotation(arm_id: int, axis: np.ndarray, sign: float) -> None:
        """Rotate target_quat[arm_id] by ``sign * _ROT_STEP`` around world axis."""
        half = _ROT_STEP * sign * 0.5
        dq = np.array([np.cos(half), *(np.sin(half) * axis)], dtype=np.float64)
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, dq, target_quat[arm_id])
        target_quat[arm_id] = out / max(float(np.linalg.norm(out)), 1e-9)

    def print_active_pose() -> None:
        a = active_arm[0]
        ee = ee_body_ids[a]
        print(
            f"[arm {a}] EE pos={np.array2string(np.asarray(data.xpos[ee]), precision=3, suppress_small=True)}  "
            f"tgt={np.array2string(target_pos[a], precision=3, suppress_small=True)}  "
            f"grip={gripper_fraction[a]:.2f}"
        )

    def set_gripper(arm_id: int, fraction: float) -> None:
        gripper_fraction[arm_id] = float(np.clip(fraction, 0.0, 1.0))
        state = "open" if gripper_fraction[arm_id] > 0.5 else "closed"
        print(f"[arm {arm_id}] gripper -> {state}")

    def set_active_arm(arm_id: int) -> None:
        active_arm[0] = int(np.clip(arm_id, 0, 1))
        print(f"[teleop] active arm -> {'right' if active_arm[0] else 'left'}")
        print_active_pose()

    def set_both_grippers(fraction: float) -> None:
        set_gripper(0, fraction)
        set_gripper(1, fraction)

    def arm_for_selected_body(body_id: int) -> int | None:
        """Return the arm whose gripper body owns ``body_id``.

        Dragging arbitrary Franka links makes the viewer apply a physical
        perturb to the arm while IK also chases the wrist target. Restricting
        teleop to hand/finger/eef bodies keeps the interaction well behaved.
        """
        if body_id < 0:
            return None
        cur = int(body_id)
        while cur > 0:
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, cur) or ""
            if bname.startswith("gripper0_"):
                return 0
            if bname.startswith("gripper1_"):
                return 1
            cur = int(model.body_parentid[cur])
        return None

    def clear_viewer_perturb_forces() -> None:
        # Passive viewer drag uses MuJoCo perturb mechanics under the hood.
        # We only want the perturb pose as an input device, never physical
        # forces/torques that can twist the articulated robot.
        data.xfrc_applied[:] = 0.0
        data.qfrc_applied[:] = 0.0

    def sync_from_viewer_drag(viewer) -> None:
        """Use MuJoCo viewer perturb translation as an XYZ-only IK target.

        In the passive viewer, Ctrl-dragging a selected body updates the
        perturb reference pose. The eef bodies are kinematic links, so MuJoCo
        cannot move them directly; we instead read the desired reference
        position and solve IK for the matching Panda arm while preserving the
        arm's current orientation target.
        """
        pert = getattr(viewer, "perturb", getattr(viewer, "pert", None))
        if pert is None:
            dragging_arm[0] = None
            dragging_body[0] = None
            return
        if int(getattr(pert, "active", 0)) == 0:
            dragging_arm[0] = None
            dragging_body[0] = None
            return
        selected_body = int(getattr(pert, "select", -1))
        arm_id = arm_for_selected_body(selected_body)
        if arm_id is None:
            dragging_arm[0] = None
            dragging_body[0] = None
            return

        requested_body_pos = np.array(pert.refpos, dtype=np.float64)
        if dragging_arm[0] != arm_id or dragging_body[0] != selected_body:
            # On a fresh drag, preserve the orientation the arm is currently
            # holding instead of inheriting the viewer perturb quaternion.
            target_pos[arm_id] = np.array(data.xpos[ee_body_ids[arm_id]], dtype=np.float64)
            target_quat[arm_id] = np.array(data.xquat[ee_body_ids[arm_id]], dtype=np.float64)
            target_quat[arm_id] /= max(float(np.linalg.norm(target_quat[arm_id])), 1e-9)
            dragging_body_quat[arm_id] = np.array(data.xquat[selected_body], dtype=np.float64)
            dragging_body_quat[arm_id] /= max(float(np.linalg.norm(dragging_body_quat[arm_id])), 1e-9)
            dragging_last_refpos[arm_id] = requested_body_pos
            addrs = left_qpos_addrs if arm_id == 0 else right_qpos_addrs
            dragging_posture[arm_id] = np.array([data.qpos[a] for a in addrs], dtype=np.float64)
            dragging_arm[0] = arm_id
            dragging_body[0] = selected_body
            return

        # Keep MuJoCo's perturb handle orientation pinned. Right-drag should
        # only update refpos; if refquat drifts, the native interaction and IK
        # feel coupled even when we ignore the incoming perturb quaternion.
        pert.refquat[:] = dragging_body_quat[arm_id]
        delta = requested_body_pos - dragging_last_refpos[arm_id]
        dragging_last_refpos[arm_id] = requested_body_pos
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > _MAX_DRAG_STEP:
            delta *= _MAX_DRAG_STEP / delta_norm
        new_pos = target_pos[arm_id] + delta

        if np.linalg.norm(new_pos - target_pos[arm_id]) < 1e-5:
            return

        active_arm[0] = arm_id
        old_target_pos = np.array(target_pos[arm_id], dtype=np.float64)
        target_pos[arm_id] = new_pos
        # Right-drag should translate only. Keep the quaternion captured at
        # drag start; never inherit MuJoCo's perturb quaternion.
        accepted = resolve_ik(
            arm_id,
            rot_weight=4.0,
            posture_weight=0.03,
            posture_target=dragging_posture[arm_id],
            max_predicted_rot_err=0.04,
        )
        if not accepted:
            target_pos[arm_id] = old_target_pos
            dragging_last_refpos[arm_id] -= delta
            return

    def key_callback(keycode: int) -> None:
        a = active_arm[0]
        nudged_cart = False
        if keycode == _K_TAB:
            set_active_arm(1 - active_arm[0])
            return
        elif keycode == _K_1:
            set_active_arm(0)
            return
        elif keycode == _K_2:
            set_active_arm(1)
            return
        elif keycode == _K_I: target_pos[a][1] += _TRANS_STEP; nudged_cart = True
        elif keycode == _K_K: target_pos[a][1] -= _TRANS_STEP; nudged_cart = True
        elif keycode == _K_J: target_pos[a][0] -= _TRANS_STEP; nudged_cart = True
        elif keycode == _K_L: target_pos[a][0] += _TRANS_STEP; nudged_cart = True
        elif keycode == _K_U: target_pos[a][2] += _TRANS_STEP; nudged_cart = True
        elif keycode == _K_O: target_pos[a][2] -= _TRANS_STEP; nudged_cart = True
        elif keycode == _K_N: nudge_rotation(a, np.array([0.0, 0.0, 1.0]), -1.0); nudged_cart = True
        elif keycode == _K_M: nudge_rotation(a, np.array([0.0, 0.0, 1.0]), +1.0); nudged_cart = True
        elif keycode == _K_B: nudge_rotation(a, np.array([0.0, 1.0, 0.0]), -1.0); nudged_cart = True
        elif keycode == _K_V: nudge_rotation(a, np.array([0.0, 1.0, 0.0]), +1.0); nudged_cart = True
        elif keycode == _K_COMMA: nudge_rotation(a, np.array([1.0, 0.0, 0.0]), -1.0); nudged_cart = True
        elif keycode == _K_PERIOD: nudge_rotation(a, np.array([1.0, 0.0, 0.0]), +1.0); nudged_cart = True
        elif keycode == _K_Z:
            set_gripper(a, 0.0)
            return
        elif keycode == _K_X:
            set_gripper(a, 1.0)
            return
        elif keycode == _K_C:
            set_gripper(a, 0.0 if gripper_fraction[a] > 0.5 else 1.0)
            return
        elif keycode == _K_5:
            set_gripper(0, 0.0)
            return
        elif keycode == _K_6:
            set_gripper(0, 1.0)
            return
        elif keycode == _K_3:
            set_gripper(1, 0.0)
            return
        elif keycode == _K_4:
            set_gripper(1, 1.0)
            return
        elif keycode == _K_E:
            set_both_grippers(0.0)
            return
        elif keycode == _K_D:
            set_both_grippers(1.0)
            return
        elif keycode == _K_R:
            seat_target_at_current(a)
            print(f"[arm {a}] target re-seated at current EE pose"); print_active_pose()
            return
        elif keycode == _K_T:
            if not crate_collision_geom_ids and not machine_collision_geom_ids:
                print("[teleop] no decomposed contact surfaces to toggle")
                return
            show_collision[0] = not show_collision[0]
            apply_show_collision(show_collision[0])
            print(
                f"[teleop] contact surfaces -> "
                f"{'shown' if show_collision[0] else 'hidden'} "
                f"(crate={len(crate_collision_geom_ids)}, machine={len(machine_collision_geom_ids)})"
            )
            return
        elif keycode == _K_H:
            for i in range(2):
                ee = ee_body_ids[i]
                print(
                    f"[arm {i}] EE pos={np.array2string(np.asarray(data.xpos[ee]), precision=3, suppress_small=True)}  "
                    f"tgt={np.array2string(target_pos[i], precision=3, suppress_small=True)}  "
                    f"grip={gripper_fraction[i]:.2f}"
                )
            return
        else:
            return

        # Only re-IK when the Cartesian target actually changed. Latched joint
        # targets between presses are what eliminate per-tick drift.
        if nudged_cart:
            resolve_ik(a)
            print_active_pose()

    def apply_arm_torques() -> None:
        """PD + gravity-comp on each arm, written to motor actuator ctrl."""
        # qfrc_bias contains Coriolis + gravitational forces in generalized
        # coordinates for the *current* state (qpos, qvel). Using it as
        # feedforward exactly cancels gravity so PD only needs to handle
        # the residual error.
        bias = data.qfrc_bias
        arm_addrs = (
            (0, left_qpos_addrs, left_dof_addrs),
            (1, right_qpos_addrs, right_dof_addrs),
        )
        for arm_id, qpos_addrs, dof_addrs in arm_addrs:
            q = np.array([data.qpos[a] for a in qpos_addrs], dtype=np.float64)
            qd = np.array([data.qvel[a] for a in dof_addrs], dtype=np.float64)
            err = joint_targets[arm_id] - q
            tau = _KP * err - _KD * qd + np.array([bias[a] for a in dof_addrs])
            tau = np.clip(tau, -_TAU_LIMIT, _TAU_LIMIT)
            for i, act_id in enumerate(arm_act_ids[arm_id]):
                data.ctrl[act_id] = tau[i]

    def apply_grip_ctrl() -> None:
        """Map gripper_fraction (0 closed, 1 open) to the two finger position
        actuators' ctrlranges."""
        for arm_id in range(2):
            frac = float(np.clip(gripper_fraction[arm_id], 0.0, 1.0))
            for i, act_id in enumerate(grip_act_ids[arm_id]):
                lo, hi = grip_ctrl_ranges[arm_id][i]
                # When frac=1 (open), pick the end of the range with larger |value|
                # which corresponds to fingers apart. The two finger joints have
                # opposite-sign ranges: (0, 0.04) for j1, (-0.04, 0) for j2. At
                # closed the joints are at 0 (range corner near 0); at open they
                # extend to the far end.
                target = lo + frac * (hi - lo) if abs(hi) > abs(lo) else hi + frac * (lo - hi)
                data.ctrl[act_id] = target

    def on_reset_settled() -> None:
        """After env.reset(), re-seat targets so the arms don't snap."""
        for arm_id in range(2):
            seat_target_at_current(arm_id)

    print("[teleop] launching mujoco passive viewer...")
    print(f"[teleop] sim_substeps/tick = {sim_substeps}  (sim_dt={model.opt.timestep})")
    print("[teleop] Ctrl-drag an end-effector body in the viewer to move XYZ only.")
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        # Drop into our own physics loop. We bypass env.step entirely so the
        # arm's joint torques come from our PD controller (no robosuite goal
        # reset, no gravity-drift).
        dt = 1.0 / args.control_hz
        while viewer.is_running():
            loop_start = time.time()

            sync_from_viewer_drag(viewer)
            clear_viewer_perturb_forces()
            for _ in range(sim_substeps):
                clear_viewer_perturb_forces()
                apply_arm_torques()
                apply_grip_ctrl()
                mujoco.mj_step(model, data)
                clear_viewer_perturb_forces()
            viewer.sync()
            clear_viewer_perturb_forces()

            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)


if __name__ == "__main__":
    main()
