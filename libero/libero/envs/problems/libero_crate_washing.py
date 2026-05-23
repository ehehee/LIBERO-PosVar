"""Bimanual crate-washing task: lift the top crate onto the washing machine.

Drop-in LIBERO-PRO problem class that pairs with the ``libero_crate_washing``
benchmark. The scene is fully described in MJCF
(``assets/scenes/crate_washing/scenes/crate_washing.xml``); two
``OnTheGroundPanda`` arms (no mount pedestal — the scene's own
``robot_platform`` body is the visible mount, matching the original
crate-washing-sim setup) are merged in by :class:`BimanualBDDLBaseDomain`.
Success is tracked as a
2-stage monotonic cursor, mirroring the popcorn-production pattern:

* **Stage 1 (Lifted)** — the free top crate (``crate_box_11``) is above the
  static stack top (z > 0.90 m).
* **Stage 2 (Placed)** — the crate is resting on the washing-machine table
  (xy inside the table footprint, z within a thin band around the table top
  at ~0.905 m).

Once a stage flips to True the cursor advances and never regresses, so a
transient lift -> drop still counts toward partial credit. The two raw stage
flags and the cursor are exposed via :attr:`crate_stage_idx` and
:attr:`crate_num_stages` so harnesses can report per-stage attrition.
"""

import numpy as np

from libero.envs.bimanual_bddl_base_domain import (
    BimanualBDDLBaseDomain,
    register_problem,
)

# Geometric thresholds derived directly from the scene XML:
#   - static stack of ten 0.075 m crates topped by the free crate; stack top
#     centre sits around z ≈ 0.825 m, so 0.90 m clears the stack with margin.
#   - the washing-machine top "table" geom is a thin box at pos.z = 0.855 m
#     with half-z 0.05, so its top surface is at z ≈ 0.905 m. The crate body
#     is 0.075 m thick, so its centre is ≈ 0.0375 m above the table top.
#   - the table footprint half-extents are 1.154 m × 0.504 m around the
#     world origin; we use slightly tighter xy bounds so a crate that just
#     dangles off the edge still doesn't register as placed.
_LIFTED_Z_MIN = 0.90
_PLACED_Z_MIN = 0.90
_PLACED_Z_MAX = 1.00
_PLACED_X_HALF = 1.154
_PLACED_Y_HALF = 0.504
_TOP_CRATE_BODY = "crate_box_11"
_GRIPPER_COLLISION_FRICTION = np.array([0.1, 0.005, 0.0001], dtype=np.float64)
_GRIPPER_COLLISION_SUFFIXES = (
    "finger1_collision",
    "finger1_pad_collision",
    "finger2_collision",
    "finger2_pad_collision",
)


@register_problem
class Libero_Crate_Washing(BimanualBDDLBaseDomain):
    """LIBERO problem: bimanual lift of the top crate onto the washing machine."""

    def __init__(self, bddl_file_name, *args, **kwargs):
        # Auto-double a single "Panda" entry into a bimanual pair, then wrap
        # each with OnTheGroundPanda so no RethinkMount pedestal is added.
        # The scene's own `robot_platform` body already provides the visible
        # mount, matching the original crate-washing-sim setup.
        robots = list(kwargs.get("robots", ["Panda"]))
        if len(robots) == 1:
            robots = robots * 2

        def _wrap(name: str) -> str:
            if name.startswith("OnTheGround") or name.startswith("Mounted"):
                return name
            return f"OnTheGround{name}"

        kwargs["robots"] = [_wrap(r) for r in robots]
        kwargs.setdefault("arena_type", "crate_washing")
        kwargs.setdefault(
            "scene_xml", "scenes/crate_washing/scenes/crate_washing.xml"
        )
        kwargs.setdefault("workspace_offset", (1.74, 0.0, 0.76))

        # Stage tracking is initialised before super().__init__ because the
        # robosuite reset chain invoked from there will eventually call
        # `_reset_internal` (which expects the attribute to exist).
        self._crate_stage_idx = 0

        super().__init__(bddl_file_name, *args, **kwargs)

    # ------------------------------------------------------------------
    # No BDDL-driven objects / fixtures / sites: the whole scene is in MJCF.
    # ------------------------------------------------------------------

    def _load_fixtures_in_arena(self, mujoco_arena):
        return None

    def _load_objects_in_arena(self, mujoco_arena):
        return None

    def _load_sites_in_arena(self, mujoco_arena):
        return None

    # ------------------------------------------------------------------
    # Stage tracking
    # ------------------------------------------------------------------

    def _reset_internal(self):
        super()._reset_internal()
        self._apply_gripper_collision_friction()
        self._crate_stage_idx = 0

    def _apply_gripper_collision_friction(self):
        """Crate-washing-only Panda finger contact tuning.

        The shared PandaGripper XML keeps its default friction. We only patch
        this compiled MuJoCo model instance so other LIBERO tasks are untouched.
        """
        model = self.sim.model
        for geom_id, name in enumerate(model.geom_names):
            if name.endswith(_GRIPPER_COLLISION_SUFFIXES):
                model.geom_friction[geom_id] = _GRIPPER_COLLISION_FRICTION

    def _top_crate_pos(self):
        body_id = self.sim.model.body_name2id(_TOP_CRATE_BODY)
        return np.asarray(self.sim.data.body_xpos[body_id], dtype=np.float64)

    def _stage_flags(self):
        """Raw per-stage Boolean checks from the current sim state."""
        pos = self._top_crate_pos()
        lifted = bool(pos[2] > _LIFTED_Z_MIN)
        placed = bool(
            _PLACED_Z_MIN < pos[2] < _PLACED_Z_MAX
            and abs(pos[0]) < _PLACED_X_HALF
            and abs(pos[1]) < _PLACED_Y_HALF
        )
        return [lifted, placed]

    def _check_success(self):
        flags = self._stage_flags()
        # Greedy monotonic advance: once a stage has been observed True we
        # don't roll the cursor back even if a later step violates it.
        while self._crate_stage_idx < len(flags) and flags[self._crate_stage_idx]:
            self._crate_stage_idx += 1
        return self._crate_stage_idx >= len(flags)

    def reward(self, action=None):
        """Partial sparse reward: fraction of monotonic stages reached.

        ``_check_success`` is still the strict binary completion signal used
        for pass/fail. Reward is more diagnostic:

        * 0.0: neither lift nor placement has been observed
        * 0.5: the top crate was lifted off the stack
        * 1.0: the top crate reached the table placement box
        """
        self._check_success()
        reward = float(self._crate_stage_idx) / float(self.crate_num_stages)
        if self.reward_scale is not None:
            reward *= self.reward_scale / 1.0
        return reward

    # ------------------------------------------------------------------
    # External read-only progress accessors (mirror popcorn convention)
    # ------------------------------------------------------------------

    @property
    def crate_stage_idx(self):
        """Largest stage index reached so far (monotonic), in ``[0, num_stages]``."""
        return self._crate_stage_idx

    @property
    def crate_num_stages(self):
        return 2

    @property
    def crate_stage_names(self):
        return ("lifted", "placed")
