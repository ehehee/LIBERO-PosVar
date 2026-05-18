"""Bimanual sibling of :class:`libero.envs.bddl_base_domain.BDDLBaseDomain`.

This base subclasses robosuite's :class:`TwoArmEnv` (rather than ``SingleArmEnv``)
so two independent ``SingleArm`` robots are managed natively by robosuite. It
intentionally implements only the slice of BDDLBaseDomain's machinery needed by
the crate-washing task (BDDL parsing, problem-name assert, custom arena hookup,
per-arm base placement, two-arm state-vector helper, success hook). Anything
that depends on BDDL-declared :objects / :fixtures / :regions / placement
samplers is left as a no-op hook so single-arm LIBERO problems are unaffected.

The single global ``TASK_MAPPING`` from
:mod:`libero.envs.bddl_base_domain` is reused (and re-exported) so a
``@register_problem``-decorated subclass of this base lands in the same
registry that :class:`~libero.envs.env_wrapper.ControlEnv` looks up.
"""

import os

import numpy as np
from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from robosuite.models.tasks import ManipulationTask

import libero.envs.bddl_utils as BDDLUtils
from libero.envs.arenas import CrateWashingArena
from libero.envs.bddl_base_domain import TASK_MAPPING, register_problem  # noqa: F401

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class BimanualBDDLBaseDomain(TwoArmEnv):
    """BDDL-driven base for two-arm tasks.

    Args mirror :class:`BDDLBaseDomain` for the subset that applies; anything
    object/region-related is dropped because two-arm scenes in this stack are
    fully described by their MJCF.
    """

    def __init__(
        self,
        bddl_file_name,
        robots,
        env_configuration="single-arm-parallel",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mujoco",
        renderer_config=None,
        workspace_offset=(0.0, 0.0, 0.0),
        arena_type="crate_washing",
        scene_xml="scenes/crate_washing/scenes/crate_washing.xml",
        scene_properties=None,
        robot_base_positions=((1.74, -0.2, 0.76), (1.74, 0.2, 0.76)),
        robot_base_orientations=((0.0, 0.0, np.pi), (0.0, 0.0, np.pi)),
        **kwargs,
    ):
        self.workspace_offset = workspace_offset
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs

        self.objects_dict = {}
        self.fixtures_dict = {}
        self.object_sites_dict = {}
        self.object_states_dict = {}
        self.tracking_object_states_change = []
        self.objects = []
        self.fixtures = []

        self.custom_asset_dir = os.path.abspath(os.path.join(DIR_PATH, "../assets"))

        self.bddl_file_name = bddl_file_name
        self.parsed_problem = BDDLUtils.robosuite_parse_problem(self.bddl_file_name)
        self.obj_of_interest = self.parsed_problem["obj_of_interest"]

        self._assert_problem_name()

        self._arena_type = arena_type
        self._arena_xml = os.path.join(self.custom_asset_dir, scene_xml)
        self._arena_properties = dict(scene_properties or {})
        self._robot_base_positions = [tuple(p) for p in robot_base_positions]
        self._robot_base_orientations = [tuple(o) for o in robot_base_orientations]

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # BDDL helpers
    # ------------------------------------------------------------------

    def _assert_problem_name(self):
        assert (
            self.parsed_problem["problem_name"] == self.__class__.__name__.lower()
        ), (
            "Problem name mismatched: BDDL says "
            f"{self.parsed_problem['problem_name']!r}, class is "
            f"{self.__class__.__name__.lower()!r}."
        )

    @property
    def language_instruction(self):
        return self.parsed_problem["language"]

    # ------------------------------------------------------------------
    # Subclass hooks (default to no-op since the crate-washing scene is
    # fully baked into MJCF — no BDDL-driven object / fixture / region
    # processing is needed). Single-arm LIBERO problems override these.
    # ------------------------------------------------------------------

    def _load_fixtures_in_arena(self, mujoco_arena):
        return None

    def _load_objects_in_arena(self, mujoco_arena):
        return None

    def _load_sites_in_arena(self, mujoco_arena):
        return None

    def _add_placement_initializer(self):
        return None

    # ------------------------------------------------------------------
    # Reward / success
    # ------------------------------------------------------------------

    def reward(self, action=None):
        reward = 1.0 if self._check_success() else 0.0
        if self.reward_scale is not None:
            reward *= self.reward_scale / 1.0
        return reward

    def _check_success(self):
        """Subclasses override; default: never successful."""
        return False

    def _eval_predicate(self, state):
        """Mirror of :meth:`BDDLBaseDomain._eval_predicate` for subclasses that
        want to evaluate single-arm-style predicates against
        ``self.object_states_dict``. The crate-washing problem class doesn't
        use this (success is computed from raw body xpos), but keeping the
        method here means subclasses that later populate
        ``object_states_dict`` can ``super()._eval_predicate(state)``.
        """
        from libero.envs.predicates import eval_predicate_fn

        if len(state) == 3:
            return eval_predicate_fn(
                state[0],
                self.object_states_dict[state[1]],
                self.object_states_dict[state[2]],
            )
        if len(state) == 2:
            return eval_predicate_fn(state[0], self.object_states_dict[state[1]])
        return False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _setup_camera(self, mujoco_arena):
        """Configure LIBERO's canonical cameras on top of any defined in MJCF.

        The crate-washing scene already ships ``top`` and ``scene`` cameras,
        so we only add an ``agentview`` / ``frontview`` framed roughly at the
        workspace so wrappers that default to those names still produce a
        sensible image.
        """
        # Frame the workspace from front-left, looking back at the crate
        # stack. Positioned outside the robot platform (x ≈ +1.87) so the
        # arms and crate stack are both in frame. The orientation is a
        # look-at toward the two-arm grasp region (world ≈ (1.55, 0,
        # 0.95), between crate_box_11 @ x=1.38 and the robots @ x=1.74);
        # the previous quat looked back at the machine/wall and missed
        # the arms entirely.
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[1.0, -1.5, 1.6],
            quat=[0.816937, 0.5495927, -0.0975822, -0.1450502],
        )
        mujoco_arena.set_camera(
            camera_name="frontview",
            pos=[-0.5, 0.0, 1.6],
            quat=[0.5, 0.5, 0.5, 0.5],
        )

    def _load_model(self):
        """Builds the MuJoCo model: arena XML + two robot models merged in."""
        super()._load_model()

        if self._arena_type == "crate_washing":
            mujoco_arena = CrateWashingArena(
                xml=self._arena_xml, **self._arena_properties
            )
        else:
            raise ValueError(
                f"BimanualBDDLBaseDomain does not know how to build arena "
                f"of type {self._arena_type!r}."
            )

        # Place each Panda on the robot platform at the position the source
        # scene used. Skip robosuite's `base_xpos_offset[arena_type]` logic
        # (it only handles single-arm placements).
        for i, robot in enumerate(self.robots):
            if i < len(self._robot_base_positions):
                robot.robot_model.set_base_xpos(self._robot_base_positions[i])
            if i < len(self._robot_base_orientations):
                robot.robot_model.set_base_ori(self._robot_base_orientations[i])

        mujoco_arena.set_origin([0.0, 0.0, 0.0])

        self._setup_camera(mujoco_arena)

        # Hooks are no-ops in this base; subclasses may populate
        # objects_dict / fixtures_dict.
        self._load_fixtures_in_arena(mujoco_arena)
        self._load_objects_in_arena(mujoco_arena)
        self._load_sites_in_arena(mujoco_arena)

        self.objects = list(self.objects_dict.values())
        self.fixtures = list(self.fixtures_dict.values())

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.objects + self.fixtures,
        )

        for fixture in self.fixtures:
            self.model.merge_assets(fixture)

    # ------------------------------------------------------------------
    # References / step plumbing
    # ------------------------------------------------------------------

    def _setup_references(self):
        super()._setup_references()
        self.obj_body_id = {}
        for object_name, object_body in self.objects_dict.items():
            self.obj_body_id[object_name] = self.sim.model.body_name2id(
                object_body.root_body
            )
        for fixture_name, fixture_body in self.fixtures_dict.items():
            self.obj_body_id[fixture_name] = self.sim.model.body_name2id(
                fixture_body.root_body
            )

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)
        self._post_process()
        return reward, done, info

    def _post_process(self):
        for object_state in self.tracking_object_states_change:
            object_state.update_state()

    def step(self, action):
        obs, reward, done, info = super().step(action)
        done = self._check_success()
        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_robot_state_vector(self, obs):
        """Concatenate per-arm gripper qpos + eef pos + eef quat for both arms."""
        parts = []
        for pf in ("robot0_", "robot1_"):
            for key in (f"{pf}gripper_qpos", f"{pf}eef_pos", f"{pf}eef_quat"):
                if key in obs:
                    parts.append(np.asarray(obs[key]).ravel())
        return np.concatenate(parts) if parts else np.zeros(0)

    def is_fixture(self, object_name):
        return object_name in self.fixtures_dict

    def get_object(self, object_name):
        for query_dict in (self.objects_dict, self.fixtures_dict, self.object_sites_dict):
            if object_name in query_dict:
                return query_dict[object_name]
        raise KeyError(f"Unknown object {object_name!r}")
