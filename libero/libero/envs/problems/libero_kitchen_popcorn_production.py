"""Custom LIBERO problem with sequential, per-stage success tracking.

Vanilla LIBERO evaluates ``_check_success`` as a flat conjunction of the
BDDL ``:goal`` block (see ``libero_kitchen_tabletop_manipulation.py``).
That can't express tasks whose sub-goals are mutually exclusive across
time — e.g. "Turnon stove" then "Turnoff stove".

This problem class accepts a ``(Sequence ...)`` goal block. Each child
of the Sequence is a predicate (optionally wrapped in ``(Not ...)``)
that must be satisfied *in order*. ``_check_success`` advances a
per-episode stage cursor; once a stage is satisfied the cursor moves on
and never regresses, so transient violations of an earlier stage don't
disqualify the trajectory.

Example BDDL goal::

    (:goal
      (Sequence
        (On chefmate_8_frypan_1 flat_stove_1_cook_region)
        (Turnon flat_stove_1)
        (Turnoff flat_stove_1)
        (Not (On chefmate_8_frypan_1 flat_stove_1_cook_region))))
"""

from libero.envs.bddl_base_domain import register_problem
from libero.envs.problems.libero_kitchen_tabletop_manipulation import (
    Libero_Kitchen_Tabletop_Manipulation,
)
from libero.envs.regions import REGION_SAMPLERS, Libero100TableRegionSampler


# The region-sampler registry is keyed by the BDDL ``problem_name``
# (lower-cased class name), so reusing the kitchen sampler requires a
# direct entry — we don't inherit the parent's key.
REGION_SAMPLERS.setdefault("libero_kitchen_popcorn_production", {})[
    "kitchen_table"
] = Libero100TableRegionSampler


@register_problem
class Libero_Kitchen_Popcorn_Production(Libero_Kitchen_Tabletop_Manipulation):
    """Kitchen scene with ordered sub-goals (Sequence ...)."""

    def __init__(self, bddl_file_name, *args, **kwargs):
        # These get populated after the BDDL is parsed by the base init.
        self._popcorn_stages = []
        super().__init__(bddl_file_name, *args, **kwargs)
        self._popcorn_parse_stages()
        # Initialise the per-episode stage cursor; reset() will set it
        # back to zero before each episode.
        self._popcorn_stage_idx = 0

    # ---- BDDL goal parsing ------------------------------------------------

    def _popcorn_parse_stages(self):
        """Extract the ordered stage predicates from the parsed goal.

        ``package_predicates`` (bddl.parsing) returns the goal block as a
        list whose top item is either ``and`` (flattened away) or any
        other predicate name (kept whole). For our task the top item is
        ``sequence``, and its tail is the ordered stage list.
        """
        goal_state = self.parsed_problem["goal_state"]
        if not goal_state:
            return
        head = goal_state[0]
        if (
            isinstance(head, list)
            and head
            and isinstance(head[0], str)
            and head[0].lower() == "sequence"
        ):
            self._popcorn_stages = list(head[1:])
        else:
            # No Sequence wrapper — fall back to default conjunction
            # semantics by leaving _popcorn_stages empty.
            self._popcorn_stages = []

    # ---- Stage cursor reset ----------------------------------------------

    def _reset_internal(self):
        super()._reset_internal()
        self._popcorn_stage_idx = 0

    # ---- Per-step success check ------------------------------------------

    def _eval_stage(self, stage):
        """Evaluate one stage, handling ``(Not ...)`` wrappers."""
        if (
            isinstance(stage, list)
            and stage
            and isinstance(stage[0], str)
            and stage[0].lower() == "not"
        ):
            inner = stage[1]
            return not self._eval_predicate(inner)
        return self._eval_predicate(stage)

    def _check_success(self):
        """Walk the stage list; advance cursor on each satisfied stage.

        Returns True only when every stage has been observed in order.
        Falls back to the base conjunction check when no Sequence goal
        was supplied (e.g. when this class is loaded with a plain
        kitchen BDDL — preserves drop-in compatibility).
        """
        if not self._popcorn_stages:
            return super()._check_success()

        # Greedily advance through any stages currently satisfied. Once
        # advanced, we never roll back — the trajectory has *observed*
        # the milestone, which is the contract callers want.
        while self._popcorn_stage_idx < len(self._popcorn_stages):
            stage = self._popcorn_stages[self._popcorn_stage_idx]
            try:
                satisfied = self._eval_stage(stage)
            except Exception:
                satisfied = False
            if not satisfied:
                break
            self._popcorn_stage_idx += 1

        return self._popcorn_stage_idx >= len(self._popcorn_stages)

    # ---- External progress query (handy for evaluators / loggers) --------

    @property
    def popcorn_stage_idx(self):
        return self._popcorn_stage_idx

    @property
    def popcorn_num_stages(self):
        return len(self._popcorn_stages)
