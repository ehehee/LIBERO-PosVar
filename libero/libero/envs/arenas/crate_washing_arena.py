"""Custom arena that loads the bimanual crate-washing scene verbatim.

The scene XML (``assets/scenes/crate_washing/scenes/crate_washing.xml``) is a
self-contained MJCF describing the room, the washing-machine fixture, the
eleven-crate stack, and the robot platform. The two Panda arms are attached at
runtime by the LIBERO problem class via the standard robosuite
``mujoco_robots=[...]`` merge, so this arena class is a thin wrapper that just
loads the XML and exposes the body references we need elsewhere.
"""

import os

from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import xml_path_completion


_DEFAULT_REL_XML = "scenes/crate_washing/scenes/crate_washing.xml"


def _default_xml_path():
    """Absolute default path to the stripped crate-washing scene XML."""
    libero_assets_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "assets")
    )
    return os.path.join(libero_assets_dir, _DEFAULT_REL_XML)


class CrateWashingArena(Arena):
    """Self-contained crate-washing scene loaded from a single MJCF.

    Args:
        xml (str or None): Absolute filesystem path to the scene XML, or a
            path inside robosuite's own assets dir. If ``None`` the default
            LIBERO-shipped crate-washing scene under
            ``libero/libero/assets/scenes/crate_washing/scenes/`` is used.
    """

    def __init__(self, xml=None):
        if xml is None:
            full_path = _default_xml_path()
        else:
            full_path = xml_path_completion(xml)
        super().__init__(full_path)

        self.crate_machine_body = self.worldbody.find(
            "./body[@name='crate_machine']"
        )
        self.top_crate_body = self.worldbody.find(
            "./body[@name='crate_box_11']"
        )
        self.robot_platform_body = self.worldbody.find(
            "./body[@name='robot_platform']"
        )
