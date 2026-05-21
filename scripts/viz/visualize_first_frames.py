"""Render one settled frame of one task per variance suite as still images.

This mirrors the render path of ``scripts/render_permuted_trials.py`` (and of
the runtime's ``SaveVideo`` handler) -- ``env.reset()``,
``env.set_init_state(init_states[0])``, step a zero action for a while so the
dropped objects settle onto the floor, ``np.flipud(obs[...])``, ``imageio`` --
but writes a single PNG per (suite, camera) instead of an MP4, at the same
resolution the runtime records rollout video (800 x 512 by default; see
``vos/compose/config.py``).

(The baked ``*.pruned_init`` states put each object at the sampler's small
``z_offset`` above the floor -- mid-drop -- so without the settle loop the
objects look like they're floating / clipping through the floor.)

For each suite it renders the first task's first init state from:
  * ``agentview``           -- the camera the runtime captures,
  * ``birdview``            -- a re-aimed top-down view at the same resolution,
  * ``sideview``            -- the scene XML's off-axis side view, useful for
                               seeing the agent + wrist cameras as physical
                               ZED 2i bodies in 3D (see the XML patches in
                               panda/robot.xml + libero_floor_base_style.xml), and
  * ``robot0_eye_in_hand``  -- the wrist camera bolted to the gripper.

Output: ``<out-dir>/<suite>__agentview.png``, ``<out-dir>/<suite>__topdown.png``,
``<out-dir>/<suite>__wrist.png``, ``<out-dir>/<suite>__sideview.png``,
``<out-dir>/<suite>__sideview_with_axes.png`` (the sideview with a labelled
world-frame triad at the world origin *and* the agentview + wrist camera
frames drawn at their lens positions, all in X=red, Y=green, Z=blue with the
camera frames using the OpenCV / ROS optical convention +x right, +y down,
+z forward), and
``<out-dir>/<suite>__agentview_with_wrist.png`` (the agentview with the wrist
camera's mount projected in as a circle + arrow showing which way it's
looking). ``out-dir`` defaults to ``scripts/viz/``.

Run with the project's interpreter and an offscreen GL backend, e.g.:

    MUJOCO_GL=egl python scripts/viz/visualize_first_frames.py
    MUJOCO_GL=egl python scripts/viz/visualize_first_frames.py \
        --suites libero_object_all_variance --task 0 --settle 60 \
        --render-width 1920 --render-height 1080
"""

import argparse
import os
import sys

# This repo is usually run alongside an editable install of upstream LIBERO-PRO
# (which lacks the variance suites and the brightened scene XMLs). Force `import
# libero` to resolve to *this* checkout before anything pulls it in.
_THIS_DIR = os.path.dirname(os.path.realpath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
_LIBERO_PKG_ROOT = os.path.join(_REPO_ROOT, "libero")  # holds the `libero` package
for _p in (_REPO_ROOT, _LIBERO_PKG_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# MuJoCo needs the GL backend chosen before it is imported (transitively, below).
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import imageio.v2 as imageio


def _install_numpy_unpickle_compat():
    """Let an old NumPy (<2.0) unpickle arrays saved by NumPy >=2.0.

    NumPy 2.0 renamed the private ``numpy.core`` package to ``numpy._core``;
    some of this repo's ``*.pruned_init`` files were pickled under 2.x. Alias
    the new module paths back onto ``numpy.core`` so ``torch.load`` succeeds.
    """
    if hasattr(np, "_core"):  # NumPy >= 2.0: nothing to do.
        return
    import numpy.core  # noqa: F401
    for sub in ("multiarray", "umath", "numeric", "numerictypes",
                "_multiarray_umath", "_exceptions"):
        try:
            __import__(f"numpy.core.{sub}")
        except ImportError:
            pass
    sys.modules.setdefault("numpy._core", sys.modules["numpy.core"])
    for name, mod in list(sys.modules.items()):
        if name.startswith("numpy.core."):
            sys.modules.setdefault("numpy._core." + name[len("numpy.core."):], mod)


_install_numpy_unpickle_compat()

try:
    from libero.libero.envs import OffScreenRenderEnv
except ModuleNotFoundError:
    from libero.envs import OffScreenRenderEnv


BDDL_ROOT = os.path.join(_LIBERO_PKG_ROOT, "libero", "bddl_files")
INIT_ROOT = os.path.join(_LIBERO_PKG_ROOT, "libero", "init_files")

AGENT_CAM = "agentview"
TOPDOWN_CAM = "birdview"
SIDE_CAM = "sideview"  # off to +y, looking back at the workspace
EYE_CAM = "robot0_eye_in_hand"  # wrist camera bolted to the gripper

# Marker overlay: where the wrist camera projects into the agentview image.
WRIST_MARKER_RADIUS = 9            # px, outer ring radius
WRIST_MARKER_THICKNESS = 2         # px, ring stroke
WRIST_MARKER_COLOR = (255, 60, 60)   # red, BGR-agnostic since we write RGB
WRIST_ARROW_LEN_M = 0.10           # meters, world-frame arrow length along view axis
WRIST_ARROW_THICKNESS = 2          # px

# Axis triads drawn on the sideview image. Three frames are stacked into the
# same image:
#   * the world frame, anchored at the world origin, in big bold arrows so it
#     orients the reader;
#   * the agentview ZED camera frame, anchored at its lens (OpenCV / ROS
#     convention: +x = image-right, +y = image-down, +z = forward, out of the
#     lens), in smaller arrows so it doesn't crowd the world triad;
#   * the wrist ZED camera frame, same convention, same scale as the agent
#     frame.
# Each frame uses the standard (X=red, Y=green, Z=blue) colour scheme. The
# camera frames carry a name label near the origin so the reader can tell which
# is which.
AXIS_X_COLOR = (230, 40, 40)    # red
AXIS_Y_COLOR = (40, 200, 60)    # green
AXIS_Z_COLOR = (60, 120, 255)   # blue
AXIS_ARROW_HEAD_LEN_PX = 18
AXIS_ARROW_HEAD_HALF_ANGLE = np.deg2rad(22)

# World frame: drawn at the world origin (raised a hair so the dot clears the
# floor texture). Larger and thicker than the camera frames.
WORLD_AXIS_LEN_M = 0.30
WORLD_AXIS_THICKNESS = 4
WORLD_AXIS_ORIGIN_RADIUS = 6
WORLD_AXIS_ORIGIN_Z = 0.01
WORLD_AXIS_LABEL_PX = 26

# Per-camera frame: drawn at the camera's world position. Smaller arrows so the
# three frames coexist legibly in the same image.
CAM_AXIS_LEN_M = 0.12
CAM_AXIS_THICKNESS = 3
CAM_AXIS_ORIGIN_RADIUS = 4
CAM_LABEL_PX = 18

# Runtime rollout-video defaults (vos/compose/config.py); honour the same env
# vars the sim_bridge process reads so a one-off `VOS_RENDER_WIDTH=... ` works.
DEFAULT_RENDER_WIDTH = int(os.environ.get("VOS_RENDER_WIDTH", "800"))
DEFAULT_RENDER_HEIGHT = int(os.environ.get("VOS_RENDER_HEIGHT", "512"))

# `birdview` ships pointed straight down from far above, so the workspace is a
# tiny patch in a sea of floor. Re-aim it: stay high (above the arm) but narrow
# the FOV and recentre over where the objects actually sit.
TOPDOWN_POS = (-0.05, 0.0, 3.0)
TOPDOWN_FOVY = 22.0

# `sideview` ships from a high oblique angle that reads almost top-down. Re-aim
# to a 3/4 perspective from the front-right that frames the robot + both ZEDs
# (agentview pedestal on the floor, wrist mounted on the gripper) + the
# workspace simultaneously, so the world-frame triad and the per-camera frames
# all land in the same legible image.
SIDEVIEW_POS = (1.4, 1.0, 1.0)
SIDEVIEW_LOOK_AT = (0.05, 0.05, 0.20)
SIDEVIEW_FOVY = 55.0

# Variance suites contributed by this repo (those that ship both BDDL and
# baked-init files). Override on the command line with --suites.
DEFAULT_SUITES = [
    "libero_object_target_xy_variance",
    "libero_object_target_pos_var20x20",
    "libero_object_permutation",
    "libero_object_target_permutation_variance",
    "libero_object_basket_swap",
    "libero_object_target_basket_swap_variance",
    "libero_object_target_combined_variance",
    "libero_object_all_variance",
]


def task_of_suite(suite, task_idx):
    """Return (bddl_path, init_path_or_None, task_name) for a suite's task.

    Tasks are ordered by BDDL filename. Returns ``None`` if the suite or the
    requested task index isn't present in this checkout.
    """
    bddl_dir = os.path.join(BDDL_ROOT, suite)
    if not os.path.isdir(bddl_dir):
        return None
    bddls = sorted(f for f in os.listdir(bddl_dir) if f.endswith(".bddl"))
    if not bddls or task_idx >= len(bddls):
        return None
    task_name = bddls[task_idx][: -len(".bddl")]
    bddl_path = os.path.join(bddl_dir, bddls[task_idx])

    init_path = None
    init_dir = os.path.join(INIT_ROOT, suite)
    for ext in (".pruned_init", ".init"):  # benchmark uses .pruned_init
        cand = os.path.join(init_dir, task_name + ext)
        if os.path.exists(cand):
            init_path = cand
            break
    return bddl_path, init_path, task_name


def _aim_topdown_camera(env, pos, fovy):
    """Move/zoom the ``birdview`` camera (a no-op if both args are None)."""
    if pos is None and fovy is None:
        return
    model = env.sim.model
    cid = model.camera_name2id(TOPDOWN_CAM)
    if pos is not None:
        model.cam_pos[cid] = np.asarray(pos, dtype=model.cam_pos.dtype)
    if fovy is not None:
        model.cam_fovy[cid] = float(fovy)


def _mat_to_quat(R):
    """Convert a 3x3 rotation matrix to a (w, x, y, z) quaternion (MuJoCo order)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return np.array([
            0.25 * s,
            (R[2, 1] - R[1, 2]) / s,
            (R[0, 2] - R[2, 0]) / s,
            (R[1, 0] - R[0, 1]) / s,
        ])
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return np.array([
            (R[2, 1] - R[1, 2]) / s,
            0.25 * s,
            (R[0, 1] + R[1, 0]) / s,
            (R[0, 2] + R[2, 0]) / s,
        ])
    if R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return np.array([
            (R[0, 2] - R[2, 0]) / s,
            (R[0, 1] + R[1, 0]) / s,
            0.25 * s,
            (R[1, 2] + R[2, 1]) / s,
        ])
    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return np.array([
        (R[1, 0] - R[0, 1]) / s,
        (R[0, 2] + R[2, 0]) / s,
        (R[1, 2] + R[2, 1]) / s,
        0.25 * s,
    ])


def _aim_sideview_camera(env, pos, look_at, fovy):
    """Place ``sideview`` at ``pos`` looking at ``look_at`` (world +z is up)."""
    if pos is None and look_at is None and fovy is None:
        return
    model = env.sim.model
    cid = model.camera_name2id(SIDE_CAM)
    if pos is not None and look_at is not None:
        pos = np.asarray(pos, dtype=np.float64)
        look_at = np.asarray(look_at, dtype=np.float64)
        # MuJoCo: camera looks down its own -z; image-up is +y_cam.
        z_cam = pos - look_at
        z_cam /= np.linalg.norm(z_cam) or 1.0
        world_up = np.array([0.0, 0.0, 1.0])
        x_cam = np.cross(world_up, z_cam)
        x_cam /= np.linalg.norm(x_cam) or 1.0
        y_cam = np.cross(z_cam, x_cam)
        R = np.column_stack([x_cam, y_cam, z_cam])
        model.cam_pos[cid] = pos
        model.cam_quat[cid] = _mat_to_quat(R)
    if fovy is not None:
        model.cam_fovy[cid] = float(fovy)


def _draw_circle(img, cy, cx, radius, thickness, color):
    """Draw an unfilled circle (ring) in-place; clipped to image bounds."""
    H, W = img.shape[:2]
    r_outer = radius
    r_inner = max(0, radius - thickness)
    y0 = max(0, cy - r_outer)
    y1 = min(H, cy + r_outer + 1)
    x0 = max(0, cx - r_outer)
    x1 = min(W, cx + r_outer + 1)
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    mask = (d2 <= r_outer * r_outer) & (d2 >= r_inner * r_inner)
    img[y0:y1, x0:x1][mask] = color


def _draw_line(img, y0, x0, y1, x1, thickness, color):
    """Naive line rasteriser (samples N points; thickens with a square kernel)."""
    H, W = img.shape[:2]
    steps = int(max(abs(y1 - y0), abs(x1 - x0))) + 1
    ts = np.linspace(0.0, 1.0, steps * 2)
    ys = (y0 + (y1 - y0) * ts).round().astype(np.int64)
    xs = (x0 + (x1 - x0) * ts).round().astype(np.int64)
    half = max(1, thickness) // 2
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            yy = np.clip(ys + dy, 0, H - 1)
            xx = np.clip(xs + dx, 0, W - 1)
            img[yy, xx] = color


def _world_axes_endpoints(length, origin_z=WORLD_AXIS_ORIGIN_Z):
    """Return (origin, x_tip, y_tip, z_tip) in world coords for the world frame.

    The origin sits slightly above the floor (``origin_z`` metres) so the dot
    and arrow bases aren't lost in the floor texture.
    """
    origin = np.array([0.0, 0.0, origin_z], dtype=np.float64)
    return (
        origin,
        origin + np.array([length, 0.0, 0.0]),
        origin + np.array([0.0, length, 0.0]),
        origin + np.array([0.0, 0.0, length]),
    )


def _draw_arrow(img, y0, x0, y1, x1, thickness, color,
                head_len=AXIS_ARROW_HEAD_LEN_PX,
                head_half_angle=AXIS_ARROW_HEAD_HALF_ANGLE):
    """Draw a straight line plus a small triangular arrowhead at (y1, x1)."""
    _draw_line(img, y0, x0, y1, x1, thickness, color)
    dy, dx = y1 - y0, x1 - x0
    length = float(np.hypot(dy, dx))
    if length < 1.0:
        return
    # Unit vector along the shaft, pointing toward the tip.
    uy, ux = dy / length, dx / length
    cos_a, sin_a = float(np.cos(head_half_angle)), float(np.sin(head_half_angle))
    # The barbs are at angle ±head_half_angle from the *reverse* shaft direction.
    for sign in (+1.0, -1.0):
        # Rotate (-uy, -ux) by ±head_half_angle.
        by = -uy * cos_a - sign * (-ux) * sin_a
        bx = -ux * cos_a + sign * (-uy) * sin_a
        _draw_line(img, y1, x1, y1 + head_len * by, x1 + head_len * bx,
                   thickness, color)


def _project_point(world_to_pixel, p_world):
    """Project a 3D world point through the agentview transform.

    Returns ``(px, py, in_front)``. ``in_front`` is False if the point lies
    behind the agentview camera (in which case px/py are still finite but the
    pinhole math has flipped them through the focal point).
    """
    p_h = np.array([p_world[0], p_world[1], p_world[2], 1.0])
    pix = world_to_pixel @ p_h  # shape [4]
    w = pix[2]
    if w == 0:
        return None, None, False
    return pix[0] / w, pix[1] / w, w > 0


def _load_label_font(size_px):
    """Find a TTF on the system for axis labels; fall back to PIL's default."""
    from PIL import ImageFont
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size_px)
            except OSError:
                continue
    return ImageFont.load_default()


def _draw_text_with_outline(img, text, cy, cx, font, color,
                            outline=(0, 0, 0), outline_px=2):
    """Stamp ``text`` onto ``img`` centred at (cy, cx) with a contrasting outline."""
    from PIL import Image, ImageDraw

    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    # Anchor text by its visible top-left so PIL's font ascent/descent doesn't
    # offset short ASCII glyphs (X/Y/Z all sit on the baseline).
    px = int(round(cx - text_w / 2 - bbox[0]))
    py = int(round(cy - text_h / 2 - bbox[1]))
    for dy in range(-outline_px, outline_px + 1):
        for dx in range(-outline_px, outline_px + 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((px + dx, py + dy), text, font=font, fill=outline)
    draw.text((px, py), text, font=font, fill=color)
    img[:] = np.asarray(pil)


def _camera_frame_axes_world(sim, cam_name, length):
    """Return (origin, x_tip, y_tip, z_tip) in world coords for a camera frame.

    Axes follow the OpenCV / ROS convention:
        +x = image-right, +y = image-down, +z = forward (out of the lens).
    MuJoCo's native cam_xmat columns are (+x_right, +y_up, +z_back), so we
    negate the y and z columns when reading.
    """
    cid = sim.model.camera_name2id(cam_name)
    cam_pos = np.asarray(sim.data.cam_xpos[cid], dtype=np.float64)
    R = np.asarray(sim.data.cam_xmat[cid], dtype=np.float64).reshape(3, 3)
    return (
        cam_pos,
        cam_pos + length * R[:, 0],   # +x_cv = +x_mj
        cam_pos - length * R[:, 1],   # +y_cv = -y_mj  (down)
        cam_pos - length * R[:, 2],   # +z_cv = -z_mj  (forward, out of lens)
    )


def _draw_frame(out, world_to_pixel, origin_w, x_tip_w, y_tip_w, z_tip_w,
                axis_thickness, origin_radius, label_offset_px,
                axis_label_font=None, axis_labels=("X", "Y", "Z"),
                name=None, name_font=None, name_color=(255, 255, 255),
                arrow_head_len=AXIS_ARROW_HEAD_LEN_PX):
    """Project + draw a 3-axis frame (X=red, Y=green, Z=blue) onto ``out``.

    ``origin_w`` and the three ``*_tip_w`` are world-frame points; the function
    projects them through ``world_to_pixel`` (which returns pixel coords in
    OpenCV / top-down convention -- matching the np.flipud-applied saved
    image), draws an arrow per axis, then stamps axis-tip labels and -- if
    ``name`` is set -- a frame name near the origin.

    Returns True if the frame was drawn, False if the origin was behind the
    sideview camera (in which case nothing is rendered for this frame).
    """
    def to_img(world_pt):
        x, y, in_front = _project_point(world_to_pixel, world_pt)
        return int(round(y)), int(round(x)), in_front

    opy, opx, in_front = to_img(origin_w)
    if not in_front:
        return False

    axes = (
        (axis_labels[0], x_tip_w, AXIS_X_COLOR),
        (axis_labels[1], y_tip_w, AXIS_Y_COLOR),
        (axis_labels[2], z_tip_w, AXIS_Z_COLOR),
    )

    # Arrows first, then the origin dot on top so the triad reads as one frame.
    tip_pixels = []
    for _, tip_w, color in axes:
        tpy, tpx, tip_in_front = to_img(tip_w)
        if not tip_in_front:
            tip_pixels.append(None)
            continue
        _draw_arrow(out, opy, opx, tpy, tpx, axis_thickness, color,
                    head_len=arrow_head_len)
        tip_pixels.append((tpy, tpx))
    _draw_circle(out, opy, opx, origin_radius, max(2, axis_thickness // 2),
                 (255, 255, 255))

    # Axis-tip labels: push each label outward along its arrow's screen direction.
    if axis_label_font is not None:
        for (label, _, color), tp in zip(axes, tip_pixels):
            if tp is None:
                continue
            tpy, tpx = tp
            dy, dx = float(tpy - opy), float(tpx - opx)
            norm = float(np.hypot(dy, dx)) or 1.0
            ly = int(round(tpy + label_offset_px * dy / norm))
            lx = int(round(tpx + label_offset_px * dx / norm))
            _draw_text_with_outline(out, label, ly, lx, axis_label_font, color)

    # Frame name (e.g. "agent", "wrist") just below-right of the origin so it
    # doesn't sit on top of the axes. The offset is keyed off the font size so
    # the label clears the origin dot at any rendering resolution.
    if name and name_font is not None:
        try:
            font_size = int(getattr(name_font, "size", 18) or 18)
        except (TypeError, ValueError):
            font_size = 18
        offset = origin_radius + max(8, font_size // 2)
        outline_px = max(2, font_size // 12)
        _draw_text_with_outline(
            out, name,
            cy=opy + offset,
            cx=opx + offset,
            font=name_font, color=name_color, outline_px=outline_px,
        )
    return True


def annotate_frames_in_sideview(side, env, side_w, side_h):
    """Overlay world frame + agentview cam frame + wrist cam frame on the sideview.

    All three frames share the (X=red, Y=green, Z=blue) colour scheme. The
    world triad is drawn larger and at the world origin to anchor the reader;
    the camera triads are drawn smaller at each camera's world pose and use
    the OpenCV / ROS optical-frame convention (+x right, +y down, +z forward).

    Pixel-fixed overlay sizes (arrowhead barbs, origin dots, label fonts) are
    scaled by ``side_h / DEFAULT_RENDER_HEIGHT`` so the triads remain visually
    proportional whether the image is rendered at the runtime-default 800x512
    or at a poster-sized 4096-wide.
    """
    from robosuite.utils.camera_utils import get_camera_transform_matrix

    sim = env.env.sim
    world_to_pixel = get_camera_transform_matrix(sim, SIDE_CAM, side_h, side_w)

    out = side.copy()
    s = side_h / DEFAULT_RENDER_HEIGHT
    world_thick = max(1, int(round(WORLD_AXIS_THICKNESS * s)))
    world_origin_r = max(2, int(round(WORLD_AXIS_ORIGIN_RADIUS * s)))
    cam_thick = max(1, int(round(CAM_AXIS_THICKNESS * s)))
    cam_origin_r = max(2, int(round(CAM_AXIS_ORIGIN_RADIUS * s)))
    arrow_head_len = max(4, int(round(AXIS_ARROW_HEAD_LEN_PX * s)))
    name_font = _load_label_font(max(10, int(round(CAM_LABEL_PX * s))))

    # World frame at the origin, big and bold. No X/Y/Z tip labels -- the
    # arrow colours alone carry the convention, and skipping the glyphs keeps
    # the world triad uncluttered next to the camera frames.
    origin_w, xtip_w, ytip_w, ztip_w = _world_axes_endpoints(WORLD_AXIS_LEN_M)
    drew = _draw_frame(
        out, world_to_pixel, origin_w, xtip_w, ytip_w, ztip_w,
        axis_thickness=world_thick,
        origin_radius=world_origin_r,
        label_offset_px=arrow_head_len + max(2, int(round(8 * s))),
        axis_label_font=None,
        name="world", name_font=name_font,
        arrow_head_len=arrow_head_len,
    )
    if not drew:
        print("  [warn] world origin is behind the sideview; skipping world frame")

    # Camera frames at each camera's lens, in OpenCV optical-frame convention.
    for cam_name, label in ((AGENT_CAM, "agent"), (EYE_CAM, "wrist")):
        origin_w, xtip_w, ytip_w, ztip_w = _camera_frame_axes_world(
            sim, cam_name, CAM_AXIS_LEN_M,
        )
        drew = _draw_frame(
            out, world_to_pixel, origin_w, xtip_w, ytip_w, ztip_w,
            axis_thickness=cam_thick,
            origin_radius=cam_origin_r,
            label_offset_px=arrow_head_len + max(1, int(round(4 * s))),
            # No axis-tip labels on the smaller cam frames -- the world triad
            # already tells the reader what red/green/blue mean, and stamping
            # six more X/Y/Z glyphs would crowd the image. The frame *name*
            # is still drawn near the origin.
            axis_label_font=None,
            name=label, name_font=name_font,
            arrow_head_len=arrow_head_len,
        )
        if not drew:
            print(f"  [warn] {cam_name} is behind the sideview; skipping its frame")
    return out


def annotate_wrist_in_agentview(agent, env, width, height):
    """Project the wrist camera's pose into the agentview and draw a marker."""
    from robosuite.utils.camera_utils import get_camera_transform_matrix

    sim = env.env.sim
    model, data = sim.model, sim.data
    wrist_id = model.camera_name2id(EYE_CAM)
    cam_pos = np.asarray(data.cam_xpos[wrist_id], dtype=np.float64)
    # MuJoCo camera convention: -Z is the optical axis (the direction the camera looks).
    cam_mat = np.asarray(data.cam_xmat[wrist_id], dtype=np.float64).reshape(3, 3)
    view_dir = cam_mat @ np.array([0.0, 0.0, -1.0])
    tip = cam_pos + WRIST_ARROW_LEN_M * view_dir

    world_to_pixel = get_camera_transform_matrix(sim, AGENT_CAM, height, width)

    bx, by, in_front = _project_point(world_to_pixel, cam_pos)
    tx, ty, tip_in_front = _project_point(world_to_pixel, tip)
    if not in_front:
        print(f"  [warn] wrist camera is behind the agentview camera; skipping marker")
        return agent

    out = agent.copy()
    # ``get_camera_transform_matrix`` returns pixel y in the OpenCV / top-down
    # convention, which already matches the ``np.flipud``-applied saved image.
    base_py = int(round(by))
    base_px = int(round(bx))
    if tip_in_front:
        tip_py = int(round(ty))
        tip_px = int(round(tx))
        _draw_line(out, base_py, base_px, tip_py, tip_px,
                   WRIST_ARROW_THICKNESS, WRIST_MARKER_COLOR)
    _draw_circle(out, base_py, base_px,
                 WRIST_MARKER_RADIUS, WRIST_MARKER_THICKNESS, WRIST_MARKER_COLOR)
    return out


def render_settled_frame(bddl_path, init_path, width, height, settle,
                         topdown_pos, topdown_fovy):
    """Render the agentview, re-aimed birdview, and wrist images after settle."""
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_names=[AGENT_CAM, EYE_CAM, TOPDOWN_CAM, SIDE_CAM],
        camera_heights=height,
        camera_widths=width,
    )
    try:
        obs = env.reset()
        # Pre-size MuJoCo's offscreen framebuffer to the requested render
        # resolution *before* the first sim.render call. Robosuite's
        # ``update_offscreen_size`` rebuilds the MjrContext on the fly when a
        # render is requested larger than the current FBO, which leaves model
        # textures unbound on some drivers (every camera then reads back as
        # all zeros). Setting the model's global offwidth/offheight up front
        # and forcing a context rebuild here avoids that race.
        sim = env.env.sim
        if (sim.model.vis.global_.offwidth < width
                or sim.model.vis.global_.offheight < height):
            sim._render_context_offscreen.update_offscreen_size(width, height)
        # reset() does a hard reset (rebuilds the sim from XML), so re-aim after.
        _aim_topdown_camera(env, topdown_pos, topdown_fovy)
        _aim_sideview_camera(env, SIDEVIEW_POS, SIDEVIEW_LOOK_AT, SIDEVIEW_FOVY)
        if init_path is not None:
            try:
                init_states = torch.load(init_path, weights_only=False)
                obs = env.set_init_state(init_states[0])
            except Exception as exc:  # noqa: BLE001 - keep going with the reset state
                print(f"  [warn] could not load {os.path.basename(init_path)} "
                      f"({exc}); using the post-reset state")
        # Step a zero action so the mid-drop objects fall and settle, exactly
        # like render_permuted_trials.py's settle loop.
        noop = np.zeros(env.env.action_dim, dtype=np.float64)
        for _ in range(settle):
            obs, _, _, _ = env.step(noop)
        # Re-apply the sideview pose: set_init_state + step() pipelines elsewhere
        # in libero/robosuite have been observed to overwrite cam_pos/cam_quat
        # for the sideview camera on the qpos path, leaving the obs-baked
        # sideview at the XML default (which is nearly top-down). Re-aim and
        # force a forward pass before pulling fresh pixels through sim.render.
        _aim_sideview_camera(env, SIDEVIEW_POS, SIDEVIEW_LOOK_AT, SIDEVIEW_FOVY)
        env.env.sim.forward()
        # MuJoCo renders bottom-up; flip vertically like render_permuted_trials.py.
        agent = np.flipud(obs[f"{AGENT_CAM}_image"]).astype(np.uint8)
        topdown = np.flipud(obs[f"{TOPDOWN_CAM}_image"]).astype(np.uint8)
        wrist = np.flipud(obs[f"{EYE_CAM}_image"]).astype(np.uint8)
        # Pull the sideview straight from the offscreen renderer so it reflects
        # the just-re-aimed cam_pos/cam_quat instead of the stale obs entry.
        side_raw = env.env.sim.render(width=width, height=height, camera_name=SIDE_CAM)
        side = np.flipud(side_raw).astype(np.uint8)
        # Overlays must be computed while env (and sim.data) is still alive.
        agent_with_wrist = annotate_wrist_in_agentview(agent, env, width, height)
        side_with_axes = annotate_frames_in_sideview(
            side, env, side_w=width, side_h=height,
        )
    finally:
        env.close()
    return agent, topdown, wrist, side, agent_with_wrist, side_with_axes


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--suites", nargs="+", default=DEFAULT_SUITES,
                        help="suite names to render (default: this repo's variance suites)")
    parser.add_argument("--task", type=int, default=0,
                        help="task index within each suite (BDDLs sorted by name; default 0)")
    parser.add_argument("--settle", type=int, default=40,
                        help="zero-action steps to let dropped objects settle (default 40)")
    parser.add_argument("--render-width", type=int, default=DEFAULT_RENDER_WIDTH,
                        help=f"render width (default {DEFAULT_RENDER_WIDTH}, matches runtime video)")
    parser.add_argument("--render-height", type=int, default=DEFAULT_RENDER_HEIGHT,
                        help=f"render height (default {DEFAULT_RENDER_HEIGHT}, matches runtime video)")
    parser.add_argument("--topdown-pos", type=float, nargs=3, default=list(TOPDOWN_POS),
                        metavar=("X", "Y", "Z"), help="birdview camera position")
    parser.add_argument("--topdown-fovy", type=float, default=TOPDOWN_FOVY,
                        help="birdview camera vertical FOV in degrees")
    parser.add_argument("--raw-topdown", action="store_true",
                        help="leave the birdview camera at its scene default")
    parser.add_argument("--out-dir", default=_THIS_DIR, help="where to write the PNGs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    topdown_pos = None if args.raw_topdown else args.topdown_pos
    topdown_fovy = None if args.raw_topdown else args.topdown_fovy

    rendered = 0
    for suite in args.suites:
        info = task_of_suite(suite, args.task)
        if info is None:
            print(f"[skip] {suite}: no task #{args.task} under {os.path.join(BDDL_ROOT, suite)}")
            continue
        bddl_path, init_path, task_name = info
        if init_path is None:
            print(f"[warn] {suite}: no baked init file for '{task_name}'; "
                  f"using the post-reset state instead")
        print(f"[render] {suite}  task[{args.task}]={task_name}  "
              f"@ {args.render_width}x{args.render_height}  settle={args.settle}")
        agent, topdown, wrist, side, agent_with_wrist, side_with_axes = render_settled_frame(
            bddl_path, init_path, args.render_width, args.render_height, args.settle,
            topdown_pos, topdown_fovy,
        )
        agent_path = os.path.join(args.out_dir, f"{suite}__agentview.png")
        topdown_path = os.path.join(args.out_dir, f"{suite}__topdown.png")
        wrist_path = os.path.join(args.out_dir, f"{suite}__wrist.png")
        side_path = os.path.join(args.out_dir, f"{suite}__sideview.png")
        side_axes_path = os.path.join(args.out_dir, f"{suite}__sideview_with_axes.png")
        marker_path = os.path.join(args.out_dir, f"{suite}__agentview_with_wrist.png")
        imageio.imwrite(agent_path, agent)
        imageio.imwrite(topdown_path, topdown)
        imageio.imwrite(wrist_path, wrist)
        imageio.imwrite(side_path, side)
        imageio.imwrite(side_axes_path, side_with_axes)
        imageio.imwrite(marker_path, agent_with_wrist)
        print(f"  saved {agent_path}  ({agent.shape[1]}x{agent.shape[0]})")
        print(f"  saved {topdown_path}  ({topdown.shape[1]}x{topdown.shape[0]})")
        print(f"  saved {wrist_path}  ({wrist.shape[1]}x{wrist.shape[0]})")
        print(f"  saved {side_path}  ({side.shape[1]}x{side.shape[0]})")
        print(f"  saved {side_axes_path}  ({side_with_axes.shape[1]}x{side_with_axes.shape[0]})")
        print(f"  saved {marker_path}  ({agent_with_wrist.shape[1]}x{agent_with_wrist.shape[0]})")
        rendered += 1

    if not rendered:
        raise SystemExit("no suites rendered — check --suites / --task")


if __name__ == "__main__":
    main()
