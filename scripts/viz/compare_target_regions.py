"""Side-by-side topdown comparing alphabet_soup spawn between baseline and var20x20.

For each suite the script:
  1. resets the alphabet_soup BDDL,
  2. applies the first ``.pruned_init`` state so the soup sits at its actual
     scripted spawn (not the BDDL region center),
  3. hides the robot/gripper geoms,
  4. renders the re-aimed birdview camera.

Each panel then overlays the BDDL's ``target_object_region`` rectangle and a
crosshair at the soup's measured ``body_xpos`` so the "actual change" between
the two suites is visible against the same workspace.

Run:
    MUJOCO_GL=egl python scripts/viz/compare_target_regions.py
"""

import argparse
import math
import os
import re
import sys

_THIS_DIR = os.path.dirname(os.path.realpath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
_LIBERO_PKG_ROOT = os.path.join(_REPO_ROOT, "libero")
for _p in (_REPO_ROOT, _LIBERO_PKG_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import imageio.v2 as imageio  # noqa: F401 - kept for parity with sibling scripts
from PIL import Image, ImageDraw, ImageFont


def _install_numpy_unpickle_compat():
    if hasattr(np, "_core"):
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

TOPDOWN_CAM = "birdview"
TOPDOWN_POS = (-0.05, 0.0, 3.0)
TOPDOWN_FOVY = 22.0
FLOOR_Z = -0.035  # libero_floor_manipulation.workspace_offset[2]
TARGET_BODY = "alphabet_soup_1_main"
TASK_FILE = "pick_up_the_alphabet_soup_and_place_it_in_the_basket"


def parse_target_region(bddl_path):
    """Return (xmin, ymin, xmax, ymax) for the ``target_object_region`` block."""
    with open(bddl_path) as f:
        text = f.read()
    m = re.search(
        r"\(target_object_region\b.*?\(:ranges\s*\(\s*\(([^)]+)\)",
        text, re.DOTALL,
    )
    if not m:
        return None
    return tuple(float(v) for v in m.group(1).split())


def _aim_topdown_camera(env):
    model = env.sim.model
    cid = model.camera_name2id(TOPDOWN_CAM)
    model.cam_pos[cid] = np.asarray(TOPDOWN_POS, dtype=model.cam_pos.dtype)
    model.cam_fovy[cid] = float(TOPDOWN_FOVY)


def _hide_robot(env):
    model = env.sim.model
    hide_body_ids = set()
    for bid in range(model.nbody):
        name = model.body_id2name(bid)
        if name is None:
            continue
        lname = name.lower()
        if any(t in lname for t in ("robot0", "panda", "gripper", "mount")):
            hide_body_ids.add(bid)
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] in hide_body_ids:
            model.geom_rgba[gid, 3] = 0.0


def render_scene(bddl_path, init_path, width, height, settle):
    """Render birdview using init[0]; also collect target xy across all states."""
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_names=[TOPDOWN_CAM],
        camera_heights=height,
        camera_widths=width,
    )
    try:
        obs = env.reset()
        _aim_topdown_camera(env)
        _hide_robot(env)
        all_xy = []
        if init_path is not None and os.path.exists(init_path):
            states = torch.load(init_path, weights_only=False)
            bid = env.sim.model.body_name2id(TARGET_BODY)
            for s in states:
                env.set_init_state(s)
                all_xy.append(env.sim.data.body_xpos[bid][:2].copy())
            obs = env.set_init_state(states[0])
        else:
            bid = env.sim.model.body_name2id(TARGET_BODY)
        noop = np.zeros(env.env.action_dim, dtype=np.float64)
        for _ in range(settle):
            obs, _, _, _ = env.step(noop)
        target_xy = env.sim.data.body_xpos[bid][:2].copy()
        if not all_xy:
            all_xy = [target_xy]
        img = np.flipud(obs[f"{TOPDOWN_CAM}_image"]).astype(np.uint8)
    finally:
        env.close()
    return img, np.asarray(target_xy), np.asarray(all_xy)


def world_to_pixel(xw, yw, W, H,
                   cam_pos=TOPDOWN_POS, fovy_deg=TOPDOWN_FOVY,
                   floor_z=FLOOR_Z):
    """Project world (x, y) on the floor to displayed-image pixel (u, v).

    Birdview scene-XML quat is (0.7071, 0, 0, 0.7071) — a 90° CCW rotation
    about world +z — so the camera's local +x ≡ world +y and local +y ≡
    world -x. The script's ``np.flipud`` then flips the bottom-up raster.
    """
    fovy_rad = math.radians(fovy_deg)
    depth = cam_pos[2] - floor_z
    f = (H / 2.0) / math.tan(fovy_rad / 2.0)
    u_img = f * (yw - cam_pos[1]) / depth
    v_img = f * (cam_pos[0] - xw) / depth
    raw_px = W / 2.0 + u_img
    raw_py = H / 2.0 + v_img
    return raw_px, H - 1 - raw_py


def draw_overlay(img, region, all_xy, W, H, color, line_w):
    """Overlay the BDDL bbox plus a dot per (collected) actual spawn xy."""
    rgba = Image.fromarray(img).convert("RGBA")
    layer = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    xmin, ymin, xmax, ymax = region
    corners = [
        world_to_pixel(xmin, ymin, W, H),
        world_to_pixel(xmin, ymax, W, H),
        world_to_pixel(xmax, ymax, W, H),
        world_to_pixel(xmax, ymin, W, H),
    ]
    fill = color + (50,)
    outline = color + (255,)
    draw.polygon(corners, fill=fill)
    cx_pix = corners + [corners[0]]
    for i in range(4):
        draw.line([cx_pix[i], cx_pix[i + 1]], fill=outline, width=line_w)

    # Also draw the empirical bbox of the actual scatter (dashed-feel = thinner).
    xs = all_xy[:, 0]; ys = all_xy[:, 1]
    emp = (xs.min(), ys.min(), xs.max(), ys.max())
    emp_corners = [
        world_to_pixel(emp[0], emp[1], W, H),
        world_to_pixel(emp[0], emp[3], W, H),
        world_to_pixel(emp[2], emp[3], W, H),
        world_to_pixel(emp[2], emp[1], W, H),
    ]
    emp_pix = emp_corners + [emp_corners[0]]
    thin = max(1, line_w // 2)
    for i in range(4):
        draw.line([emp_pix[i], emp_pix[i + 1]], fill=outline, width=thin)

    # One filled dot per actual init spawn.
    r = max(4, W // 240)
    dot_fill = color + (220,)
    for x, y in all_xy:
        px, py = world_to_pixel(x, y, W, H)
        draw.ellipse([px - r, py - r, px + r, py + r],
                     fill=dot_fill, outline=outline, width=1)
    return Image.alpha_composite(rgba, layer).convert("RGB")


def label_panel(img, title, lines, font_size):
    pad = font_size // 2
    header_h = font_size + 2 * pad
    foot_h = font_size * len(lines) + 2 * pad if lines else 0
    out = Image.new("RGB", (img.width, img.height + header_h + foot_h), "white")
    out.paste(img, (0, header_h))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        font_mono = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            int(font_size * 0.78))
    except OSError:
        font = ImageFont.load_default()
        font_mono = font
    tw = draw.textlength(title, font=font)
    draw.text(((img.width - tw) / 2, pad), title, fill="black", font=font)
    for i, ln in enumerate(lines):
        draw.text((pad, img.height + header_h + pad + i * font_size),
                  ln, fill="black", font=font_mono)
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suites", nargs="+",
                        default=["libero_object",
                                 "libero_object_target_pos_var20x20"],
                        help="suite names to render side-by-side (order = panel order)")
    parser.add_argument("--render-width", type=int, default=1600)
    parser.add_argument("--render-height", type=int, default=1024)
    parser.add_argument("--settle", type=int, default=40)
    parser.add_argument("--out",
                        default=os.path.join(_THIS_DIR,
                                             "target_region_compare.png"))
    args = parser.parse_args()

    palette = [
        (40, 110, 220),   # blue
        (40, 160, 60),    # green
        (220, 50, 50),    # red
        (200, 130, 0),    # amber
    ]
    suite_entries = [(s, palette[i % len(palette)])
                     for i, s in enumerate(args.suites)]

    panels = []
    for suite, color in suite_entries:
        bddl_path = os.path.join(BDDL_ROOT, suite, TASK_FILE + ".bddl")
        init_path = None
        for ext in (".pruned_init", ".init"):
            cand = os.path.join(INIT_ROOT, suite, TASK_FILE + ext)
            if os.path.exists(cand):
                init_path = cand
                break
        if not os.path.exists(bddl_path):
            raise SystemExit(f"BDDL not found: {bddl_path}")
        region = parse_target_region(bddl_path)
        if region is None:
            raise SystemExit(f"target_object_region missing in {bddl_path}")
        print(f"[render] {suite}/{TASK_FILE}  "
              f"region=({region[0]:+.4f},{region[1]:+.4f})→"
              f"({region[2]:+.4f},{region[3]:+.4f})  "
              f"init={'pruned' if init_path and init_path.endswith('.pruned_init') else 'reset'}")
        img, actual, all_xy = render_scene(bddl_path, init_path,
                                           args.render_width, args.render_height,
                                           args.settle)
        emp = (all_xy[:, 0].min(), all_xy[:, 1].min(),
               all_xy[:, 0].max(), all_xy[:, 1].max())
        n_uniq = len(np.unique(np.round(all_xy, 4), axis=0))
        print(f"           n_states={len(all_xy)}  unique_xy={n_uniq}  "
              f"empirical bbox x=[{emp[0]:+.4f},{emp[2]:+.4f}]  "
              f"y=[{emp[1]:+.4f},{emp[3]:+.4f}]")

        line_w = max(2, args.render_width // 400)
        overlaid = draw_overlay(img, region, all_xy,
                                args.render_width, args.render_height,
                                color, line_w)

        font_size = max(28, args.render_height // 28)
        lines = [
            f"BDDL target_object_region: "
            f"x in [{region[0]:+.3f}, {region[2]:+.3f}]  "
            f"y in [{region[1]:+.3f}, {region[3]:+.3f}]",
            f"actual spawn empirical bbox  ({len(all_xy)} states, "
            f"{n_uniq} unique): "
            f"x in [{emp[0]:+.3f}, {emp[2]:+.3f}]  "
            f"y in [{emp[1]:+.3f}, {emp[3]:+.3f}]",
        ]
        title = suite
        panels.append(label_panel(overlaid, title, lines, font_size))

    gap = 24
    total_w = sum(p.width for p in panels) + gap * (len(panels) - 1)
    max_h = max(p.height for p in panels)
    combined = Image.new("RGB", (total_w, max_h), "white")
    x = 0
    for p in panels:
        combined.paste(p, (x, 0))
        x += p.width + gap
    combined.save(args.out)
    print(f"[save] {args.out}  ({combined.size[0]}x{combined.size[1]})")


if __name__ == "__main__":
    main()
