"""Render trials of a LIBERO task as a settling video.

For each trial in --trials:
  1. env.reset()
  2. env.set_init_state(init_states[trial])
  3. step the env with a zero action for --settle frames, recording each frame
All trials are concatenated end-to-end into a single MP4.

    MUJOCO_GL=egl python scripts/render_permuted_trials.py \
        --benchmark libero_object_all_variance \
        --task 0 --trials 0 1 2 3 --settle 40 --out /tmp/permuted_trials.mp4

First-frame rendering of the stove+pot variance bake (libero_90 task 45,
init states under libero_90_kitchen_scene9_stove_pot_xy_variance/). The
stove is a MuJoCo fixture (no free joint), so its perturbed pose lives
in the model XML rather than qpos — without --variance the stove looks
identical in every frame. --variance rebuilds the env per trial with
the same perturbed BDDL the bake used (same base-seed schedule):

    MUJOCO_GL=egl python scripts/render_permuted_trials.py \
        --benchmark libero_90 --task 45 \
        --variance stove_pot_xy --base-seed 2250 \
        --init-folder libero_90_kitchen_scene9_stove_pot_xy_variance \
        --trials $(seq 0 49) --settle 0 --fps 4 \
        --out /tmp/stove_pot_first_frames.mp4
"""

import os
import sys
import argparse

# Force `import libero` to resolve to THIS checkout (LIBERO-PosVar), not any
# editable LIBERO-PRO install that may also be on sys.path. Matches the
# behaviour of scripts/viz/visualize_first_frames.py.
_THIS_DIR = os.path.dirname(os.path.realpath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
_LIBERO_PKG_ROOT = os.path.join(_REPO_ROOT, "libero")
for _p in (_REPO_ROOT, _LIBERO_PKG_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("MUJOCO_GL", "egl")


def _redirect_libero_config_to_this_checkout():
    """Point ``libero.get_libero_path`` at this checkout's BDDL / init files.

    The system-wide ``~/.libero/config.yaml`` ships pointing at LIBERO-PRO,
    which lacks the variance suites baked here. Write a per-process config
    into a tempdir and have libero read that instead.
    """
    import tempfile, yaml  # local import to keep top imports tidy

    tmpdir = tempfile.mkdtemp(prefix="libero_posvar_cfg_")
    cfg_path = os.path.join(tmpdir, "config.yaml")
    benchmark_root = os.path.join(_LIBERO_PKG_ROOT, "libero")
    cfg = {
        "benchmark_root": benchmark_root,
        "bddl_files": os.path.join(benchmark_root, "bddl_files"),
        "init_states": os.path.join(benchmark_root, "init_files"),
        "assets": os.path.join(benchmark_root, "assets"),
        "datasets": os.path.join(benchmark_root, os.pardir, "datasets"),
    }
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)
    os.environ["LIBERO_CONFIG_PATH"] = tmpdir


_redirect_libero_config_to_this_checkout()

import init_path  # noqa: F401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import imageio.v2 as imageio  # noqa: E402


def _install_numpy_unpickle_compat():
    """Let an old NumPy (<2.0) unpickle arrays saved by NumPy >=2.0."""
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
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
except ModuleNotFoundError:
    from libero import benchmark, get_libero_path
    from libero.envs import OffScreenRenderEnv


TOPDOWN_POS = (-0.05, 0.0, 3.0)
TOPDOWN_FOVY = 22.0
FLOOR_Z = -0.035  # libero_floor_manipulation.workspace_offset[2]


def _world_to_pixel(xw, yw, W, H, cam_pos, fovy_deg, floor_z=FLOOR_Z):
    """Birdview projection (camera scene-XML quat is (0.7071,0,0,0.7071))."""
    import math
    fovy_rad = math.radians(fovy_deg)
    depth = cam_pos[2] - floor_z
    f = (H / 2.0) / math.tan(fovy_rad / 2.0)
    u_img = f * (yw - cam_pos[1]) / depth
    v_img = f * (cam_pos[0] - xw) / depth
    raw_px = W / 2.0 + u_img
    raw_py = H / 2.0 + v_img
    return raw_px, H - 1 - raw_py


def _parse_target_body(bddl_path):
    """First name in ``:obj_of_interest`` block + ``_main`` (the soup/can body)."""
    import re
    with open(bddl_path) as f:
        text = f.read()
    m = re.search(r"\(:obj_of_interest\b\s*([^()]+)\)", text)
    if not m:
        return None
    first = m.group(1).split()[0]
    return f"{first}_main"


def _aim_topdown_camera(env, cam_name, pos, fovy):
    model = env.sim.model
    cid = model.camera_name2id(cam_name)
    model.cam_pos[cid] = np.asarray(pos, dtype=model.cam_pos.dtype)
    model.cam_fovy[cid] = float(fovy)


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


def _resolve_base_bddl_path(bench, task_idx, source_suite):
    """Find the unperturbed source BDDL.

    Variance suites typically ship pre-perturbed BDDLs under their own
    folder; re-perturbing those would compound the changes. ``source_suite``
    optionally points us at a sibling folder (e.g. ``libero_90``) that
    holds the canonical un-perturbed BDDL.
    """
    bddl_path = bench.get_task_bddl_file_path(task_idx)
    if source_suite:
        candidate = os.path.join(
            get_libero_path("bddl_files"),
            source_suite,
            os.path.basename(bddl_path),
        )
        if os.path.exists(candidate):
            return candidate
    return bddl_path


def _build_env(bddl_file, cam_list, cam_h, cam_w):
    return OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_names=cam_list,
        camera_heights=cam_h,
        camera_widths=cam_w,
    )


def render_trials(
    benchmark_name, task_idx, trials, settle, out_path,
    cam_h=256, cam_w=256, fps=30, camera="agentview",
    aim_topdown=False, hide_arm=False,
    topdown_pos=TOPDOWN_POS, topdown_fovy=TOPDOWN_FOVY,
    draw_target_bbox=False, target_body=None,
    variance=None, base_seed=0,
    init_folder=None, source_suite=None,
):
    bench = benchmark.get_benchmark_dict()[benchmark_name]()
    task = bench.tasks[task_idx]
    bddl_path = _resolve_base_bddl_path(bench, task_idx, source_suite)

    init_folder_name = init_folder if init_folder else task.problem_folder
    init_path = os.path.join(
        get_libero_path("init_states"), init_folder_name, task.init_states_file
    )
    rows = torch.load(init_path, weights_only=False)

    # Ensure the requested camera is actually rendered (the default
    # OffScreenRenderEnv camera list only includes agentview + wrist).
    default_cams = ["agentview", "robot0_eye_in_hand"]
    cam_list = default_cams if camera in default_cams else default_cams + [camera]

    # When ``variance`` is given, we rebuild the env per trial with the
    # same perturbed BDDL the bake used — needed for any fixtures whose
    # pose is baked into the model XML (e.g. flat_stove_1 in libero_90
    # task 45) rather than carried in qpos. Otherwise we build a single
    # env up front and reuse it across trials.
    perturb_fn = None
    bake_routing = None
    if variance is not None:
        # Import lazily so envs without this script's siblings still work.
        import importlib
        sys.path.insert(0, _THIS_DIR)
        bake_routing = importlib.import_module("bake_first_frame_init")
        if variance in bake_routing.VARIANCE_OVERRIDES:
            variance_suite = bake_routing.VARIANCE_OVERRIDES[variance]
        else:
            variance_suite = variance  # caller passed a full suite name
        perturb_fn = lambda content, rng: bake_routing.perturb_bddl_for_suite(
            variance_suite, content, rng
        )

    if perturb_fn is None:
        env = _build_env(bddl_path, cam_list, cam_h, cam_w)
        env.reset()
        if aim_topdown:
            _aim_topdown_camera(env, camera, topdown_pos, topdown_fovy)
        if hide_arm:
            _hide_robot(env)
    else:
        env = None  # built per trial below
        with open(bddl_path) as f:
            base_bddl_content = f.read()

    # action_dim only matters when --settle > 0; sample once with whatever
    # env we have, or defer to per-trial setup when env is None.
    if env is not None:
        action_dim = env.env.action_dim
        noop = np.zeros(action_dim, dtype=np.float64)
    else:
        noop = None  # populated lazily per-trial when needed

    import random
    import tempfile

    def _perturb_to_tempfile(trial_t):
        rng = random.Random(base_seed + trial_t)
        new_content = perturb_fn(base_bddl_content, rng)
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".bddl", delete=False, dir="/tmp"
        )
        tmp.write(new_content)
        tmp.close()
        return tmp.name

    # Pre-compute the empirical bbox of the target body across the selected
    # init states, so every frame can draw the same rectangle.
    bbox_pixels = None
    if draw_target_bbox:
        if target_body is None:
            target_body = _parse_target_body(bddl_path)
        if target_body is None:
            print("[warn] could not infer target body; skipping bbox overlay")
        else:
            xs, ys = [], []
            bid = None
            for t in trials:
                if perturb_fn is not None:
                    tmp_bddl = _perturb_to_tempfile(t)
                    bbox_env = _build_env(tmp_bddl, cam_list, cam_h, cam_w)
                else:
                    bbox_env = env
                    tmp_bddl = None
                try:
                    if perturb_fn is None:
                        bbox_env.reset()
                    if aim_topdown:
                        _aim_topdown_camera(bbox_env, camera, topdown_pos, topdown_fovy)
                    if hide_arm:
                        _hide_robot(bbox_env)
                    try:
                        bid = bbox_env.sim.model.body_name2id(target_body)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[warn] body '{target_body}' not in model ({exc}); "
                              "skipping bbox overlay")
                        bid = None
                        break
                    bbox_env.set_init_state(rows[t])
                    p = bbox_env.sim.data.body_xpos[bid]
                    xs.append(float(p[0]))
                    ys.append(float(p[1]))
                finally:
                    if perturb_fn is not None:
                        bbox_env.close()
                        os.unlink(tmp_bddl)
            if bid is not None and xs:
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                print(f"[bbox] {target_body} over {len(xs)} init states: "
                      f"x in [{xmin:+.4f}, {xmax:+.4f}]  "
                      f"y in [{ymin:+.4f}, {ymax:+.4f}]")
                # Project the 4 corners to displayed-image pixel space.
                bbox_pixels = [
                    _world_to_pixel(xmin, ymin, cam_w, cam_h, topdown_pos, topdown_fovy),
                    _world_to_pixel(xmin, ymax, cam_w, cam_h, topdown_pos, topdown_fovy),
                    _world_to_pixel(xmax, ymax, cam_w, cam_h, topdown_pos, topdown_fovy),
                    _world_to_pixel(xmax, ymin, cam_w, cam_h, topdown_pos, topdown_fovy),
                ]

    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)
    try:
        for t in trials:
            if perturb_fn is not None:
                tmp_bddl = _perturb_to_tempfile(t)
                env = _build_env(tmp_bddl, cam_list, cam_h, cam_w)
                env.reset()
                if aim_topdown:
                    _aim_topdown_camera(env, camera, topdown_pos, topdown_fovy)
                if hide_arm:
                    _hide_robot(env)
                action_dim = env.env.action_dim
                noop = np.zeros(action_dim, dtype=np.float64)
            else:
                env.reset()
                # ``env.reset()`` rebuilds the mjModel from XML, so re-apply
                # camera / visibility tweaks each trial.
                if aim_topdown:
                    _aim_topdown_camera(env, camera, topdown_pos, topdown_fovy)
                if hide_arm:
                    _hide_robot(env)
                tmp_bddl = None

            obs = env.set_init_state(rows[t])
            for _ in range(settle):
                obs, _, _, _ = env.step(noop)
            img = np.flipud(obs[f"{camera}_image"]).astype(np.uint8)
            if bbox_pixels is not None:
                from PIL import Image, ImageDraw  # local: keep top imports tidy
                pil = Image.fromarray(img).convert("RGBA")
                layer = Image.new("RGBA", pil.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(layer)
                outline = (255, 235, 0, 255)  # bright yellow for visibility
                fill = (255, 235, 0, 40)
                draw.polygon(bbox_pixels, fill=fill)
                line_w = max(2, cam_w // 320)
                pts = bbox_pixels + [bbox_pixels[0]]
                for i in range(4):
                    draw.line([pts[i], pts[i + 1]], fill=outline, width=line_w)
                img = np.asarray(Image.alpha_composite(pil, layer).convert("RGB"))
            writer.append_data(img)
            print(f"  trial {t}: frame {settle} written")

            if perturb_fn is not None:
                env.close()
                env = None
                if tmp_bddl is not None:
                    try:
                        os.unlink(tmp_bddl)
                    except OSError:
                        pass
    finally:
        writer.close()
        if env is not None:
            env.close()

    print(f"saved {out_path}  trials={list(trials)}  task={task.name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="libero_object_all_variance")
    p.add_argument("--task", type=int, default=0, help="task index in benchmark")
    p.add_argument("--trials", type=int, nargs="+", default=list(range(50)))
    p.add_argument("--settle", type=int, default=40, help="frames to settle per trial")
    p.add_argument("--fps", type=int, default=2)
    p.add_argument("--cam-h", type=int, default=256)
    p.add_argument("--cam-w", type=int, default=256)
    p.add_argument("--out", default="/tmp/permuted_trials.mp4")
    p.add_argument("--camera", default="agentview",
                   help="camera key (e.g. agentview, robot0_eye_in_hand, birdview)")
    p.add_argument("--aim-topdown", action="store_true",
                   help="re-aim the camera to the showcase topdown framing "
                        f"(pos {TOPDOWN_POS}, fovy {TOPDOWN_FOVY})")
    p.add_argument("--hide-arm", action="store_true",
                   help="zero alpha of robot/gripper geoms before rendering")
    p.add_argument("--topdown-pos", type=float, nargs=3,
                   default=list(TOPDOWN_POS), metavar=("X", "Y", "Z"))
    p.add_argument("--topdown-fovy", type=float, default=TOPDOWN_FOVY)
    p.add_argument("--draw-target-bbox", action="store_true",
                   help="on every frame, overlay the empirical bbox of the "
                        "target object's spawn positions across --trials")
    p.add_argument("--target-body", default=None,
                   help="MuJoCo body name to track (default: parsed from BDDL "
                        "':obj_of_interest' + '_main')")
    p.add_argument("--variance", default=None,
                   help="If set, rebuild the env per trial with a BDDL "
                        "perturbed by the named generator (uses bake_first_"
                        "frame_init.VARIANCE_OVERRIDES; e.g. 'stove_pot_xy'). "
                        "Needed when the bake also perturbed scene fixtures "
                        "whose pose isn't in qpos.")
    p.add_argument("--base-seed", type=int, default=0,
                   help="Per-trial perturbation seed = base_seed + trial. "
                        "Match the value the bake used: "
                        "args.seed + task_idx * n_trials. "
                        "Default 0 (same as the bake's --seed default).")
    p.add_argument("--init-folder", default=None,
                   help="Override the init-states folder under init_files/. "
                        "Defaults to the task's own problem_folder. Use this "
                        "when the bake wrote to a sibling variance folder "
                        "(e.g. libero_90_kitchen_scene9_stove_pot_xy_variance).")
    p.add_argument("--source-suite", default=None,
                   help="Override the BDDL source folder under bddl_files/. "
                        "Useful when --variance is applied on top of a stock "
                        "benchmark and you want to make the base-BDDL lookup "
                        "explicit.")
    args = p.parse_args()

    render_trials(
        args.benchmark,
        args.task,
        args.trials,
        args.settle,
        args.out,
        cam_h=args.cam_h,
        cam_w=args.cam_w,
        fps=args.fps,
        camera=args.camera,
        aim_topdown=args.aim_topdown,
        hide_arm=args.hide_arm,
        topdown_pos=tuple(args.topdown_pos),
        topdown_fovy=args.topdown_fovy,
        draw_target_bbox=args.draw_target_bbox,
        target_body=args.target_body,
        variance=args.variance,
        base_seed=args.base_seed,
        init_folder=args.init_folder,
        source_suite=args.source_suite,
    )


if __name__ == "__main__":
    main()
