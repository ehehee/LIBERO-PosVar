"""Render each row of a baked .pruned_init / .init file so you can eyeball
whether the saved states actually contain upright objects. Pairs with
`permute_distractor_init.py`: if the baker accepted a row with low tilt
but the rendered image shows a tipped bottle, the tilt measurement is
wrong; if accepted rows render upright, it is right.

Usage (from LIBERO-PRO repo root):
    MUJOCO_GL=egl python scripts/verify_baked_init.py \
        --benchmark libero_object_all_variance --task-idx 0 \
        --n-rows 10 --output-dir /tmp/init_verify

Renders agentview at 256x256 for the first --n-rows rows of
<task>.pruned_init. Filenames include the file row index so you can
cross-reference with the baker's per-task summary.
"""

import argparse
import os

import init_path  # noqa: F401  -- prepends repo root to sys.path

import numpy as np
import torch
from PIL import Image

try:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
except ModuleNotFoundError:
    from libero import benchmark, get_libero_path
    from libero.envs import OffScreenRenderEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="libero_object_all_variance")
    p.add_argument("--task-idx", type=int, default=0)
    p.add_argument("--n-rows", type=int, default=10)
    p.add_argument("--output-dir", default="/tmp/init_verify")
    p.add_argument(
        "--init-file",
        default=None,
        help="Override path to the .pruned_init / .init file.",
    )
    p.add_argument(
        "--cam",
        default="agentview",
        help="Camera to render (e.g. agentview, frontview, sideview).",
    )
    p.add_argument("--res", type=int, default=256)
    args = p.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")

    bench = benchmark.get_benchmark_dict()[args.benchmark]()
    task = bench.tasks[args.task_idx]
    bddl_path = bench.get_task_bddl_file_path(args.task_idx)

    if args.init_file:
        init_path_in = args.init_file
    else:
        init_path_in = os.path.join(
            get_libero_path("init_states"),
            task.problem_folder,
            task.init_states_file,
        )

    rows = torch.load(init_path_in, weights_only=False)
    rows = np.asarray(rows)
    print(f"Loaded {init_path_in}: shape={rows.shape}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Build env once — set_init_state writes qpos directly, so the same
    # (un-perturbed) BDDL works for any baked row regardless of which
    # permutation produced it.
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=args.res,
        camera_widths=args.res,
        camera_names=[args.cam],
    )

    n = min(args.n_rows, rows.shape[0])
    for i in range(n):
        env.reset()
        obs = env.set_init_state(rows[i])

        # Pull the camera image straight from the obs dict that
        # set_init_state regenerated. LIBERO names it "<cam>_image".
        key = f"{args.cam}_image"
        if key not in obs:
            raise RuntimeError(
                f"Camera '{args.cam}' not in obs (have: {list(obs.keys())})"
            )
        rgb = np.asarray(obs[key])
        # LIBERO returns images flipped vertically vs. PIL convention.
        rgb = rgb[::-1]

        out = os.path.join(
            args.output_dir, f"{task.name}_row{i:02d}.png"
        )
        Image.fromarray(rgb).save(out)
        print(f"  row {i:02d}  ->  {out}")

    env.close()


if __name__ == "__main__":
    main()
