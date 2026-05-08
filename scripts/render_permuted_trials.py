"""Render trials of a LIBERO task as a settling video.

For each trial in --trials:
  1. env.reset()
  2. env.set_init_state(init_states[trial])
  3. step the env with a zero action for --settle frames, recording each frame
All trials are concatenated end-to-end into a single MP4.

    MUJOCO_GL=egl python scripts/render_permuted_trials.py \
        --benchmark libero_object_all_variance \
        --task 0 --trials 0 1 2 3 --settle 40 --out /tmp/permuted_trials.mp4
"""

import os
import argparse

import init_path  # noqa: F401

import numpy as np
import torch
import imageio.v2 as imageio

try:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
except ModuleNotFoundError:
    from libero import benchmark, get_libero_path
    from libero.envs import OffScreenRenderEnv


def render_trials(
    benchmark_name, task_idx, trials, settle, out_path, cam_h=256, cam_w=256, fps=30
):
    bench = benchmark.get_benchmark_dict()[benchmark_name]()
    task = bench.tasks[task_idx]
    bddl_path = bench.get_task_bddl_file_path(task_idx)
    init_path = os.path.join(
        get_libero_path("init_states"), task.problem_folder, task.init_states_file
    )
    rows = torch.load(init_path, weights_only=False)

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=cam_h,
        camera_widths=cam_w,
    )
    env.reset()
    action_dim = env.env.action_dim
    noop = np.zeros(action_dim, dtype=np.float64)

    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)
    try:
        for t in trials:
            env.reset()
            env.set_init_state(rows[t])
            for _ in range(settle):
                obs, _, _, _ = env.step(noop)
            img = np.flipud(obs["agentview_image"])
            writer.append_data(img.astype(np.uint8))
            print(f"  trial {t}: frame {settle} written")
    finally:
        writer.close()
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
    )


if __name__ == "__main__":
    main()
