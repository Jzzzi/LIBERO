#!/usr/bin/env python3
# coding=utf-8
import argparse
import json
import sys
import time
from pathlib import Path

from typing import List, Tuple

from libero.lifelong import evaluate as libero_evaluate

def _resolve_output_path(repo_root: Path, output_file: str) -> Path:
    output_path = Path(output_file)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    return output_path


def _ensure_libero_path(repo_root: Path) -> None:
    libero_root = repo_root / "third_party" / "libero"
    if str(libero_root) not in sys.path:
        sys.path.insert(0, str(libero_root))


def _build_eval_argv(args: argparse.Namespace, task_id: int) -> List[str]:
    argv = [
        "--benchmark",
        args.benchmark,
        "--task_id",
        str(task_id),
        "--agent-server-host",
        args.agent_server_host,
        "--agent-server-port",
        str(args.agent_server_port),
        "--agent-timeout",
        str(args.agent_timeout),
        "--agent-env-num",
        str(args.agent_env_num),
        "--camera-size",
        str(args.camera_size),
        "--experiment_dir",
        args.experiment_dir,
    ]
    if args.agent_action_horizon is not None:
        argv.extend(["--agent-action-horizon", str(args.agent_action_horizon)])
    if args.flip_agent_images:
        argv.append("--flip-agent-images")
    else:
        argv.append("--no-flip-agent-images")
    if args.save_videos:
        argv.append("--save-videos")
    return argv


def _run_single_eval(repo_root: Path, args: argparse.Namespace, task_id: int) -> Tuple[dict, str]:
    _ensure_libero_path(repo_root)

    argv = _build_eval_argv(args, task_id)
    eval_args = libero_evaluate.parse_args(argv)
    return libero_evaluate.run_evaluation(eval_args)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch evaluate all tasks in a LIBERO benchmark")
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=["libero_10", "libero_spatial", "libero_object", "libero_goal"],
    )
    parser.add_argument("--agent-server-host", required=True)
    parser.add_argument("--agent-server-port", type=int, default=6006)
    parser.add_argument("--agent-timeout", type=float, default=10.0)
    parser.add_argument("--agent-env-num", type=int, default=5)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--agent-action-horizon", type=int, default=1)
    flip_group = parser.add_mutually_exclusive_group()
    flip_group.add_argument(
        "--flip-agent-images",
        dest="flip_agent_images",
        action="store_true",
        help="Flip images before sending to agent (default: enabled).",
    )
    flip_group.add_argument(
        "--no-flip-agent-images",
        dest="flip_agent_images",
        action="store_false",
        help="Disable flipping images before sending to agent.",
    )
    parser.add_argument("--save-videos", action="store_true", default=False)
    parser.add_argument("--experiment_dir", type=str, default="experiments")
    parser.add_argument("--output-file", type=str, default=None)
    parser.set_defaults(flip_agent_images=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[4]

    if args.output_file is None:
        args.output_file = str(
            Path(f"{args.experiment_dir}_saved") / f"{args.benchmark}_benchmark_eval.json"
        )

    # task_ids = list(range(10))
    task_ids = list(range(0, 10))  # IGNORE
    task_success = {}

    for task_id in task_ids:
        print(f"[benchmark_eval] Running task_id={task_id} with env_num={args.agent_env_num}")
        try:
            eval_stats, _save_path = _run_single_eval(repo_root, args, task_id)
            success_rate = float(eval_stats.get("success_rate", 0.0))
            task_success[str(task_id)] = success_rate
            print(f"[benchmark_eval] task_id={task_id} success_rate={success_rate:.3f}")

        except KeyboardInterrupt:
            # 专门捕获 Ctrl+C
            print(f"\n[benchmark_eval] Detected Ctrl+C! Skipping task_id={task_id} and continuing to next...")
            task_success[str(task_id)] = 0.0
            # 这里不需要写 continue，只要不 raise，代码就会自然执行完 try-except 块，
            # 然后进入下一次 for 循环（假设这段代码是在 for 循环里的）

        except Exception as e:
            # 捕获其他运行时错误
            print(f"[benchmark_eval] task_id={task_id} evaluation failed with error: {e}")
            task_success[str(task_id)] = 0.0

    overall_success = sum(task_success.values()) / len(task_ids) if task_ids else 0.0
    output_payload = {
        "benchmark": args.benchmark,
        "task_success_rates": task_success,
        "overall_success_rate": overall_success,
        "agent_env_num": args.agent_env_num,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    output_path = _resolve_output_path(repo_root, args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_payload, f, indent=2)
        f.write("\n")

    print(f"[benchmark_eval] Wrote results to {output_path}")


if __name__ == "__main__":
    main()
