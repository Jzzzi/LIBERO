import argparse
import sys
import os
import pickle
import socket
import struct
from typing import Any, Dict, List, Optional, Tuple

# TODO: find a better way for this?
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import hydra
import json
import numpy as np
import pprint
import time
import torch
import wandb
import yaml
from PIL import Image
from easydict import EasyDict
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoModel, pipeline, AutoTokenizer, logging
from pathlib import Path

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from libero.libero.utils.time_utils import Timer
from libero.libero.utils.video_utils import VideoWriter
from libero.lifelong.algos import *
from libero.lifelong.datasets import get_dataset, SequenceVLDataset, GroupedTaskDataset
from libero.lifelong.metric import (
    evaluate_loss,
    evaluate_success,
    raw_obs_to_tensor_obs,
)
from libero.lifelong.utils import (
    control_seed,
    safe_device,
    torch_load_model,
    NpEncoder,
    compute_flops,
)

from libero.lifelong.main import get_task_embs

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils

import time


benchmark_map = {
    "libero_10": "LIBERO_10",
    "libero_spatial": "LIBERO_SPATIAL",
    "libero_object": "LIBERO_OBJECT",
    "libero_goal": "LIBERO_GOAL",
}

algo_map = {
    "base": "Sequential",
    "er": "ER",
    "ewc": "EWC",
    "packnet": "PackNet",
    "multitask": "Multitask",
}

policy_map = {
    "bc_rnn_policy": "BCRNNPolicy",
    "bc_transformer_policy": "BCTransformerPolicy",
    "bc_vilt_policy": "BCViLTPolicy",
}

try:
    from tqdm import tqdm
except ImportError:
    # Lightweight fallback to avoid hard dependency
    class _TqdmStub:
        def __init__(self, *args, **kwargs):
            pass

        def update(self, n=1):
            pass

        def close(self):
            pass

    def tqdm(*args, **kwargs):
        return _TqdmStub()


def _recvall(sock: socket.socket, length: int) -> Optional[bytes]:
    data = bytearray()
    while len(data) < length:
        try:
            packet = sock.recv(length - len(data))
        except socket.timeout:
            return None
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)


def _recv_message(sock: socket.socket) -> Optional[Any]:
    raw_len = _recvall(sock, 4)
    if not raw_len:
        return None
    msg_len = struct.unpack(">I", raw_len)[0]
    payload = _recvall(sock, msg_len)
    if payload is None:
        return None
    return pickle.loads(payload)


def _send_message(sock: socket.socket, message: Any) -> None:
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    header = struct.pack(">I", len(payload))
    sock.sendall(header + payload)


class AgentActionClient:
    def __init__(self, host: str, port: int, timeout: float = 10.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None

    def connect(self) -> None:
        if self.sock is None:
            self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
            self.sock.settimeout(self.timeout)

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def infer_actions(self, obs_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.connect()
        if not isinstance(obs_batch, list):
            obs_batch = [obs_batch]
        _send_message(self.sock, {"obs": obs_batch, "batch_size": len(obs_batch)})
        response = _recv_message(self.sock)
        if response is None:
            raise RuntimeError("Agent server returned empty response")
        return response.get("result", [])


def _resize_image_if_needed(image: np.ndarray, target_size: int) -> np.ndarray:
    if target_size is None:
        return image
    if (
        hasattr(image, "shape")
        and len(image.shape) == 3
        and image.shape[0] == target_size
        and image.shape[1] == target_size
    ):
        return image
    return np.array(Image.fromarray(image).resize((target_size, target_size)))


def _build_agent_obs(env_obs: Dict[str, Any], instruction: str, obs_key_mapping: Dict[str, str], target_size: int, flip: bool = False) -> Dict[str, Any]:
    agentview_key = obs_key_mapping.get("agentview_rgb", "agentview_image")
    wrist_key = obs_key_mapping.get("eye_in_hand_rgb", "robot0_eye_in_hand_image")

    payload: Dict[str, Any] = {
        "instruction": instruction,
        "images": {},
    }

    if agentview_key in env_obs:
        img = env_obs[agentview_key]
        if flip:
            img = np.flipud(img)
            img = np.fliplr(img)
        payload["images"]["image"] = _resize_image_if_needed(img, target_size)
    if wrist_key in env_obs:
        img = env_obs[wrist_key]
        if flip:
            img = np.flipud(img)
            img = np.fliplr(img)
        payload["images"]["wrist_image"] = _resize_image_if_needed(img, target_size)
    return payload


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("--experiment_dir", type=str, default="experiments")
    # for which task suite
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["libero_10", "libero_spatial", "libero_object", "libero_goal"],
    )
    parser.add_argument("--task_id", type=int, required=False, default=0)
    # method detail
    parser.add_argument(
        "--algo",
        type=str,
        required=False,
        choices=["base", "er", "ewc", "packnet", "multitask"],
    )
    parser.add_argument(
        "--policy",
        type=str,
        required=False,
        choices=["bc_rnn_policy", "bc_transformer_policy", "bc_vilt_policy"],
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ep", type=int)
    parser.add_argument("--load_task", type=int)
    parser.add_argument("--device_id", type=int)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument(
        "--agent-server-host",
        type=str,
        default=None,
        help="Connect to external ShowVLA agent server instead of local policy.",
    )
    parser.add_argument(
        "--agent-server-port", type=int, default=6006, help="Agent server TCP port."
    )
    parser.add_argument(
        "--agent-timeout", type=float, default=10.0, help="Socket timeout in seconds."
    )
    flip_group = parser.add_mutually_exclusive_group()
    flip_group.add_argument(
        "--flip-agent-images",
        dest="flip_agent_images",
        action="store_true",
        help="Flip images vertically before sending to agent (default: enabled).",
    )
    flip_group.add_argument(
        "--no-flip-agent-images",
        dest="flip_agent_images",
        action="store_false",
        help="Disable flipping images before sending to agent.",
    )
    parser.add_argument(
        "--agent-env-num",
        type=int,
        default=None,
        help="Override env num when using agent server (default: use eval.num_procs).",
    )
    parser.add_argument(
        "--camera-size",
        type=int,
        default=256,
        help="Camera width/height when driving ShowVLA agent.",
    )
    parser.add_argument(
        "--agent-action-horizon",
        type=int,
        default=1,
        help="Use only the first N predicted actions from the agent response (default: 1).",
    )
    parser.set_defaults(flip_agent_images=True)
    args = parser.parse_args(argv)
    args.use_agent_server = args.agent_server_host is not None

    # Validate arguments
    if not args.use_agent_server:
        missing = []
        for field in ["algo", "policy", "seed", "device_id"]:
            if getattr(args, field) is None:
                missing.append(field)
        if missing:
            parser.error(
                f"{', '.join(missing)} required unless using --agent-server-host"
            )
    if args.device_id is not None:
        args.device_id = "cuda:" + str(args.device_id)
    else:
        args.device_id = "cuda"

    args.save_dir = f"{args.experiment_dir}_saved"

    if not args.use_agent_server:
        if args.algo == "multitask":
            assert args.ep in list(
                range(0, 50, 5)
            ), "[error] ep should be in [0, 5, ..., 50]"
        else:
            assert args.load_task in list(
                range(10)
            ), "[error] load_task should be in [0, ..., 9]"
    return args


def run_evaluation(args: argparse.Namespace) -> Tuple[Dict[str, Any], str]:
    use_agent_server = args.use_agent_server
    agent_client = None

    if use_agent_server:
        with hydra.initialize_config_module(
            config_module="libero.configs", version_base=None
        ):
            cfg = hydra.compose(config_name="config")
        cfg.folder = get_libero_path("datasets")
        cfg.bddl_folder = get_libero_path("bddl_files")
        cfg.init_states_folder = get_libero_path("init_states")
        cfg.device = args.device_id
        cfg.seed = (
            args.seed
            if args.seed is not None
            else getattr(cfg, "seed", 10000)
        )
        cfg.benchmark_name = benchmark_map[args.benchmark]
        cfg.data.img_h = args.camera_size
        cfg.data.img_w = args.camera_size
        if not hasattr(cfg.data, "obs_key_mapping"):
            cfg.data.obs_key_mapping = {}
        if not hasattr(cfg.data, "task_order_index"):
            cfg.data.task_order_index = 0
        algo = None
        task_embs = None
        run_folder = f"agent_server_{args.agent_server_host}:{args.agent_server_port}"
        agent_client = AgentActionClient(
            args.agent_server_host, args.agent_server_port, timeout=args.agent_timeout
        )
    else:
        # e.g., experiments/LIBERO_SPATIAL/Multitask/BCRNNPolicy_seed100/
        experiment_dir = os.path.join(
            args.experiment_dir,
            f"{benchmark_map[args.benchmark]}/"
            + f"{algo_map[args.algo]}/"
            + f"{policy_map[args.policy]}_seed{args.seed}",
        )

        # find the checkpoint
        experiment_id = 0
        for path in Path(experiment_dir).glob("run_*"):
            if not path.is_dir():
                continue
            try:
                folder_id = int(str(path).split("run_")[-1])
                if folder_id > experiment_id:
                    experiment_id = folder_id
            except BaseException:
                pass
        if experiment_id == 0:
            print(f"[error] cannot find the checkpoint under {experiment_dir}")
            sys.exit(0)

        run_folder = os.path.join(experiment_dir, f"run_{experiment_id:03d}")
        try:
            if args.algo == "multitask":
                model_path = os.path.join(run_folder, f"multitask_model_ep{args.ep}.pth")
                sd, cfg, previous_mask = torch_load_model(
                    model_path, map_location=args.device_id
                )
            else:
                model_path = os.path.join(run_folder, f"task{args.load_task}_model.pth")
                sd, cfg, previous_mask = torch_load_model(
                    model_path, map_location=args.device_id
                )
        except:
            print(f"[error] cannot find the checkpoint at {str(model_path)}")
            sys.exit(0)

        cfg.folder = get_libero_path("datasets")
        cfg.bddl_folder = get_libero_path("bddl_files")
        cfg.init_states_folder = get_libero_path("init_states")

        cfg.device = args.device_id
        algo = safe_device(eval(algo_map[args.algo])(10, cfg), cfg.device)
        algo.policy.previous_mask = previous_mask

        if cfg.lifelong.algo == "PackNet":
            algo.eval()
            for module_idx, module in enumerate(algo.policy.modules()):
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                    weight = module.weight.data
                    mask = algo.previous_masks[module_idx].to(cfg.device)
                    weight[mask.eq(0)] = 0.0
                    weight[mask.gt(args.task_id + 1)] = 0.0
                if "BatchNorm" in str(type(module)) or "LayerNorm" in str(type(module)):
                    module.eval()

        algo.policy.load_state_dict(sd)

        if not hasattr(cfg.data, "task_order_index"):
            cfg.data.task_order_index = 0

    # get the benchmark the task belongs to
    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    if not use_agent_server:
        descriptions = [benchmark.get_task(i).language for i in range(10)]
        task_embs = get_task_embs(cfg, descriptions)
        benchmark.set_task_embs(task_embs)

    task = benchmark.get_task(args.task_id)
    print(f"任务指令: {task.language}")

    # 2. evaluate success rate
    if use_agent_server:
        save_folder = os.path.join(
            args.save_dir,
            f"{args.benchmark}_showvla_agent_on{args.task_id}.stats",
        )
        video_folder = os.path.join(
            args.save_dir,
            f"{args.benchmark}_showvla_agent_on{args.task_id}_videos",
        )
    elif args.algo == "multitask":
        save_folder = os.path.join(
            args.save_dir,
            f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_ep{args.ep}_on{args.task_id}.stats",
        )
        video_folder = os.path.join(
            args.save_dir,
            f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_ep{args.ep}_on{args.task_id}_videos",
        )
    else:
        save_folder = os.path.join(
            args.save_dir,
            f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_load{args.load_task}_on{args.task_id}.stats",
        )
        video_folder = os.path.join(
            args.save_dir,
            f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_load{args.load_task}_on{args.task_id}_videos",
        )

    test_loss = 0.0

    with Timer() as t, VideoWriter(video_folder, args.save_videos) as video_writer:
        env_args = {
            "bddl_file_name": os.path.join(
                cfg.bddl_folder, task.problem_folder, task.bddl_file
            ),
            "camera_heights": cfg.data.img_h,
            "camera_widths": cfg.data.img_w,
        }

        eval_num_procs = getattr(cfg.eval, "num_procs", 1)
        eval_n_eval = getattr(cfg.eval, "n_eval", 1)
        env_num = (
            args.agent_env_num if use_agent_server and args.agent_env_num is not None else min(eval_num_procs, eval_n_eval)
        )
        env = SubprocVectorEnv(
            [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
        )
        env.reset()
        env.seed(cfg.seed)
        if not use_agent_server:
            algo.reset()

        init_states_path = os.path.join(
            cfg.init_states_folder, task.problem_folder, task.init_states_file
        )
        init_states = torch.load(init_states_path)
        indices = np.arange(env_num) % init_states.shape[0]
        init_states_ = init_states[indices]

        dones = [False] * env_num
        steps = 0
        obs = env.set_init_state(init_states_)
        task_emb = benchmark.get_task_emb(args.task_id) if not use_agent_server else None
        action_buffers: Optional[List[List[np.ndarray]]] = (
            [[] for _ in range(env_num)] if use_agent_server else None
        )
        agentview_key = cfg.data.obs_key_mapping.get("agentview_rgb", "agentview_image")
        wrist_key = cfg.data.obs_key_mapping.get("eye_in_hand_rgb", "robot0_eye_in_hand_image")

        num_success = 0
        for _ in range(20):  # simulate the physics without any actions
            env.step(np.zeros((env_num, 7)))
        obs, _, _, _ = env.step(np.zeros((env_num, 7)))

        progress = tqdm(total=cfg.eval.max_steps, desc=f"Eval task {args.task_id}", leave=False)
        with torch.no_grad():
            while steps < cfg.eval.max_steps:
                steps += 1

                if use_agent_server:
                    env_action_dim = 7
                    if action_buffers is not None and all(len(buf) == 0 for buf in action_buffers):
                        obs_batch = [
                            _build_agent_obs(
                                obs[k],
                                instruction=task.language,
                                obs_key_mapping=cfg.data.obs_key_mapping,
                                target_size=args.camera_size,
                                flip=args.flip_agent_images,
                            )
                            for k in range(env_num)
                        ]
                        agent_outputs = agent_client.infer_actions(obs_batch)
                        for k, res in enumerate(agent_outputs):
                            prepared = []
                            if isinstance(res, dict) and "actions" in res:
                                arr = np.asarray(res["actions"], dtype=np.float32)
                                if args.agent_action_horizon is not None and args.agent_action_horizon > 0:
                                    arr = arr[: args.agent_action_horizon]
                            elif isinstance(res, dict) and "action" in res:
                                arr = np.asarray(res["action"], dtype=np.float32)
                                if args.agent_action_horizon is not None and args.agent_action_horizon > 0:
                                    arr = arr.reshape(1, -1)
                            else:
                                horizon = args.agent_action_horizon if args.agent_action_horizon and args.agent_action_horizon > 0 else 1
                                arr = np.zeros((horizon, env_action_dim), dtype=np.float32)
                            if arr.ndim == 1:
                                arr = arr.reshape(1, -1)
                            for act in arr:
                                act = act.reshape(-1)
                                if act.shape[0] > env_action_dim:
                                    act = act[:env_action_dim]
                                elif act.shape[0] < env_action_dim:
                                    pad = np.zeros(env_action_dim, dtype=np.float32)
                                    pad[: act.shape[0]] = act
                                    act = pad
                                prepared.append(act)
                            action_buffers[k] = prepared

                    actions_list = []
                    for k in range(env_num):
                        if action_buffers is not None and len(action_buffers[k]) > 0:
                            act = action_buffers[k].pop(0)
                        else:
                            act = np.zeros(env_action_dim, dtype=np.float32)
                        actions_list.append(act)
                    actions = np.stack(actions_list, axis=0)
                else:
                    data = raw_obs_to_tensor_obs(obs, task_emb, cfg)
                    actions = algo.policy.get_action(data)

                obs, reward, done, info = env.step(actions)
                video_writer.append_vector_obs(
                    obs, dones, camera_name=agentview_key, stream_name="agent"
                )
                video_writer.append_vector_obs(
                    obs, dones, camera_name=wrist_key, stream_name="wrist"
                )
                video_writer.append_vector_combined(
                    obs, dones, left_key=agentview_key, right_key=wrist_key, stream_name="combined"
                )
                progress.update(1)

                # check whether succeed
                for k in range(env_num):
                    dones[k] = dones[k] or done[k]
                if all(dones):
                    break

            for k in range(env_num):
                num_success += int(dones[k])

        success_rate = num_success / env_num
        env.close()
        progress.close()

        eval_stats = {
            "loss": test_loss,
            "success_rate": success_rate,
        }

        os.system(f"mkdir -p {args.save_dir}")
        torch.save(eval_stats, save_folder)
    print(
        f"[info] finish for ckpt at {run_folder} in {t.get_elapsed_time()} sec for rollouts"
    )
    print(f"Results are saved at {save_folder}")
    print(test_loss, success_rate)
    if agent_client is not None:
        agent_client.close()
    return eval_stats, save_folder


def main():
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
