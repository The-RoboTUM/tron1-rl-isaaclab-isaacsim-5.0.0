"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments

# 🧩 第一步：创建 ArgumentParser（命令行参数解析器）
# 这行代码的意思是：
#
# 创建一个命令行参数解析器（parser），并为它设置一个描述文字。
#
# 这就像你告诉 Python：“接下来我会定义有哪些参数可以从命令行输入。”

parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")

# 🧩 第二步：添加命令行参数（用户可输入的选项）
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=400, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=24000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--max_iterations", type=int, default=None, help="Maximum number of iterations to train.")
parser.add_argument("--save_interval", type=int, default=None, help="The number of iterations between saves")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Relative path to checkpoint file.")


# 这部分定义了常见的训练控制参数，比如：
#
# 参数	            类型	默认值	含义
# --video	        bool flag	False	是否录制视频
# --video_length	int	 400	每段视频的长度（步数）
# --video_interval	int	 24000	每隔多少步录一段视频
# --num_envs	    int	 None	同时并行的环境数量
# --max_iterations	int	 None	最大训练迭代次数
# --save_interval	int	 None	模型保存间隔
# --task	        str	 None	要训练的任务名称（如 Isaac-Anymal-v0）
# --seed	        int	 None	随机种子
# --checkpoint_path	str	 None	断点恢复路径

# 属于“普通 argparse 参数”，是给脚本直接使用的。
#
# 比如你运行：
#
# python train.py --task Isaac-Anymal-v0 --num_envs 2048 --video
#
# 脚本就能在这里捕获这些值。



# 🧩 第三步：添加 RSL-RL 相关参数
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# 这行会调用一个工具函数，往同一个 parser 里追加 RSL-RL 专属参数，比如：
#
# PPO 超参数（学习率、clip ratio、batch size、γ、λ 等）
#
# 网络结构（MLP 层数、hidden size）
#
# Runner 控制参数（log 目录、device、resume 等）


# 🧩 第四步：添加 AppLauncher（Isaac Sim 启动器）参数
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# 这行再往解析器里加一批与 Isaac Sim 本身相关的选项，比如：
#
# --headless：是否无界面渲染；
#
# --renderer：选择渲染后端（RTX, PathTracer, 2D）；
#
# --physics_dt：物理仿真步长；
#
# --device：CUDA 设备号；
#
# --enable_cameras：是否启用相机；
#
# 等等。


# 👉 所以到目前为止，这个 parser 已经包含三类参数：
#
# 通用训练参数（手动添加的那几条）
#
# RSL-RL 参数（算法配置）
#
# AppLauncher 参数（Isaac Sim 配置）


# 🧩 第五步：解析参数，并区分 Hydra 参数
args_cli, hydra_args = parser.parse_known_args()

# 这行非常关键！它的作用是：
#
# 把命令行中“脚本认识的参数，也就是代码在上面刚刚提到过的、定义过的那些参数，比如- -task ”解析到 args_cli；
#
# 把“不认识的参数，也就是代码在上面刚刚没有提到过的、定义过的那些参数”（即留给 Hydra 的那些）放进 hydra_args。

# 所以这行是把参数分流。


# 🧩 第六步：如果录视频，就启用相机
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# 逻辑很简单：

# 如果用户要求录视频（--video），则必须打开相机；

# 所以自动设置 enable_cameras=True，防止忘记加。

# 🧩 第七步：把剩下的参数交给 Hydra
# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args


# 🧩 第八步：启动 Isaac Sim 应用
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
# 这是启动 Omniverse Isaac Sim 仿真引擎的步骤。
#
# AppLauncher 是 NVIDIA 提供的一个包装类；
#
# 它根据 args_cli 参数（比如 --headless, --renderer, --enable_cameras）来配置 Isaac Sim；
#
# 启动后会返回一个 simulation_app 实例（相当于“仿真引擎在内存中已经运行”）。


"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

# from rsl_rl.runners import OnPolicyRunner
from rsl_rl.runner import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Import extensions to set up environment tasks
from bipedal_locomotion.utils.wrappers.rsl_rl import RslRlPpoAlgorithmMlpCfg


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# @hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main():
    """Train with RSL-RL agent."""
    # parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    agent_cfg: RslRlPpoAlgorithmMlpCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.save_interval is not None:
        agent_cfg.save_interval = args_cli.save_interval

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    # on_policy_runner_class = eval(agent_cfg.runner_type)
    # runner: OnPolicyRunner | OnPolicyRunnerMlp = on_policy_runner_class(
    #     env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
    # )
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # write git state to logs
    # runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        if args_cli.checkpoint_path is not None:
            resume_path = args_cli.checkpoint_path
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # set seed of the environment
    env.seed(agent_cfg.seed)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
