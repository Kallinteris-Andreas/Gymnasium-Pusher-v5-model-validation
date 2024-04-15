import gymnasium
import numpy as np
import copy
import os
import argparse
import cloudpickle
import torch
import mujoco

import stable_baselines3
from stable_baselines3 import TD3, PPO, A2C, SAC, DQN, DDPG
from stable_baselines3.common.logger import configure
# from stable_baselines3.common.callbacks import EvalCallback
from my_eval import EvalCallback
from gymnasium.wrappers import TransformReward, PassiveEnvChecker, OrderEnforcing, TimeLimit
from gymnasium.wrappers import RescaleAction
from stable_baselines3.common.monitor import Monitor

assert np.__version__ == "1.26.4"
assert gymnasium.__version__ == "1.0.0a1"
assert cloudpickle.__version__ == "3.0.0"
assert torch.__version__ == "2.2.2+cu121"
assert stable_baselines3.__version__ == "2.3.0a3"
#assert mujoco.__version__ == "3.1.3"


def make_env(env_id: str, render_mode=None):
    if env_id == "Pusher-v5rc4":
        return gymnasium.make("Pusher-v5", xml_file="./pusher_v5rc4.xml", render_mode=render_mode, width=1280, height=720)
    if env_id == "Pusher-v5rc5":
        return gymnasium.make("Pusher-v5", xml_file="./pusher_v5rc5.xml", render_mode=render_mode, width=1280, height=720)
    if env_id == "Pusher-v5rc6":
        return gymnasium.make("Pusher-v5", xml_file="./pusher_v5rc6.xml", render_mode=render_mode, width=1280, height=720)
    #if env_id == "Humanoid-v5":
    #if env_id == "Humanoid-v5":
    else:
        return gymnasium.make(env_id, render_mode=render_mode, width=1280, height=720)


def make_model(algorithm: str):
    match args.algo:
        case "TD3":  # note does not work with Discrete
            return TD3("MlpPolicy", env, seed=run, verbose=1, device='cuda', learning_starts=100)
            #return TD3("MlpPolicy", env, seed=run, verbose=1, device='cuda', learning_starts=100, gamma=1)  # Swimmer Only
        case "DDPG":  # note does not work with Discrete
            return DDPG("MlpPolicy", env, seed=run, verbose=1, device='cuda', learning_starts=100)
        case "PPO":
            return PPO("MlpPolicy", env, seed=run, verbose=1, device='cuda')
        case "SAC":  # note does not work with Discrete
            return SAC("MlpPolicy", env, seed=run, verbose=1, device='cuda', learning_starts=100)
        case "A2C":
            return A2C("MlpPolicy", env, seed=run, verbose=1, device='cuda')
        case "DQN":
            return DQN("MlpPolicy", env, seed=run, verbose=1, device='cuda', learning_starts=100)


parser = argparse.ArgumentParser()
parser.add_argument("--algo", default="TD3")
parser.add_argument("--env_id")
parser.add_argument("--eval_env_id", default=None)
parser.add_argument("--starting_run", default=0, type=int)
args = parser.parse_args()

RUNS = 10  # Number of Statistical Runs
TOTAL_TIME_STEPS = 2_000_000
EVAL_SEED = 1234
EVAL_FREQ = 500
EVAL_ENVS = 50


for run in range(args.starting_run, RUNS):
    env = Monitor(make_env(args.env_id))
    #env = Monitor(RescaleAction(make_env(args.env_id), min_action=-1, max_action=1))
    if args.eval_env_id is None:
        eval_env = copy.deepcopy(env)
    else:
        eval_env = Monitor(make_env(args.eval_env_id))
    eval_path = f"results/{args.env_id}/{args.algo}/run_" + str(run)

    assert not os.path.exists(eval_path)

    #eval_callback = EvalCallback(eval_env, best_model_save_path=eval_path, log_path=eval_path, n_eval_episodes=EVAL_ENVS, eval_freq=EVAL_FREQ, deterministic=True, render=False, verbose=True)
    eval_callback = EvalCallback(eval_env, best_model_save_path=eval_path, log_path=eval_path, n_eval_episodes=EVAL_ENVS, eval_freq=EVAL_FREQ, deterministic=True, render=False, verbose=True, seed=EVAL_SEED)

    model = make_model(args.algo)
    # model.set_logger(configure(eval_path, ["stdout", "csv"]))
    model.set_logger(configure(eval_path, ["csv"]))

    model.learn(total_timesteps=TOTAL_TIME_STEPS, callback=eval_callback)
    print(f"Finished run: {run}")
