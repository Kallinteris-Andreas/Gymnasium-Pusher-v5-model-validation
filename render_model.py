import gymnasium as gym
import gymnasium
#from gymnasium.experimental.wrappers import RescaleActionV0
import time
import argparse
#import moviepy
from stable_baselines3.common.vec_env import VecVideoRecorder

import numpy as np

from stable_baselines3 import TD3, PPO, A2C, SAC, DDPG, DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.wrappers import RescaleAction


#from bench import make_env
def make_env(env_id: str, render_mode=None):
    if env_id == "Pusher-v5rc4":
        return gymnasium.make("Pusher-v5", xml_file="./pusher_v5rc4.xml", render_mode=render_mode, width=1280, height=720)
    #if env_id == "Humanoid-v5":
        #return RescaleAction(gymnasium.make(env-id, render_mode=render_mode, width=1280, height=720), min_action=-1, max_action=1)
    if env_id == "Pusher-v5rc5":
        return gymnasium.make("Pusher-v5", xml_file="./pusher_v5rc5.xml", render_mode=render_mode, width=1280, height=720)
    if env_id == "Pusher-v5rc6":
        return gymnasium.make("Pusher-v5", xml_file="./pusher_v5rc6.xml", render_mode=render_mode, width=1280, height=720)
    else:
        return gymnasium.make(env_id, render_mode=render_mode, width=1280, height=720)


parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="render", type=str)
parser.add_argument("--env", type=str)
parser.add_argument("--algo", type=str)
parser.add_argument("--run", default=0, type=int)
args = parser.parse_args()


match args.mode:
    case "render":
        RENDER_MODE = "human"
    case "info":
        RENDER_MODE = "rgb_array"
    case "eval":
        RENDER_MODE = "rgb_array"
    case "video":
        RENDER_MODE = "rgb_array"
    case _:
        assert False

eval_env = make_env(f"{args.env}", render_mode=RENDER_MODE)
# eval_env = gymnasium.make(f"{args.env}", render_mode=RENDER_MODE)
# eval_env = RescaleActionV0(eval_env, min_action=-1, max_action=1)


# make model
model_path = f"./results/{args.env}/{args.algo}/run_{args.run}/best_model"
match args.algo:
    case "SAC":
        model = SAC.load(path=model_path, env=eval_env, device='cpu')
    case "TD3":
        model = TD3.load(path=model_path, env=eval_env, device='cpu')
    case "DDPG":
        model = DDPG.load(path=model_path, env=eval_env, device='cpu')
    case "DQN":
        model = DQN.load(path=model_path, env=eval_env, device='cpu')
    case "PPO":
        model = PPO.load(path=model_path, env=eval_env, device='cpu')
    case "A2C":
        model = A2C.load(path=model_path, env=eval_env, device='cpu')


#
# RECORD VIDEO
#
if args.mode == "video":
    video_folder = "videos/"
    video_length = 1000
    VIDEO_NAME = f"{args.env}_{args.algo}_run_{args.run}"

    frame_list = []

    obs = model.get_env().reset()
    for _ in range(video_length + 1):
        action, _state = model.predict(obs, deterministic=True)
        obs, _, _, _ = model.get_env().step(action)
        frame_list.append(model.get_env().render())

    model.get_env().close()

    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    import os
    clip = ImageSequenceClip(frame_list, fps=eval_env.metadata["render_fps"])
    #moviepy_logger = None if self.disable_logger else "bar"
    path = os.path.join(video_folder, f"{VIDEO_NAME}.mp4")
    clip.write_videofile(path, logger="bar")
    #clip.write_videofile(path, logger=moviepy_logger)
    exit()

    #vec_env = VecVideoRecorder(model.get_env(), video_folder, record_video_trigger=lambda x: x == 0, video_length=video_length, name_prefix=f"{VIDEO_NAME}")
    #obs = vec_env.reset()
    #for _ in range(video_length + 1):
        #action, _state = model.predict(obs, deterministic=True)
        #obs, _, _, _ = vec_env.step(action)
    # Save the video
    #vec_env.close()
    #breakpoint()


#
# Evaluate Policy
if args.mode == "eval":
    avg_return, std_return = evaluate_policy(model, eval_env, n_eval_episodes=1000)
    print(f"the average return is {avg_return}")


#
# Render Human
#
STEPS = 10000
if args.mode in ["render", "info"]:
    vec_env = model.get_env()
    obs = vec_env.reset()
    infos = []
    for step in range(STEPS):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        #print(action)
        #print(info)
        infos.append(info)
        if args.mode == "render":
            time.sleep(0.100)

    print(f"reward_foward = {sum([info[0]['reward_forward']for info in infos])/STEPS}")
    print(f"reward_ctrl = {sum([info[0]['reward_ctrl']for info in infos])/STEPS}")
    print(f"reward_contact = {sum([info[0]['reward_contact']for info in infos])/STEPS}")
    print(f"reward_survive = {sum([info[0]['reward_survive']for info in infos])/STEPS}")

