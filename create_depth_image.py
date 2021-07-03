import sys 
import cv2
import gym
import random
import numpy as np
from PIL import Image
from gym_grasping.envs.robot_sim_env import RobotSimEnv
import json
import argparse



def main(args):
    with open (args.param, "r") as f:
        config = json.load(f)
    path = "result"
    env= gym.make(config["env_name"], renderer='egl')
    obs = env.reset()
    print(obs)
    t_frame = 0
    for i_episode in range(10000):
        obs = env.reset()
        # gym.seed(i_episode)
        for step in range(100):
            t_frame += 1
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            depth_image = info["depth"]*255
            frame = cv2.imwrite("{}/depth{}.png".format(path, step), np.array(info["depth"]*255))
        cv2.imshow('im', info['depth'])
        frame = cv2.imwrite("{}/wi{}.png".format(path, step), np.array(obs))
        obs = next_obs
        if done:
            print("done")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--param', default="param.json", type=str)
    parser.add_argument('--locexp', default="experiments/kuka", type=str)
    arg = parser.parse_args()
    main(arg)
