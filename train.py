# Copyright 2020
# Author: Christian Leininger <info2016frei@gmail.com>
 
import os
import sys 
import cv2
import gym
import time
import torch 
import random
import numpy as np
from collections import deque
from datetime import datetime
from replay_buffer import ReplayBuffer
from agent import TQC
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from gym_grasping.envs.robot_sim_env import RobotSimEnv
from framestack import FrameStack, mkdir
import torchvision.transforms.functional as TF
import logging



def set_egl_device(device):
    assert "EGL_VISIBLE_DEVICES" not in os.environ, "Do not manually set EGL_VISIBLE_DEVICES"
    try:
        egl_id = get_egl_device_id(cuda_id)
    except Exception: 
        logging.warning(
                "Couldn't find correct EGL device. Setting EGL_VISIBLE_DEVICE=0. "
                "When using DDP with many GPUs this can lead to OOM errors. "
                "Did you install PyBullet correctly? Please refer to VREnv README"
                )
        egl_id = 0
    os.environ["EGL_VISIBLE_DEVICES"] = str(egl_id)
    # logging.info(f"EGL_DEVICE_ID {egl_id} <==> CUDA_DEVICE_ID {cuda_id}")
 

def get_egl_device_id(cuda_id):
    """
    >>> i = get_egl_device_id(0)
    >>> isinstance(i, int)
    True
    """
    assert isinstance(cuda_id, int), "cuda_id has to be integer"
    dir_path = Path(__file__).absolute().parents[2] / "egl_check"
    dir_path = "/home/leiningc"
    if not os.path.isfile(dir_path / "EGL_options.o"):
        if os.environ.get("LOCAL_RANK", "0") == "0":
            print("Building EGL_options.o")
            subprocess.call(["bash", "build.sh"], cwd=dir_path)
        else:
            # In case EGL_options.o has to be built and multiprocessing is used, give rank 0 process time to build
            time.sleep(5)
    result = subprocess.run(["./EGL_options.o"], capture_output=True, cwd=dir_path)
    n = int(result.stderr.decode("utf-8").split(" of ")[1].split(".")[0])
    for egl_id in range(n):
        my_env = os.environ.copy()
        my_env["EGL_VISIBLE_DEVICE"] = str(egl_id)
        result = subprocess.run(["./EGL_options.o"], capture_output=True, cwd=dir_path, env=my_env)
        match = re.search(r"CUDA_DEVICE=[0-9]+", result.stdout.decode("utf-8"))
        if match:
            current_cuda_id = int(match[0].split("=")[1])
            if cuda_id == current_cuda_id:
                return egl_id
    raise EglDeviceNotFoundError


def evaluate_policy(policy, writer, total_timesteps, size, env, episode=5):
    """
    
    
    Args:
       param1(): policy
       param2(): writer
       param3(): episode default 1 number for path to save the video
    """

    path = mkdir("","eval/" + str(total_timesteps) + "/")
    print(path)
    avg_reward = 0.
    seeds = [x for x in range(episode)]
    goal= 0
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        env.seed(s)
        obs = env.reset()
        done = False
        for step in range(50):
            action = policy.select_action(np.array(obs))
            
            obs, reward, done, img = env.step(action)
            # img = cv2.resize(img.astype(np.float), (300,300))
            # cv2.imshow("wi", cv2.resize(obs[:,:,::-1], (300,300)))
            # frame = cv2.imwrite("{}/wi{}.png".format(path, step), np.array(obs))
            if done:
                avg_reward += reward 
                if step < 49:
                    goal +=1
                break
            #cv2.waitKey(10)
            avg_reward += reward 

    avg_reward /= len(seeds)
    writer.add_scalar('Evaluation reward', avg_reward, total_timesteps)
    print ("---------------------------------------")
    print ("Average Reward over the Evaluation Step: {}  goal reached {} of  {} ".format(avg_reward, goal, episode))
    print ("---------------------------------------")
    return avg_reward




def write_into_file(pathname, text):
    """
    """
    with open(pathname+".txt", "a") as myfile:
        myfile.write(text)
        myfile.write('\n')

def time_format(sec):
    """
    
    Args:
        param1():

    """
    hours = sec // 3600
    rem = sec - hours * 3600
    mins = rem // 60
    secs = rem - mins * 60
    return hours, mins, round(secs,2)



def train_agent(config):
    """

    Args:
    """
    print("train")
    now = datetime.now()    
    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    file_name = str(config["locexp"]) + "/pytorch_models/{}".format(str(config["env_name"]))    
    pathname = dt_string 
    tensorboard_name = str(config["locexp"]) + '/runs/' + pathname
    print("Tensorboard filename ", tensorboard_name)
    writer = SummaryWriter(tensorboard_name)
    size = config["size"]
    print("create env ..")
    cuda = int(os.environ['CUDA_VISIBLE_DEVICES'])
    set_egl_device(cuda)
    env = gym.make(config["env_name"], renderer='egl')
    #env = gym.make(config["env_name"])
    print("... done")
    env = FrameStack(env, config)
    print("...   framdone")
    obs = env.reset()
    print("...   reset done")
    print("state ", obs.shape)
    state_dim = 512
    print("State dim, " , state_dim)
    action_dim = 5 
    print("action_dim ", action_dim)
    max_action = 1
    config["target_entropy"] =-np.prod(action_dim)
    obs_shape = (config["history_length"], size, size)
    action_shape = (action_dim,)
    print("obs", obs_shape)
    print("act", action_shape)
    policy = TQC(state_dim, action_dim, max_action, config)    
    replay_buffer = ReplayBuffer(obs_shape, action_shape, int(config["buffer_size"]), config["image_pad"], config["device"])
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    t0 = time.time()
    scores_window = deque(maxlen=100) 
    episode_reward = 0
    evaluations = []
    tb_update_counter = 0
    skip = 0
    print("continue ", config["continue_training"])
    if config["continue_training"]:
        replay_buffer.load_memory(config["memory_path"])
        print("load at point replay buffer {}".format(replay_buffer.idx))
        policy.load(config["model_path"])
        total_timesteps = config["timestep"]
        skip = config["timestep"]
        episode_num = config["episode_num"]
    # evaluations.append(evaluate_policy(policy, writer, total_timesteps, size, env))
    # save_model = file_name + '-{}reward_{:.2f}'.format(episode_num, evaluations[-1])
    # policy.save(save_model)
    done_counter =  deque(maxlen=100)
    model_better_90 = False
    model_better_95 = False
    model_better_98 = False
    saved_buffer = True
    time_save_buffer = 86000 
    
    while total_timesteps <  config["max_timesteps"]:
        tb_update_counter += 1
        # If the episode is done
        if done:
            episode_num += 1
            #env.seed(random.randint(0, 100))
            scores_window.append(episode_reward)
            average_mean = np.mean(scores_window)
            if tb_update_counter > config["tensorboard_freq"]:
                print("Write tensorboard")
                tb_update_counter = 0
                writer.add_scalar('Reward', episode_reward, total_timesteps)
                writer.add_scalar('Reward mean ', average_mean, total_timesteps)
                writer.flush()
            # If we are not at the very beginning, we start the training process of the model
            if total_timesteps != skip:
                if episode_timesteps < 50:
                    done_counter.append(1)
                else:
                    done_counter.append(0)
                goals = sum(done_counter)
                text = "Total Timesteps: {} Episode Num: {} ".format(total_timesteps, episode_num) 
                text += "Episode steps {} ".format(episode_timesteps)
                text += "Goal last 100 ep : {} ".format(goals)
                text += "Reward: {:.2f}  Average Re: {:.2f} Time: {}".format(episode_reward, np.mean(scores_window), time_format(time.time()-t0))
                writer.add_scalar('Goal_freq', goals, total_timesteps)
                
                print(text)
                write_into_file(os.path.join(str(config["locexp"]), pathname), text)
            # We evaluate the episode and we save the policy
            time_passed = time.time() - t0
            
            if time_passed > time_save_buffer and saved_buffer:
                saved_buffer = False
                save_model = file_name + '-{}_time'.format(episode_num)
                print("Save model to {}".format(save_model))
                policy.save(save_model)
                path_memory = "save_memory-time"
                replay_buffer.save_memory(os.path.join(str(config["locexp"]), path_memory))
                with open(config["locexp"] + '/total_timesteps.txt', 'w') as f:
                    f.write("{}".format(total_timesteps)) 

            if total_timesteps % config["eval_freq"] == 0:
                evaluations.append(evaluate_policy(policy, writer, total_timesteps, size,  env))
                save_model = file_name + '-{}reward_{:.2f}'.format(episode_num, evaluations[-1])
                print("Save model to {}".format(save_model))
                path_memory = "save_memory-{}".format(episode_num)
                replay_buffer.save_memory(os.path.join(str(config["locexp"]), path_memory))
                policy.save(save_model)
            # When the training step is done, we reset the state of the environment
            obs = env.reset()

            # Set the Done to False
            done = False
            # Set rewards and episode timesteps to zero
            episode_reward = 0
            episode_timesteps = 0
        # Before 10000 timesteps, we play random actions
        if total_timesteps < config["start_timesteps"]:
            action = env.action_space.sample()
        else: # After 10000 timesteps, we switch to the model
            action = policy.select_action(obs)
        # The agent performs the action in the environment, then reaches the next state and receives the reward
        new_obs, reward, done, _ = env.step(action)
        # print(reward)
        #frame = cv2.imshow("wi", np.array(new_obs))
        #cv2.waitKey(10)
        done = float(done)
        
        # We check if the episode is done
        #done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
        done_bool = 0 if episode_timesteps + 1 == config["max_episode_steps"] else float(done)
        if episode_timesteps + 1 == config["max_episode_steps"]:
            done = True
        # We increase the total reward
        episode_reward += reward
        # We store the new transition into the Experience Replay memory (ReplayBuffer)
        replay_buffer.add(obs, action, reward, new_obs, done, done_bool)
        # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
        obs = new_obs
        if total_timesteps > config["start_timesteps"]:
            for i in range(1):
                policy.train(replay_buffer, writer, 1)
        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1
