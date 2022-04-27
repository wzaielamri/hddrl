import ray
import pickle5 as pickle
import os
import gym
import numpy as np
from ray.tune.registry import get_trainable_cls
from ray.rllib.evaluation.worker_set import WorkerSet
from maze_envs.quantruped_centralizedController_environment import Quantruped_Centralized_Env
from ray.rllib.agents.ppo import PPOTrainer

#import simulation_envs
import maze_envs
import soccer_envs
import seek_envs


import models
from evaluation.rollout_episodes import rollout_episodes

"""
    Visualizing a learned (multiagent) controller,
    for evaluation or visualisation.
    
    This is adapted from rllib's rollout.py
    (github.com/ray/rllib/rollout.py)
"""

# Setting number of steps and episodes
num_steps = int(3000)
num_episodes = int(100)

render = False
camera_name = "side_fixed"  # "top_fixed"

ray.init(num_cpus=2, ignore_reinit_error=True)

smoothness_list = [1, 0.8, 0.6]

# Selecting checkpoint to load
exp_path = [
    #
    '/media/compute/homes/wzaielamri/ray_results/3_0_0_TwoRowMazeStatic_AllInfo_Sparse_Fq5_2Dense64LSTM64_NewReward_Env1600_Scratch_QuantrupedMultiEnv_Centralized_Maze',
    '/media/compute/homes/wzaielamri/ray_results/3_0_0_TwoRowMazeStatic_AllInfo_Sparse_Fq5_2Dense64LSTM64_NewReward_PPO0_VelocityCtrl_Env1600_QuantrupedMultiEnv_Centralized_Maze',

    '/media/compute/homes/wzaielamri/ray_results/3_1_0_TwoRowMazeStatic_AllInfo_Sparse_Fq5_2Dense64LSTM64_NewReward_Env1600_Scratch_QuantrupedMultiEnv_FullyDecentral_Maze',
    '/media/compute/homes/wzaielamri/ray_results/3_1_0_TwoRowMazeStatic_AllInfo_Sparse_Fq5_2Dense64LSTM64_NewReward_PPO0_VelocityCtrl_Env1600_QuantrupedMultiEnv_FullyDecentral_Maze',
]


config_checkpoints = [[os.path.join(exp_path_item, dI)+"/checkpoint_002500/checkpoint-2500" for dI in os.listdir(
    exp_path_item) if os.path.isdir(os.path.join(exp_path_item, dI))] for exp_path_item in exp_path]


config_checkpoints = np.array(config_checkpoints)

#config_checkpoints = [config_checkpoints[0][0]]

text_file = open(
    "/media/compute/homes/wzaielamri/masterarbeit/hddrl/Results/Output_4_0011_NewReward_LSTM_scratchTransfer_full_10Trials.txt", "w")

text_latex_file = open(
    "/media/compute/homes/wzaielamri/masterarbeit/hddrl/Results/Output_4_0011_NewReward_LSTM_scratchTransfer_latex_10Trials.txt", "w")


# Afterwards put together using
# ffmpeg -framerate/home/nitro/clusteruni/ray_results/0_2_0_velocityCtrl_DifferentAngleNearTarget_DirectionOnly_QuantrupedMultiEnv_Hierarchical/PPO_QuantrupedMultiEnv_Hierarchical_588b0_00003_3_2022-02-10_14-35-44/checkpoint_002500ttern_type g2500 -i '*.png' -filter:v scale=720:-1 -vcodec libx264 -pix_fmt yuv420p -g 1 out.mp4
for smoothness in smoothness_list:
    text_latex_file.write("\n\nSmoothness: " + str(smoothness) + "\n")

    for i in range(0, len(exp_path)):

        reward_eps_list = []
        steps_eps_list = []
        dist_eps_list = []
        power_total_eps_list = []
        vel_eps_list = []
        cot_eps_list = []
        goal_reached_list = []
        avg_power_success_list = []
        total_power_success_list = []

        for config_checkpoint in config_checkpoints[i]:
            config_dir = os.path.dirname(config_checkpoint)
            config_path = os.path.join(config_dir, "params.pkl")

            # Loading configuration for checkpoint.
            if not os.path.exists(config_path):
                config_path = os.path.join(config_dir, "../params.pkl")
            if os.path.exists(config_path):
                with open(config_path, "rb") as f:
                    config = pickle.load(f)
            #env = Quantruped_Centralized_Env(config)
            # Starting ray and setting up ray.
            if "num_workers" in config:
                config["num_workers"] = min(1, config["num_workers"])
            cls = get_trainable_cls('PPO')
            # Setting config values (required for compatibility between versions)
            config["create_env_on_driver"] = True
            config['env_config']['hf_smoothness'] = smoothness
            if "no_eager_on_workers" in config:
                del config["no_eager_on_workers"]

            #config["env"] = "QuantrupedMultiEnv_Hierarchical_Maze"
            #config['env_config']['target_reward'] = num_steps

            config['num_envs_per_worker'] = 1  # 4

            #config['env_config']['config_checkpoint'] = '/home/nitro/clusteruni/ray_results/0_2_0_velocity_DifferentAngleNearTarget_Fq5_LS8_FF_HLCOnlyTarget_QuantrupedMultiEnv_Hierarchical/PPO_QuantrupedMultiEnv_Hierarchical_28e97_00004_4_2021-11-22_13-38-57/checkpoint_002500/checkpoint-2500'
            #config["num_cpus_per_worker"] = 2

            # Load state from checkpoint.
            agent = cls(env=config['env'], config=config)
            agent.restore(config_checkpoint)

            # Create the checkpoint

            # agent.restore(config_checkpoint)
            # Retrieve environment for the trained agent.
            if hasattr(agent, "workers") and isinstance(agent.workers, WorkerSet):
                env = agent.workers.local_worker().env

            # save_image_dir = "/home/nitro/masterarbeit/hddrl/videos/"+config_path.partition('MultiEnv_')[2].partition(
            #    '/')[0] + config_path.partition('MultiEnv_')[2].partition('/')[1][:-1] + '_smoothn_' + str(smoothness)
            save_image_dir = "/home/nitro/Desktop/masterarbeit/hddrl/videos/" + \
                config_checkpoint.split("/")[-4]

            if render:
                try:
                    os.mkdir(save_image_dir)
                    print("new directory: ", save_image_dir)
                except:
                    print("File exists!!!: ", save_image_dir,
                          "\nDeleting Files!!!")
                    try:
                        os.system("rm "+save_image_dir+"/*")
                        os.system("rm "+save_image_dir + "_output.mp4")
                    except:
                        continue

            # Rolling out simulation = stepping through simulation.
            reward_eps, steps_eps, dist_eps, power_total_eps, vel_eps, cot_eps, goal_reached, avg_power_success, total_power_success = rollout_episodes(env, agent, num_episodes=num_episodes,
                                                                                                                                                        num_steps=num_steps, render=render, camera_name=camera_name, plot=False, save_images=save_image_dir+"/img_")
            reward_eps_list.append(reward_eps)
            steps_eps_list.append(steps_eps)
            dist_eps_list.append(dist_eps)
            power_total_eps_list.append(power_total_eps)
            vel_eps_list.append(vel_eps)
            cot_eps_list.append(cot_eps)
            goal_reached_list.append(goal_reached)
            avg_power_success_list = avg_power_success_list + avg_power_success
            total_power_success_list = total_power_success_list+total_power_success
            agent.stop()

        if avg_power_success_list:  # list not empty
            text_file.write("Architecture: " + str(config["env"]) +
                            " - Smoothness: " + str(smoothness) +
                            " - Mean return: " + str(np.mean(reward_eps_list)) + "/" + str(np.std(reward_eps_list)) +
                            " - Steps: " + str(np.mean(steps_eps_list)) + "/" + str(np.std(steps_eps_list)) +
                            " - Distance: " + str(np.mean(dist_eps_list)) + "/" + str(np.std(dist_eps_list)) +
                            " - Power Total: " + str(np.mean(power_total_eps_list)) + "/" + str(np.std(power_total_eps_list)) +
                            " - Average Power Success: " + str(np.mean(avg_power_success_list)) + "/" + str(np.std(avg_power_success_list)) +
                            " - Total Power Success: " + str(np.mean(total_power_success_list)) + "/" + str(np.std(total_power_success_list)) +
                            " - Cum. Velocity: " + str(np.mean(vel_eps_list)) + "/" + str(np.std(vel_eps_list)) +
                            " - CoT: " + str(np.mean(cot_eps_list)) + "/" + str(np.std(cot_eps_list)) +
                            " - Goal: " + str(np.sum(goal_reached_list))+"\n")

            text_latex_file.write("Architecture: " + str(config["env"]) +
                                  " - Mean: " + str(round(np.mean(reward_eps_list), 2)) + " & " + str(round((np.sum(goal_reached_list)/1000), 4)) + " & " + str(round(np.mean(total_power_success_list), 2)) +
                                  " - Std: (" + str(round(np.std(reward_eps_list), 2)) + ") & & (" + str(round(np.std(total_power_success_list), 2)) + ") \n")
        else:
            text_file.write("Architecture: " + str(config["env"]) +
                            " - Smoothness: " + str(smoothness) +
                            " - Mean return: " + str(np.mean(reward_eps_list)) + "/" + str(np.std(reward_eps_list)) +
                            " - Steps: " + str(np.mean(steps_eps_list)) + "/" + str(np.std(steps_eps_list)) +
                            " - Distance: " + str(np.mean(dist_eps_list)) + "/" + str(np.std(dist_eps_list)) +
                            " - Power Total: " + str(np.mean(power_total_eps_list)) + "/" + str(np.std(power_total_eps_list)) +
                            " - Average Power Success:  empty list" +
                            " - Total Power Success: empty list" +
                            " - Cum. Velocity: " + str(np.mean(vel_eps_list)) + "/" + str(np.std(vel_eps_list)) +
                            " - CoT: " + str(np.mean(cot_eps_list)) + "/" + str(np.std(cot_eps_list)) +
                            " - Goal: " + str(np.sum(goal_reached_list))+"\n")

            text_latex_file.write("Architecture: " + str(config["env"]) +
                                  " - Mean: " + str(round(np.mean(reward_eps_list), 2)) + " & " + str(round((np.sum(goal_reached_list)/1000), 4)) + " &  empty list" +
                                  " - Std: (" + str(round(np.std(reward_eps_list), 2)) + ") & & (-) \n")

    if render:
        # get a video
        os.system('ffmpeg -framerate 20 -pattern_type glob -i "' + save_image_dir +
                  '/img_*.png" -filter:v scale=1280:-1 -vcodec libx264 -pix_fmt yuv420p -g 1 ' + save_image_dir + '_output.mp4')

text_file.close()
text_latex_file.close()
