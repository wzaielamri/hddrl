import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

"""
    save return during learning over time, taken from the rllib logs.

    Here, run on the system trained on flat terrain.
"""

# Important: requires detailed logs of results (not part of the git).
exp_path = [

    '/media/compute/homes/wzaielamri/ray_results/3_0_0_TwoRowMazeStatic_AllInfo_Sparse_Fq5_2Dense64LSTM64_NewReward_Env1600_Scratch_QuantrupedMultiEnv_Centralized_Maze',
    '/media/compute/homes/wzaielamri/ray_results/3_0_0_TwoRowMazeStatic_AllInfo_Sparse_Fq5_2Dense64LSTM64_NewReward_PPO0_VelocityCtrl_Env1600_QuantrupedMultiEnv_Centralized_Maze',

    '/media/compute/homes/wzaielamri/ray_results/3_1_0_TwoRowMazeStatic_AllInfo_Sparse_Fq5_2Dense64LSTM64_NewReward_Env1600_Scratch_QuantrupedMultiEnv_FullyDecentral_Maze',
    '/media/compute/homes/wzaielamri/ray_results/3_1_0_TwoRowMazeStatic_AllInfo_Sparse_Fq5_2Dense64LSTM64_NewReward_PPO0_VelocityCtrl_Env1600_QuantrupedMultiEnv_FullyDecentral_Maze',
]


checkpoint = 2500

file_name = "all_seeds_4_0011_NewReward_LSTM_env1600_scratchTransfer_10Trials"

experiment_dirs = [[os.path.join(exp_path_item, dI) for dI in os.listdir(
    exp_path_item) if os.path.isdir(os.path.join(exp_path_item, dI))] for exp_path_item in exp_path]


experiment_dirs = np.array(experiment_dirs)

metric_array = []

column_toKeep = ["custom_metrics/goal_reached", "custom_metrics/goal_reached_accuracy", "custom_metrics/cost_of_transport_mean",
                 "custom_metrics/velocity_reward_mean", "episode_len_mean", "episode_reward_mean", "timesteps_total"]

for metric in column_toKeep[: -1]:
    all_exp_data = []
    print("\nMetric: ", metric)

    for exp_ind, exp_dir in enumerate(experiment_dirs):
        print("\nExperiment: ", exp_ind)
        time_steps = []
        labels = []
        for i in range(0, len(exp_dir)):
            print("Seed: ", i, "  : ", exp_dir[i].split('/')[-1])

            # df = pd.read_csv(exp_dir[i]+'/progress.csv', usecols=column_ind)
            df = pd.concat([x.loc[:, column_toKeep] for x in pd.read_csv(
                exp_dir[i]+'/progress.csv', chunksize=200)])

            labels.append(exp_dir[i].split('/')[-1])

            metric_ind = list(df.keys()).index(metric)
            rew_new = (df.iloc[:, metric_ind].values[: checkpoint])
            if i == 0:
                reward_values = np.vstack([rew_new])
                timestep_ind = list(df.keys()).index("timesteps_total")
                time_steps.append(
                    df.iloc[:, timestep_ind].values[: checkpoint])
            else:
                reward_values = np.vstack([reward_values, rew_new])
        rew_mean = np.mean(reward_values, axis=0)
        rew_std = np.std(reward_values, axis=0)
        rew_lower_std = rew_mean - rew_std
        rew_upper_std = rew_mean + rew_std
        all_exp_data.append([rew_mean, rew_std, rew_lower_std,
                            rew_upper_std, labels, reward_values, time_steps])
    metric_array.append(all_exp_data)
np.save("/media/compute/homes/wzaielamri/masterarbeit/hddrl/Results/training_progression/" +
        file_name, metric_array)
