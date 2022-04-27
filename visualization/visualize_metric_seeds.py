import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

"""
    Visualizes metrics during learning over time, taken from the rllib logs.
    
    Here, run on the system trained on flat terrain.
"""

# These are the "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.edgecolor'] = '#333F4B'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.color'] = '#333F4B'
plt.rcParams['ytick.color'] = '#333F4B'

# Remove Type 3 fonts for latex
plt.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42

# Important: requires detailed logs of results (not part of the git).

column_toKeep = ["custom_metrics/goal_reached", "custom_metrics/goal_reached_accuracy",
                 "custom_metrics/velocity_reward_mean", "episode_len_mean", "episode_reward_mean", ]


all_exp_data = np.load(
    "/home/nitro/clusteruni/masterarbeit/hddrl/visualization/all_seeds_2_1_hard_NoSplit_AllInfo_ppo_012.npy", allow_pickle=True, encoding="latin1")


file_name = "all_seeds_2_1_hard_NoSplit_AllInfo_ppo_012"
exp = 0

# Plotting functions
fig = plt.figure(figsize=(20, 10))


for metric_ind, metric in enumerate(column_toKeep):

    # Remove the plot frame lines. They are unnecessary chartjunk.
    ax_arch = plt.subplot(2, 3, metric_ind+1)
    ax_arch.spines["top"].set_visible(False)
    ax_arch.spines["right"].set_visible(False)

    # ax_arch.set_yscale('log')
    ax_arch.set_xlim(0, 4e7)
    #ax_arch.set_ylim(0, 800)

    for i in [3, 4, 5]:  # range(0, len(all_exp_data[metric_ind][exp][-2])):

        # Use matplotlib's fill_between() call to create error bars.
        plt.plot(all_exp_data[metric_ind][exp][-1][0],
                 all_exp_data[metric_ind][exp][-2][i], color=tableau20[i], label=all_exp_data[metric_ind][exp][-3][i].split('_')[7:9])

        ax_arch.set_xlabel('timesteps', fontsize=14)
    # ax_arch.set_ylabel(column_toKeep[metric_ind].split(
    #    '/')[-1]+' Return per Episode', fontsize=14)
    #plt.plot([0,500], [200,200], color=tableau20[6], linestyle='--')
    plt.title(column_toKeep[metric_ind].split('/')[-1])
    plt.legend(loc="lower right")

plt.savefig(file_name+".png")
plt.show()
