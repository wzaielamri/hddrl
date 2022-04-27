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


params = {'legend.fontsize': 'xx-large',
          'figure.figsize': (8, 6),
          'axes.labelsize': 'xx-large',
          'axes.titlesize': 'xx-large',
          'xtick.labelsize': 'xx-large',
          'ytick.labelsize': 'xx-large'}
plt.rcParams.update(params)


column_toKeep = ["custom_metrics/goal_reached", "custom_metrics/goal_reached_accuracy", "custom_metrics/cost_of_transport_mean",
                 "custom_metrics/velocity_mean", "episode_len_mean", "episode_reward_mean", ]

file_name = "all_seeds_4_0011235_NeewReward_env1600_scratchTransfer_10Trials"


labels = [
    "3_Cen_Sc",
    "3_Cen_TL",
    "3_Dec_Sc",
    "3_Dec_TL",
    "3_C_C_TL",
    "3_C_D_TL",
    # "4_D_C_TL",
    "3_D_D_TL",
]


def save_plot(all_exp_data, metric_ind, y_label, titel, architectures, seeds=False, save_name="", vmin=None, vmax=None):

    fig = plt.figure()
    ax_arch = fig.add_subplot(1, 1, 1)

    ax_arch.spines['top'].set_visible(False)
    ax_arch.spines['right'].set_visible(False)

    # ax_arch.set_yscale('log')
    ax_arch.set_xlim(0, 4e7)
    # ax_arch.set_ylim(0, 800)
    colors = [0, 1,  2, 3, 4, 6, 10, ]
    if seeds == False:
        for ind, i in enumerate(architectures):
            # Use matplotlib's fill_between() call to create error bars.
            if ind == 1 or ind == 2:
                ax_arch.fill_between(all_exp_data[metric_ind][i][-1][0], all_exp_data[metric_ind][i][2],
                                     all_exp_data[metric_ind][i][3], alpha=0.25, color=tableau20[colors[i]])  # tableau20[(i-0)*2 + 1])
            else:
                ax_arch.fill_between(all_exp_data[metric_ind][i][-1][0], all_exp_data[metric_ind][i][2],
                                     all_exp_data[metric_ind][i][3], alpha=0.25, color=tableau20[colors[i]+1])  # tableau20[(i-0)*2 + 1])
            ax_arch.plot(all_exp_data[metric_ind][i][-1][0], all_exp_data[metric_ind][i]
                         [0], color=tableau20[colors[i]], label=labels[i],)  # tableau20[(i-0)*2]

    else:
        for ind, j in enumerate(architectures):  # architectures
            for i in range(len(all_exp_data[metric_ind][j][-2])):  # trails
                if i == 0:
                    ax_arch.plot(all_exp_data[metric_ind][j][-1][0],
                                 all_exp_data[metric_ind][j][-2][i], color=tableau20[colors[j]], label=labels[j], alpha=0.5, )  # tableau20[(j-0)*2]
                else:
                    ax_arch.plot(all_exp_data[metric_ind][j][-1][0],
                                 all_exp_data[metric_ind][j][-2][i], color=tableau20[colors[j]], alpha=0.5, )  # tableau20[(j-0)*2]

    ax_arch.set_xlabel('timesteps', )
    ax_arch.set_ylabel(y_label, )

    ax_arch.set_title(titel, )
    #order = [0, 1, 3, 2, 4, 5, 6, 7]

    # add legend to plot
    #handles, labels_legend = plt.gca().get_legend_handles_labels()
    # plt.legend([handles[idx] for idx in order], [labels_legend[idx]
    #           for idx in order], ncol=2, loc='upper left')
    plt.legend(ncol=2, loc='upper left')
    plt.ylim((vmin, vmax))
    fig.tight_layout()

    plt.savefig("Results/fig/" +
                column_toKeep[metric_ind].split('/')[-1]+save_name+".png")
    # plt.show()


all_exp_data = np.load(
    "/home/nitro/clusteruni/masterarbeit/hddrl/Results/training_progression/"+file_name+".npy", allow_pickle=True, encoding="latin1")


# Plotting curves


save_plot(all_exp_data, 1, y_label='Average Ratio of Successful Episodes',
          titel=" Task 3: Average Ratio of Successful Episodes\n with Respect to the Overall Number of Episodes\n for the Different Architectres", architectures=[0, 1, 2, 3, 4, 5, 6], save_name="_3", vmax=1.4)
# save_plot(all_exp_data, 3, y_label='Average Velocity Towards Goal',
#          titel=" Task 3: Average Velocity Towards Goal\n for the Different Architectres", architectures=[0, 1, 2, 3, 4, 5, 6, 7], save_name="_3", vmin=-0.1,)
# save_plot(all_exp_data, 5, y_label='Average Return',
#          titel=" Task 3: Average Return\n for the Flat Architectures", architectures=[0, 1], save_name="_flat_3")
# save_plot(all_exp_data, 5, y_label='Average Return',
#          titel=" Task 3: Individual Average Return of 10 Trials\n for the Flat Architectures", architectures=[0, 1], seeds=True, save_name="_individual_flat_3")
# save_plot(all_exp_data, 5, y_label='Average Return',
#          titel=" Task 3: Average Return\n for the Hierarchical Architectures", architectures=[2, 3, 4], save_name="_hierearchical_3")
# save_plot(all_exp_data, 5, y_label='Average Return',
#          titel=" Task 3: Individual Average Return of 10 Trials\n for the Hierarchical Architectures", architectures=[2, 3, 4], seeds=True, save_name="_individual_hierarchical_3")
save_plot(all_exp_data, 5, y_label='Average Return',
          titel=" Task 3: Average Return\n for the Different Architectures", architectures=[0, 1, 2, 3, 4, 5, 6], save_name="_3", vmax=5)
save_plot(all_exp_data, 5, y_label='Average Return',
          titel=" Task 3: Individual Average Return of 10 Trials\n for the Different Architectures", architectures=[0, 1, 2, 3, 4, 5, 6], seeds=True, save_name="_individual_3", vmax=5)
