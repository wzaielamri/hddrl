######################
# Task: Training Task Maze
######################


import argparse
import models
import maze_envs
from ray.rllib.agents.callbacks import MultiCallbacks
from ray.rllib.agents.callbacks import DefaultCallbacks
import time
from ray.tune import grid_search
from ray import tune
from ray.tune.logger import pretty_print
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
import ray.rllib.agents.ppo as ppo
import ray
import numpy as np
import gym
from gym import spaces
from typing import Dict

from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

import os
from datetime import datetime
import tempfile
from ray.tune.logger import UnifiedLogger
from pathlib import Path
import pickle5 as pickle

# Switch between different approaches.
parser = argparse.ArgumentParser()
parser.add_argument("--policy_scope", required=False)
parser.add_argument("--num_trials", required=False)
parser.add_argument("--config_checkpoint", required=False)
parser.add_argument("--trial_name", required=False)

args = parser.parse_args()
# Possible values:
#   QuantrupedMultiEnv_Centralized - single controller, global information
#   QuantrupedMultiEnv_FullyDecentral - four decentralized controlller, information
#       from the controlled leg only
#   QuantrupedMultiEnv_SingleNeighbor - four decentralized controlller, information
#       from the controlled leg plus neighbor (ccw)
#   QuantrupedMultiEnv_SingleDiagonal - four decentralized controlller, information
#       from the controlled leg plus diagonal
#   QuantrupedMultiEnv_SingleToFront - four decentralized controlller, information
#       from the controlled leg plus one neighbor, for front legs from hind legs
#       for hind legs, the other hind leg
#   QuantrupedMultiEnv_Local - four decentralized controlller, information
#       from the controlled leg plus both neighboring legs
#   QuantrupedMultiEnv_TwoSides - two decentralized controlller, one for each side,
#       information from the controlled legs
#   QuantrupedMultiEnv_TwoDiags - two decentralized controlller, controlling a pair of
#       diagonal legs,
#       information from the controlled legs
#   QuantrupedMultiEnv_FullyDecentralGlobalCost - four decentralized controlller, information
#       from the controlled leg; variation: global costs are used.

policy_frozen = []
experiment_id = "None"
if 'policy_scope' in args and args.policy_scope:
    policy_scope = args.policy_scope
else:
    policy_scope = 'QuantrupedMultiEnv_Centralized_Maze'
    experiment_id = "3_0_0"
    policies_to_train = ["central_policy"]
    policy_frozen = ["central_policy"]

if policy_scope == "QuantrupedMultiEnv_FullyDecentral_Maze":
    from maze_envs.quantruped_fourDecentralizedController_environments import QuantrupedFullyDecentralizedEnv as QuantrupedEnv
    experiment_id = "3_1_0"
    policies_to_train = ["policy_FL", "policy_HL", "policy_HR", "policy_FR"]
    policy_frozen = ["policy_FL", "policy_HL", "policy_HR", "policy_FR"]

elif policy_scope == "QuantrupedMultiEnv_Hierarchical_Maze":
    from maze_envs.quantruped_hierarchicalController_environments import QuantrupedHierarchicalEnv as QuantrupedEnv
    policies_to_train = ["high_level_policy"]
    experiment_id = "3_2_0"
    policy_frozen = ["low_level_policy"]
elif policy_scope == "QuantrupedMultiEnv_HierarchicalFullyDecentralized_Maze":
    from maze_envs.quantruped_hierarchicalController_environments import QuantrupedHierarchicalFullyDecentralizedEnv as QuantrupedEnv
    policies_to_train = ["high_level_policy"]
    experiment_id = "3_3_0"
    policy_frozen = ["low_level_FL_policy",
                     "low_level_HL_policy", "low_level_HR_policy", "low_level_FR_policy"]
elif policy_scope == "QuantrupedMultiEnv_FullyDecentralizedHierarchical_Maze":
    from maze_envs.quantruped_decentralizedHierarchicalController_environments import QuantrupedFullyDecentralizedHierarchicalEnv as QuantrupedEnv
    policies_to_train = ["high_level_FL_policy", "high_level_HL_policy",
                         "high_level_HR_policy", "high_level_FR_policy"]
    experiment_id = "3_4_0"
    policy_frozen = ["low_level_policy"]
elif policy_scope == "QuantrupedMultiEnv_FullyDecentralizedHierarchicalFullyDecentralized_Maze":
    from maze_envs.quantruped_decentralizedHierarchicalController_environments import QuantrupedFullyDecentralizedHierarchicalFullyDecentralizedEnv as QuantrupedEnv
    policies_to_train = ["high_level_FL_policy", "high_level_HL_policy",
                         "high_level_HR_policy", "high_level_FR_policy"]
    experiment_id = "3_5_0"
    policy_frozen = ["low_level_FL_policy",
                     "low_level_HL_policy", "low_level_HR_policy", "low_level_FR_policy"]
else:
    from maze_envs.quantruped_centralizedController_environment import Quantruped_Centralized_Env as QuantrupedEnv
    experiment_id = "3_0_0"
    policies_to_train = ["central_policy"]
    policy_frozen = ["central_policy"]

# Init ray: First line on server, second for laptop
transfer_learning = True
num_workers = 2

if 'num_trials' in args and args.num_trials:
    num_trials = int(args.num_trials)
else:
    num_trials = 3


ray.init(num_cpus=num_trials*(num_workers+1), ignore_reinit_error=True)


config = ppo.DEFAULT_CONFIG.copy()

config['env'] = policy_scope
print("SELECTED ENVIRONMENT: ", policy_scope, " = ", QuantrupedEnv)

config['num_workers'] = num_workers
config['num_envs_per_worker'] = 4

# used grid_search([4000, 16000, 65536], didn't matter too much
config['train_batch_size'] = 16000

# Baseline Defaults:
config['gamma'] = 0.99
config['lambda'] = 0.95

# again used grid_search([0., 0.01]) for diff. values from lit.
config['entropy_coeff'] = 0.
config['clip_param'] = 0.2

config['vf_loss_coeff'] = 0.5
#config['vf_clip_param'] = 1

config['observation_filter'] = 'MeanStdFilter'

config['sgd_minibatch_size'] = 128  # 128
config['num_sgd_iter'] = 10
config['lr'] = 3e-5
config['grad_clip'] = 0.5

# Whether to rollout "complete_episodes" or "truncate_episodes".
#config["batch_mode"] = "complete_episodes"


config['model']['custom_model'] = "fc_glorot_uniform_init"
config['model']['fcnet_hiddens'] = [64, 64]
#config['model']['max_seq_len'] = 300

#config['seed'] = round(time.time())

# For running tune, we have to provide information on
# the multiagent which are part of the MultiEnvs
policies = QuantrupedEnv.return_policies()

config["multiagent"] = {
    "policies": policies,
    "policy_mapping_fn": QuantrupedEnv.policy_mapping_fn,
    # "policies_to_train": QuantrupedEnv.policy_names,  # , "dec_B_policy"],
    "policies_to_train": policies_to_train
}

# grid_search([0.5, 0.1, 0.05])  # 0.5
# grid_search([0.01, 0.001, 0.0001])
config['env_config']['ctrl_cost_weight'] = 0
config['env_config']['contact_cost_weight'] = 0  # 5e-2
config['env_config']['velocity_weight'] = 1
config['env_config']['direction_weight'] = 0  # 0.05
config['env_config']['target_reward'] = 1600  # 950
config['env_config']['frequency'] = 5


#config['env_config']['target_reward'] = grid_search([1000, 10000])
# Parameters for defining environment:
# Heightfield smoothness (between 0.6 and 1.0 are OK)
config['env_config']['hf_smoothness'] = 1.0
# Defining curriculum learning
config['env_config']['curriculum_learning'] = False
config['env_config']['range_smoothness'] = [1., 0.6]
config['env_config']['range_last_timestep'] = 10000000

# For curriculum learning: environment has to be updated every epoch
# added the callback class to solve callback warning


class MyCallbacks(DefaultCallbacks):

    # added  policies=worker.policy_map, in file ray/rllib/evaluation/sampler.py line 889. (like master github)

    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        assert episode.length == 0, \
            "ERROR: `on_episode_start()` callback should be called right " \
            "after env reset!"
        # print("episode {} (env-idx={}) started.".format(
        #    episode.episode_id, env_index))
        episode.user_data["velocity_reward"] = []
        episode.hist_data["velocity_reward"] = []
        #episode.user_data["direction_reward"] = []
        #episode.hist_data["direction_reward"] = []
        #episode.user_data["ctrl_reward"] = []
        #episode.hist_data["ctrl_reward"] = []
        #episode.user_data["contact_reward"] = []
        #episode.hist_data["contact_reward"] = []
        episode.user_data["cost_of_transport"] = 0

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        policies: Dict[str, Policy],
                        episode: MultiAgentEpisode, env_index: int, **kwargs):
        # Make sure this episode is ongoing.
        assert episode.length > 0, \
            "ERROR: `on_episode_step()` callback should not be called right " \
            "after env reset!"
        agents = episode.get_agents()
        # print("agents: ", agents)
        # print("info: ", episode.last_info_for(
        #    agents[-1]))
        if episode.last_info_for(
                agents[-1]):

            velocity_reward = episode.last_info_for(
                agents[-1])["velocity_reward"]
            # direction_reward = episode.last_info_for(
            #    agents[-1])["direction_reward"]
            ctrl_reward = episode.last_info_for(agents[-1])["ctrl_reward"]
            # contact_reward = episode.last_info_for(
            #    agents[-1])["contact_reward"]

            episode.user_data["velocity_reward"].append(velocity_reward)
            # episode.user_data["direction_reward"].append(direction_reward)
            # episode.user_data["ctrl_reward"].append(ctrl_reward)
            # episode.user_data["contact_reward"].append(contact_reward)

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        # Make sure this episode is really done.
        velocity_reward = np.mean(episode.user_data["velocity_reward"])
        #direction_reward = np.mean(episode.user_data["direction_reward"])
        #ctrl_reward = np.mean(episode.user_data["ctrl_reward"])
        #contact_reward = np.mean(episode.user_data["contact_reward"])

        # only whene episode ends
        agents = episode.get_agents()
        cost_of_transport = episode.last_info_for(
            agents[-1])["CoT"]

        # print("episode {} (env-idx={}) ended with length {} and velocity "
        #      "reward {}, angle reward {}, ctrl reward {}, contact reward {}.".format(episode.episode_id, env_index, episode.length,
        #                                                                              velocity_reward, direction_reward, ctrl_reward, contact_reward))

        episode.custom_metrics["velocity_reward"] = velocity_reward
        episode.hist_data["velocity_reward"] = episode.user_data["velocity_reward"]
        #episode.custom_metrics["direction_reward"] = direction_reward
        #episode.hist_data["direction_reward"] = episode.user_data["direction_reward"]
        #episode.custom_metrics["ctrl_reward"] = ctrl_reward
        #episode.hist_data["ctrl_reward"] = episode.user_data["ctrl_reward"]
        #episode.custom_metrics["contact_reward"] = contact_reward
        #episode.hist_data["contact_reward"] = episode.user_data["contact_reward"]
        episode.custom_metrics["cost_of_transport"] = cost_of_transport

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        timesteps_res = result["timesteps_total"]
        num_goal_reached = 0
        goals = trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(lambda env: env.return_GoalReached()))

        trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(lambda env: env.reset_GoalReached()))
        for i in goals:
            for j in i:
                num_goal_reached += j
        result["custom_metrics"]["goal_reached"] = num_goal_reached
        result["custom_metrics"]["goal_reached_accuracy"] = num_goal_reached / \
            result["episodes_this_iter"]


config["callbacks"] = MyCallbacks

if not transfer_learning:

    config["multiagent"] = {
        "policies": policies,
        "policy_mapping_fn": QuantrupedEnv.policy_mapping_fn,
        "policies_to_train": QuantrupedEnv.policy_names
    }

    analysis = tune.run(
        "PPO",
        name=(experiment_id +
              "_TwoRowMazeStatic_AllInfo_Sparse_Fq5_2Dense64LSTM64_NewReward_Env1600_Scratch_" + policy_scope),  # 2Dense64LSTM64
        num_samples=num_trials,
        checkpoint_at_end=True,
        checkpoint_freq=312,
        stop={"timesteps_total": 40000000},
        config=config,
    )
else:

    # load checkpoint

    # to not get ppo folders in ray_results

    def custom_log_creator(custom_path, custom_str):
        timestr = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logdir_prefix = "{}_{}".format(custom_str, timestr)

        def logger_creator(config):

            if not os.path.exists(custom_path):
                os.makedirs(custom_path)
            logdir = tempfile.mkdtemp(
                prefix=logdir_prefix, dir=custom_path)
            return UnifiedLogger(config, logdir, loggers=None)
        return logger_creator

    if 'config_checkpoint' in args and args.config_checkpoint:
        config_checkpoint = args.config_checkpoint
    else:
        config_checkpoint = "/home/nitro/clusteruni/ray_results/0_2_0_velocityCtrl_DifferentAngleNearTarget_Fq5_LS8_FF_HLCDirectionTarget_QuantrupedMultiEnv_Hierarchical/PPO_QuantrupedMultiEnv_Hierarchical_8da2d_00005_5_ctrl_cost_weight=0.1_2021-11-27_18-07-52/checkpoint_002500/checkpoint-2500"

    # to save it in the parameter if need to be used later on
    config['env_config']['config_checkpoint'] = config_checkpoint

    config_dir = os.path.dirname(config_checkpoint)
    config_path = os.path.join(config_dir, "params.pkl")

    # Loading configuration for checkpoint.
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")

    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            config_trained = pickle.load(f)

    config_trained['num_workers'] = 0
    config_trained['num_envs_per_worker'] = 1

    home = str(Path.home())

    # Create a new dummy Trainer to "fix" our checkpoint.
    preTrained_trainer = PPOTrainer(config=config_trained, env=config_trained["env"], logger_creator=custom_log_creator(
        os.path.expanduser(home+"/ray_results/Pretrained_PPO/tmp_PPO"), 'PPO_pretrained'))
    # Restore all policies from checkpoint.
    preTrained_trainer.restore(config_checkpoint)
    # Get trained weights for all policies.
    trained_weights = preTrained_trainer.get_weights()

    policies = QuantrupedEnv.return_policies()

    maze_trainer = PPOTrainer(config=config, env=policy_scope, logger_creator=custom_log_creator(
        os.path.expanduser(home+"/ray_results/Pretrained_PPO/tmp_PPO"), 'PPO_used'))
    # Set back all weights expect the high_level
    # trained weights.

    maze_trainer.set_weights({
        pid: w for pid, w in trained_weights.items()
        if (pid.startswith("low") or pid.startswith("policy") or pid.startswith("central"))
    })

    # copy the filters
    for policy_name in policy_frozen:
        maze_trainer.workers.local_worker(
        ).filters[policy_name] = preTrained_trainer.workers.local_worker().filters[policy_name]

    # Create the checkpoint from which tune can pick up the
    # experiment.

    checkpoint_path = home+"/ray_results/Pretrained_PPO/" + \
        config["env"]+"_"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+"/"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    new_checkpoint = maze_trainer.save(checkpoint_path)
    print("Checkpoint saved: ", new_checkpoint)

    preTrained_trainer.stop()
    maze_trainer.stop()

    if 'trial_name' in args and args.trial_name:
        trial_name = args.trial_name
    else:
        trial_name = "NoTrialSet"

    # Call tune and run (for evaluation: 10 seeds up to 20M steps; only centralized controller
    # required that much of time; decentralized controller should show very good results
    # after 5M steps.
    analysis = tune.run(
        "PPO",
        name=(experiment_id +
              "_TwoRowMazeStatic_AllInfo_Sparse_Fq5_2Dense64LSTM64_NewReward_"+str(trial_name)+"_VelocityCtrl_Env1600_" + policy_scope),
        num_samples=num_trials,
        checkpoint_at_end=True,
        checkpoint_freq=312,
        stop={"timesteps_total": 40000000},
        config=config,
        restore=new_checkpoint,
    )
