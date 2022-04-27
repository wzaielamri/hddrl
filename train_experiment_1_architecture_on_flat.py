######################
# Task: PreTraining Task Vanilla
######################

import argparse
import models
import simulation_envs
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

# Switch between different approaches.
parser = argparse.ArgumentParser()
parser.add_argument("--policy_scope", required=False)
parser.add_argument("--num_trials", required=False)

args = parser.parse_args()


experiment_id = "None"
if 'policy_scope' in args and args.policy_scope:
    policy_scope = args.policy_scope
else:
    policy_scope = 'QuantrupedMultiEnv_Centralized'

if policy_scope == "QuantrupedMultiEnv_FullyDecentral":
    from simulation_envs.quantruped_fourDecentralizedController_environments import QuantrupedFullyDecentralizedEnv as QuantrupedEnv
    experiment_id = "0_1_0"
elif policy_scope == "QuantrupedMultiEnv_Hierarchical":
    from simulation_envs.quantruped_hierarchicalController_environments import QuantrupedHierarchicalEnv as QuantrupedEnv
    experiment_id = "0_2_0"
elif policy_scope == "QuantrupedMultiEnv_HierarchicalFullyDecentralized":
    from simulation_envs.quantruped_hierarchicalController_environments import QuantrupedHierarchicalFullyDecentralizedEnv as QuantrupedEnv
    experiment_id = "0_3_0"
elif policy_scope == "QuantrupedMultiEnv_FullyDecentralizedHierarchical":
    from simulation_envs.quantruped_decentralizedHierarchicalController_environments import QuantrupedFullyDecentralizedHierarchicalEnv as QuantrupedEnv
    experiment_id = "0_4_0"
elif policy_scope == "QuantrupedMultiEnv_FullyDecentralizedHierarchicalFullyDecentralized":
    from simulation_envs.quantruped_decentralizedHierarchicalController_environments import QuantrupedFullyDecentralizedHierarchicalFullyDecentralizedEnv as QuantrupedEnv
    experiment_id = "0_5_0"
else:
    from simulation_envs.quantruped_centralizedController_environment import Quantruped_Centralized_Env as QuantrupedEnv
    experiment_id = "0_0_0"
# Init ray: First line on server, second for laptop

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
#config['num_gpus'] = 1


# used grid_search([4000, 16000, 65536], didn't matter too much
config['train_batch_size'] = 16000

# Baseline Defaults:
config['gamma'] = 0.99
config['lambda'] = 0.95

# again used grid_search([0., 0.01]) for diff. values from lit.
config['entropy_coeff'] = 0.
config['clip_param'] = 0.2

config['vf_loss_coeff'] = 0.5
#config['vf_clip_param'] = 4000.

config['observation_filter'] = 'MeanStdFilter'

config['sgd_minibatch_size'] = 128
config['num_sgd_iter'] = 10
config['lr'] = 3e-4
config['grad_clip'] = 0.5

config['model']['custom_model'] = "fc_glorot_uniform_init"
config['model']['fcnet_hiddens'] = [64, 64]

#config['seed'] = round(time.time())

# For running tune, we have to provide information on
# the multiagent which are part of the MultiEnvs
policies = QuantrupedEnv.return_policies()

config["multiagent"] = {
    "policies": policies,
    "policy_mapping_fn": QuantrupedEnv.policy_mapping_fn,
    "policies_to_train": QuantrupedEnv.policy_names,
}

# grid_search([0.5, 0.1, 0.05])  # 0.05
# grid_search([0.01, 0.001, 0.0001])
config['env_config']['ctrl_cost_weight'] = 0.05
config['env_config']['contact_cost_weight'] = 0
config['env_config']['velocity_weight'] = 1
config['env_config']['direction_weight'] = 0
config['env_config']['target_reward'] = 300  # 250
config['env_config']['target_weight'] = 0.02
config['env_config']['frequency'] = 5  # grid_search([5, 10])


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
        episode.user_data["ctrl_reward"] = []
        episode.hist_data["ctrl_reward"] = []
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
            episode.user_data["ctrl_reward"].append(ctrl_reward)
            # episode.user_data["contact_reward"].append(contact_reward)

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        # Make sure this episode is really done.
        velocity_reward = np.mean(episode.user_data["velocity_reward"])
        #direction_reward = np.mean(episode.user_data["direction_reward"])
        ctrl_reward = np.mean(episode.user_data["ctrl_reward"])
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
        episode.custom_metrics["ctrl_reward"] = ctrl_reward
        episode.hist_data["ctrl_reward"] = episode.user_data["ctrl_reward"]
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

# Call tune and run (for evaluation: 10 seeds up to 20M steps; only centralized controller
# required that much of time; decentralized controller should show very good results
# after 5M steps.
analysis = tune.run(
    "PPO",
    name=(experiment_id +
          "_velocityCtrl_DifferentAngleNearTarget_AllInfo_" + policy_scope),
    num_samples=num_trials,
    checkpoint_at_end=True,
    checkpoint_freq=312,
    stop={"timesteps_total": 40000000},
    config=config,
)
