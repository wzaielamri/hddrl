import collections
import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import mujoco_py
from gym import spaces
from mujoco_py import functions
import math
import random
import pickle5 as pickle
import os
from ray.tune.registry import get_trainable_cls
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.env import MultiAgentEnv
from maze_envs.maze_env_utils import plot_ray


class DefaultMapping(collections.defaultdict):
    """ Provides a default mapping.
    """

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


class QuantrupedMultiPoliciesHierarchicalEnv(MultiAgentEnv):
    """ RLLib multiagent Environment that encapsulates a quadruped walker environment.

        This is the parent class for rllib environments in which control can be
        distributed onto multiple agents.
        One simulation environment is spawned (a QuAntrupedMaze-v3) and this wrapper
        class organizes control and sensory signals.

        This parent class realizes still a hiearchical central approach which means that
        all sensory inputs are routed to the single, central control instance and
        all of the control signals of that instance are directly send towards the
        simulation through a hierarchical llc.

        Deriving classes have to overwrite basically four classes when distributing
        control to different controllers:
        - policy_mapping_fn: defines names of the distributed controllers
        - distribute_observations: how to distribute observations towards these controllers
        - concatenate_actions: how to integrate the control signals from the controllers
    """

    policy_names = ["centr_A_policy"]
    llc_policy_names = ["central_policy"]

    def __init__(self, config):
        if 'contact_cost_weight' in config.keys():
            contact_cost_weight = config['contact_cost_weight']
        else:
            contact_cost_weight = 5e-4

        if 'ctrl_cost_weight' in config.keys():
            ctrl_cost_weight = config['ctrl_cost_weight']
        else:
            ctrl_cost_weight = 0.5

        if 'hf_smoothness' in config.keys():
            hf_smoothness = config['hf_smoothness']
        else:
            hf_smoothness = 1.

        if 'velocity_weight' in config.keys():
            velocity_weight = config['velocity_weight']
        else:
            velocity_weight = 1.

        if 'direction_weight' in config.keys():
            direction_weight = config['direction_weight']
        else:
            direction_weight = 1.

        if 'target_reward' in config.keys():
            target_reward = config['target_reward']
        else:
            target_reward = 1600.
        if 'frequency' in config.keys():
            frequency = config['frequency']
        else:
            frequency = 5.

        self.env = gym.make("QuAntrupedMaze-v3",
                            ctrl_cost_weight=ctrl_cost_weight,
                            contact_cost_weight=contact_cost_weight, hf_smoothness=hf_smoothness)

        ant_mass = mujoco_py.functions.mj_getTotalmass(self.env.model)
        mujoco_py.functions.mj_setTotalmass(self.env.model, 10. * ant_mass)

        self.frequency = frequency
        self.contact_cost_weight = contact_cost_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.velocity_weight = velocity_weight
        self.direction_weight = direction_weight
        self.target_reward = target_reward
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.observation = None
        self.latent_space = None

        self.goalReachedCounter = 0
        self.env_step_counter = 0
        # For curriculum learning: scale smoothness of height field linearly over time
        # Set parameter
        if 'curriculum_learning' in config.keys():
            self.curriculum_learning = config['curriculum_learning']
        else:
            self.curriculum_learning = False
        if 'range_smoothness' in config.keys():
            self.curriculum_initial_smoothness = config['range_smoothness'][0]
            self.current_smoothness = self.curriculum_initial_smoothness
            self.curriculum_target_smoothness = config['range_smoothness'][1]
        if 'range_last_timestep' in config.keys():
            self.curriculum_last_timestep = config['range_last_timestep']

    def printMetrice(self,):
        print("debug counter: ", self.goalReachedCounter)

    def return_GoalReached(self,):
        return self.goalReachedCounter

    def reset_GoalReached(self,):
        self.goalReachedCounter = 0

    def update_environment_after_epoch(self, timesteps_total):
        """
            Called after each training epoch.
            Can be used to set a curriculum during learning.
        """
        if self.curriculum_learning:
            if self.curriculum_last_timestep > timesteps_total:
                # Two different variants:
                # First one is simply decreasing smoothness while in curriculum interval.
                # self.current_smoothness = self.curriculum_initial_smoothness - (self.curriculum_initial_smoothness - self.curriculum_target_smoothness) * (timesteps_total/self.curriculum_last_timestep)
                # Second one is selecting randomly a smoothness, chosen from an interval
                # from flat (1.) towards the decreased minimum smoothness
                self.current_smoothness = self.curriculum_initial_smoothness - np.random.rand()*(self.curriculum_initial_smoothness -
                                                                                                 self.curriculum_target_smoothness) * (timesteps_total/self.curriculum_last_timestep)
            else:
                # Two different variants:
                # First one is simply decreasing smoothness while in curriculum interval.
                # self.curriculum_learning = False
                # self.current_smoothness = self.curriculum_target_smoothness
                # Second one is selecting randomly a smoothness, chosen from an interval
                # from flat (1.) towards the decreased minimum smoothness
                self.current_smoothness = self.curriculum_target_smoothness + \
                    np.random.rand()*(self.curriculum_initial_smoothness -
                                      self.curriculum_target_smoothness)
            self.env.set_hf_parameter(self.current_smoothness)
        self.env.create_new_random_hfield()
        self.env.reset()
        # self.printMetrice()

    def distribute_observations(self, obs_full):
        """ Distribute observations in the multi agent environment.
        """
        return {
            self.policy_names[0]: obs_full,
        }

    def distribute_contact_cost(self):
        """ Calculate sum of contact costs.
        """
        contact_cost = {}
        raw_contact_forces = self.env.sim.data.cfrc_ext[0:14]
        contact_forces = np.clip(raw_contact_forces, -1., 1.)
        contact_cost[self.policy_names[0]] = self.env.contact_cost_weight * np.sum(
            np.square(contact_forces))
        return contact_cost

    def distribute_reward(self, reward_full, info, action_dict, obs_full, done_w):
        """ Describe how to distribute reward.
        """
        x_velocity = info["x_velocity"]
        y_velocity = info["y_velocity"]

        velocity = np.sqrt(x_velocity**2+y_velocity**2)

        # avoid zero division
        try:
            velocity_angle = math.atan2(y_velocity, x_velocity)
        except:
            velocity_angle = math.atan2(y_velocity, x_velocity+1e-12)

        # avoid zero division
        try:
            direction_target = (math.atan2(
                self.env.target_y-obs_full[1], self.env.target_x-obs_full[0]))
        except:
            direction_target = (math.atan2(
                self.env.target_y-obs_full[1], (self.env.target_x-obs_full[0])+1e-12))

        velocity_angle -= direction_target

        velocity_reward = velocity * math.cos(velocity_angle)

        direction_reward = - abs(obs_full[-1])

        fw_reward = self.velocity_weight * velocity_reward + \
            self.direction_weight * direction_reward

        rew = {}
        contact_costs = self.distribute_contact_cost()

        # sparce reward
        # add reward when reaching goal
        distance_rob_target = np.sqrt(
            ((obs_full[0]-self.env.target_x)**2)+((obs_full[1]-self.env.target_y)**2))

        if (distance_rob_target < 0.25):
            fw_reward += 0.2*(self.target_reward - self.env_step_counter)
        sum_act = 0
        sum_contact = 0
        for policy_name in self.policy_names:
            rew[policy_name] = fw_reward / len(self.policy_names) \
                - self.env.ctrl_cost_weight * np.sum(np.square(action_dict[policy_name])) \
                - contact_costs[policy_name]
            sum_act += np.sum(np.square(action_dict[policy_name]))
            sum_contact += contact_costs[policy_name]

        # save rewards for later metrices:
        info["velocity_reward"] = velocity_reward
        info["direction_reward"] = - direction_reward
        info["ctrl_reward"] = sum_act
        if self.contact_cost_weight != 0:
            info["contact_reward"] = sum_contact/self.contact_cost_weight
        else:
            info["contact_reward"] = sum_contact
        return rew, info

    def concatenate_actions(self, action_dict):
        """ Collect actions from all agents and combine them for the single
            call of the environment.
        """
        return action_dict[self.policy_names[0]]  # np.concatenate( (action_dict[self.policy_A],

    def reset(self):
        obs_original = self.env.reset()
        self.power_total = 0.0
        self.cost_of_transport = 0.0
        self.start_pos = obs_original[:2]

        self.env_step_counter = 0
        return self.distribute_observations(obs_original)

    def step(self, action_dict):
        # Stepping the environment.

        # Use with mujoco 2.
        # functions.mj_rnePostConstraint(self.env.model, self.env.data)

        # Combine actions from all agents and step environment.
        obs_full, rew_w, done_w, info_w = self.env.step(self.concatenate_actions(
            action_dict))  # self.env.step( np.concatenate( (action_dict[self.policy_A],
        # action_dict[self.policy_B]) ))
        self.env_step_counter += 1
        # Distribute observations and rewards to the individual agents.
        obs_dict = self.distribute_observations(obs_full)
        rew_dict, info_w = self.distribute_reward(
            rew_w, info_w, action_dict, obs_full, done_w)

        done = {
            "__all__": done_w,
        }

        # self.acc_forw_rew += info_w['reward_forward']
        # self.acc_ctrl_cost += info_w['reward_ctrl']
        # self.acc_contact_cost += info_w['reward_contact']
        # self.acc_step +=1
        # print("REWARDS: ", info_w['reward_forward'], " / ", self.acc_forw_rew/self.acc_step, "; ",
        #   info_w['reward_ctrl'], " / ", self.acc_ctrl_cost/(self.acc_step*self.env.ctrl_cost_weight), "; ",
        #  info_w['reward_contact'], " / ", self.acc_contact_cost/(self.acc_step*self.env.contact_cost_weight), self.env.contact_cost_weight)
        # self._elapsed_steps += 1
        # if self._elapsed_steps >= self._max_episode_steps:
        #   info_w['TimeLimit.truncated'] = not done
        #  done["__all__"] = True

        # check if the robot reached the target :
        distance_rob_target = np.sqrt(
            ((obs_full[0]-self.env.target_x)**2)+((obs_full[1]-self.env.target_y)**2))

        if (self.env.is_in_first_goal((obs_full[0], obs_full[1]))):
            for policy in list(action_dict.keys()):
                rew_dict[policy] = 1/len(list(action_dict.keys()))
        elif (self.env.is_in_second_goal((obs_full[0], obs_full[1]))):
            for policy in list(action_dict.keys()):
                rew_dict[policy] = 1/len(list(action_dict.keys()))
        else:
            for policy in list(action_dict.keys()):
                rew_dict[policy] = 0

        if (self.env.is_in_goal((obs_full[0], obs_full[1]))):
            done["__all__"] = True
            self.goalReachedCounter += 1
            # for central and decentral:
            for policy in list(action_dict.keys()):
                rew_dict[policy] = 1/len(list(action_dict.keys()))

        if done["__all__"]:
            distance_x = np.sqrt(
                np.sum((self.env.sim.data.qpos[0:2] - self.start_pos)**2))
            com_vel = distance_x/self.env_step_counter

            # mujoco_py.functions.mj_getTotalmass(env.env.model) is equal to the robot mass and the target in the env or even the maze because their geo is in the body of the robot
            # the weight is directly set to: 8.78710174560547

            weight_rob = 8.78710174560547
            self.cost_of_transport = (
                self.power_total/self.env_step_counter) / (weight_rob * com_vel)
            info_w["CoT"] = self.cost_of_transport

        # info custom metrices
        info_custom = {}
        # only the first policy deliver the custom matrices
        info_custom[self.policy_names[-1]] = info_w
        return obs_dict, rew_dict, done, info_custom

    def render(self):
        self.env.render()

    @ staticmethod
    def policy_mapping_fn(agent_id):
        # Each derived class has to define all agents by name.
        return QuantrupedMultiPoliciesHierarchicalEnv.policy_names[0]

    @ staticmethod
    def return_policies():
        # For each agent the policy interface hapolicy_namess to be defined.
        obs_space = spaces.Box(-np.inf, np.inf, (46,), np.float64)
        policies = {
            QuantrupedMultiPoliciesHierarchicalEnv.policy_names[0]: (None,
                                                                     obs_space, spaces.Box(np.array([-1., -1., -1., -1., -1., -1., -1., -1.]), np.array([+1., +1., +1., +1., +1., +1., +1., +1.])), {"model": {"custom_model": "fc_glorot_uniform_init_lstm", 'fcnet_hiddens': [64, 64, ], "lstm_cell_size": 64}}),
        }
        return policies
