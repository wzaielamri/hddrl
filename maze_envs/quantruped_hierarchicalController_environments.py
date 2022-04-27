import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import mujoco_py
from gym import spaces
from ray.tune import grid_search
import math
from maze_envs import QuantrupedMultiPoliciesHierarchicalEnv

import matplotlib.pyplot as plt
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
import os
import pickle5 as pickle
from ray.tune.registry import get_trainable_cls
import collections
from scipy.spatial.transform import Rotation
from maze_envs.maze_env_utils import plot_ray
from datetime import date
import tempfile
from ray.tune.logger import UnifiedLogger


class DefaultMapping(collections.defaultdict):
    """ Provides a default mapping.
    """

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


class QuantrupedHierarchicalControllerSuperEnv(QuantrupedMultiPoliciesHierarchicalEnv):
    """ Derived environment for control of the four-legged agent.
        Allows to instantiate hierarchical agents for control.

        Super class for all hierarchical controller
        into two different, control units (policies) 
        each instantiated as a single agent. 

        Class defines 
        - policy_mapping_fn: defines names of the distributed controllers
            Is defined in derived classes and differs between the different architectures.
    """

    @staticmethod
    def policy_mapping_fn(agent_id):
        if agent_id.startswith("low_level_"):
            return "low_level_policy"
        else:
            return "high_level_policy"


class QuantrupedHierarchicalEnv(QuantrupedHierarchicalControllerSuperEnv):
    """ Derived environment for control of the four-legged agent.
        Uses two different, control units (policies) 
        each instantiated as a single agent. 

        Input scope of each controller: give goal and act.

        Class defines 
        - policy_mapping_fn: defines names of the distributed controllers
        - distribute_observations: how to distribute observations towards these controllers
            Is defined in the obs_indices for each leg.
    """

    # This is ordering of the policies as applied here:
    policy_names = ["high_level_policy", "low_level_policy"]

    def __init__(self, config):
        self.obs_indices = {}
        # First global information:
        # exclude_current_positions_from_observation=False
        # 0-2: x, y, z, 3-6: quaternion orientation torso
        # 7: hip FL angle, 8: knee FL angle
        # 9: hip HL angle, 10: knee HL angle
        # 11: hip HR angle, 12: knee HR angle
        # 13: hip FR angle, 14: knee FR angle
        # Velocities follow same ordering, but have in addition x and y vel.
        # 15-17: vel, 18-20: rotational velocity
        # 21: hip FL angle, 22: knee FL angle
        # 23: hip HL angle, 24: knee HL angle
        # 25: hip HR angle, 26: knee HR angle
        # 27: hip FR angle, 28: knee FR angle
        # Passive forces same ordering, only local information used
        # 29: hip FL angle, 30: knee FL angle
        # 31: hip HL angle, 32: knee HL angle
        # 33: hip HR angle, 34: knee HR angle
        # 35: hip FR angle, 36: knee FR angle
        # Last: control signals (clipped) from last time step
        # Unfortunately, different ordering (as the action spaces...)
        # 39: hip FL angle, 40: knee FL angle
        # 41: hip HL angle, 42: knee HL angle
        # 43: hip HR angle, 44: knee HR angle
        # 37: hip FR angle, 38: knee FR angle
        # The goal vector
        # 45-46:, direction_head_to_target, robot north pole
        # The central policy gets all observations
        #0, 1, 2, 3, 4, 5, 6, 15, 16, 17, 18, 19, 20,
        self.obs_indices["high_level"] = [
            0, 1, 2, 3, 4, 5, 6, 15, 16, 17, 18, 19, 20, 45, ]

        self.obs_indices["low_level"] = [7, 8, 9, 10, 11, 12, 13, 14,
                                         21, 22, 23, 24, 25, 26, 27, 28,
                                         29, 30, 31, 32, 33, 34, 35, 36,
                                         37, 38, 39, 40, 41, 42, 43, 44]  # range(0, 45)

        super().__init__(config)

    @staticmethod
    def return_policies():
        latent_space_len = 8
        # For each agent the policy interface has to be defined.
        obs_space = spaces.Box(-np.inf, np.inf, (14,), np.float64)
        LL_obs_space = spaces.Box(-np.inf, np.inf, (32,), np.float64)
        # Heess Architecture
        """
        policies = {
            QuantrupedHierarchicalEnv.policy_names[0]: (None,
                                                        obs_space, spaces.Box(np.array([-1., ]*latent_space_len), np.array([+1., ]*latent_space_len)), {"model": {"custom_model": "hlc_glorot_uniform_init_lstm", 'fcnet_hiddens': [30, 40, 100], "lstm_cell_size": 10, }}),
            # QuantrupedHierarchicalEnv.policy_names[1]: (None,spaces.Tuple([LL_obs_space, spaces.Box(np.array([-1., ]*latent_space_len), np.array([+1., ]*latent_space_len))]), spaces.Box(np.array([-1., -1., -1., -1., -1., -1., -1., -1.]), np.array([+1., +1., +1., +1., +1., +1., +1., +1.])), {"model": {"custom_model": "llc_glorot_uniform_init", 'fcnet_hiddens': [150, 150, 150], }}),
            # Check the pretrained LLC
            QuantrupedHierarchicalEnv.policy_names[1]: (None, spaces.Tuple([LL_obs_space, spaces.Box(np.array([-1, ]*(latent_space_len)), np.array([+1, ]*(latent_space_len)))]), spaces.Box(np.array([-1., -1., -1., -1., -1., -1., -1., -1.]), np.array([+1., +1., +1., +1., +1., +1., +1., +1.])), {"model": {"custom_model": "fc_glorot_uniform_init", 'fcnet_hiddens': [64, 64]}}),
        }
        """
        policies = {
            QuantrupedHierarchicalEnv.policy_names[0]: (None,
                                                        obs_space, spaces.Box(np.array([-1, ]*latent_space_len), np.array([+1, ]*latent_space_len)), {"model": {"custom_model": "fc_glorot_uniform_init_lstm", 'fcnet_hiddens': [64, 64, ], "lstm_cell_size": 64}}),  # "lstm_cell_size": 64
            QuantrupedHierarchicalEnv.policy_names[1]: (None, spaces.Tuple([LL_obs_space, spaces.Box(np.array([-1, ]*(latent_space_len)), np.array([+1, ]*(latent_space_len)))]), spaces.Box(np.array([-1., -1., -1., -1., -1., -1., -1., -1.]), np.array([+1., +1., +1., +1., +1., +1., +1., +1.])), {"model": {"custom_model": "fc_glorot_uniform_init", 'fcnet_hiddens': [64, 64]}}),
        }

        return policies

    def distribute_contact_cost(self):
        """ Calculate sum of contact costs.
        """
        contact_cost = {}
        raw_contact_forces = self.env.sim.data.cfrc_ext[0:14]
        contact_forces = np.clip(raw_contact_forces, -1., 1.)
        contact_cost[self.policy_names[1]] = self.env.contact_cost_weight * np.sum(
            np.square(contact_forces))
        return contact_cost

    def distribute_reward(self, info, action_dict, obs_full):
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

        rew = fw_reward \
            - self.env.ctrl_cost_weight * np.sum(np.square(action_dict)) \
            - contact_costs[self.policy_names[1]]

        # calculate the ctr and contact costs
        sum_act = np.sum(np.square(action_dict))
        sum_contact = contact_costs[self.policy_names[1]]

        # save rewards for later metrices:
        info["velocity_reward"] = velocity_reward
        info["direction_reward"] = - direction_reward
        info["ctrl_reward"] = sum_act
        if self.contact_cost_weight != 0:
            info["contact_reward"] = sum_contact/self.contact_cost_weight
        else:
            info["contact_reward"] = sum_contact

        return rew, info

    def reset(self):
        self.power_total = 0.0
        self.cost_of_transport = 0.0

        self.latentSpace = np.zeros(8)
        self.cur_obs = self.env.reset()
        self.start_pos_high = self.cur_obs[:2]
        self.start_pos = self.cur_obs[:2]
        self.steps_remaining_at_level = 0
        self.env_step_counter = 0

        # w,x,y,z are pu as x,y,z,w
        quat_north_robot = np.concatenate(
            (self.cur_obs[4:7], [self.cur_obs[3]]))
        north_robot = Rotation.from_quat(quat_north_robot).as_euler("xyz")[2]
        maze_obs = self.env.get_current_maze_obs(
            self.cur_obs[0], self.cur_obs[1], north_robot)
        obs_hlc = np.concatenate(
            [maze_obs, self.cur_obs[self.obs_indices["high_level"]]])
        obs = {}
        # self.cur_obs[self.obs_indices["high_level"]]
        obs["high_level_policy"] = self.cur_obs[self.obs_indices["high_level"]]
        return obs

    def step(self, action_dict):
        #print("action_dict: ", action_dict)
        #assert len(action_dict) == 1, action_dict
        if "high_level_policy" in action_dict:
            return self._high_level_step(action_dict["high_level_policy"])
        else:
            return self._low_level_step(list(action_dict.values())[0])

    def _high_level_step(self, action):
        self.latentSpace = action
        # each 10 steps HLC executes an action
        self.steps_remaining_at_level = self.frequency

        self.start_pos_high = self.cur_obs[:2]

        # without the last goal informations
        obs = {"low_level_policy": [
            self.cur_obs[self.obs_indices["low_level"]], self.latentSpace]}

        rew = {"low_level_policy": 0}
        done = {"__all__": False}

        return obs, rew, done, {}

    def _low_level_step(self, action):
        # Stepping the environment.
        self.steps_remaining_at_level -= 1
        # Use with mujoco 2.
        #functions.mj_rnePostConstraint(self.env.model, self.env.data)

        self.env_step_counter += 1

        # Combine actions from all agents and step environment.
        obs_full, rew_w, done_w, info_w = self.env.step(
            action)

        current_power = np.sum(
            np.abs(np.roll(self.env.sim.data.ctrl, -2) * self.env.sim.data.qvel[6:]))
        self.power_total += current_power

        # Reward for only low level:
        rew_val, info_w = self.distribute_reward(info_w, action, obs_full)

        self.cur_obs = obs_full

        # save rew and obs of low agent for later run

        obs = {"low_level_policy": [
            self.cur_obs[self.obs_indices["low_level"]], self.latentSpace]}
        rew = {"low_level_policy": 0}  # rew_val

        # Handle env termination & transitions back to higher level
        done = {"__all__": False}

        # check if the robot reached the target :
        distance_rob_target = np.sqrt(
            ((obs_full[0]-self.env.target_x)**2)+((obs_full[1]-self.env.target_y)**2))

        """
        # if the agent gets upside-down stop the episode
        rot = Rotation.from_quat(np.concatenate(
            (obs_full[4:7], [obs_full[3]])))
        if rot.as_matrix()[2][2] < 0:
            done_w = True
        """

        # done without reaching the goal (reaching is after)
        if done_w:
            done["__all__"] = True
            quat_north_robot = np.concatenate(
                (obs_full[4:7], [obs_full[3]]))
            north_robot = Rotation.from_quat(
                quat_north_robot).as_euler("xyz")[2]

            maze_obs = self.env.get_current_maze_obs(
                obs_full[0], obs_full[1], north_robot)
            obs_hlc = np.concatenate(
                [maze_obs, self.cur_obs[self.obs_indices["high_level"]]])

            rew["high_level_policy"] = 0  # -10
            # obs_hlc
            # self.cur_obs[self.obs_indices["high_level"]]
            # self.cur_obs[self.obs_indices["high_level"]]
            obs["high_level_policy"] = self.cur_obs[self.obs_indices["high_level"]]

            distance_x = np.sqrt(
                np.sum((self.env.sim.data.qpos[0:2] - self.start_pos)**2))
            com_vel = distance_x/self.env_step_counter

            # mujoco_py.functions.mj_getTotalmass(env.env.model) is equal to the robot mass and the target in the env or even the maze because their geo is in the body of the robot
            # the weight is directly set to: 8.78710174560547

            weight_rob = 8.78710174560547
            self.cost_of_transport = (
                self.power_total/self.env_step_counter) / (weight_rob * com_vel)
            info_w["CoT"] = self.cost_of_transport

            #######
            """
            old_distance_rob_target = np.sqrt(
                ((self.start_pos_high[0]-self.env.target_x)**2)+((self.start_pos_high[1]-self.env.target_y)**2))
            b = np.array([self.env.target_x, self.env.target_y])
            ba = self.start_pos_high - b
            bc = obs_full[:2] - b

            try:  # if 0 in  the denominator
                cosine_angle = np.dot(ba, bc) / \
                    (np.linalg.norm(ba) * np.linalg.norm(bc))
            except:
                cosine_angle = 0
            dist_to_go = cosine_angle * distance_rob_target

            rew["high_level_policy"] = old_distance_rob_target - dist_to_go
            """
            ######

        elif self.steps_remaining_at_level == 0:
            quat_north_robot = np.concatenate(
                (obs_full[4:7], [obs_full[3]]))
            north_robot = Rotation.from_quat(
                quat_north_robot).as_euler("xyz")[2]

            maze_obs = self.env.get_current_maze_obs(
                obs_full[0], obs_full[1], north_robot)
            obs_hlc = np.concatenate(
                [maze_obs, self.cur_obs[self.obs_indices["high_level"]]])

            rew["high_level_policy"] = 0
            # obs_hlc
            # self.cur_obs[self.obs_indices["high_level"]]
            # self.cur_obs[self.obs_indices["high_level"]]
            obs["high_level_policy"] = self.cur_obs[self.obs_indices["high_level"]]

            ######
            """
            # for distance
            old_distance_rob_target = np.sqrt(
                ((self.start_pos_high[0]-self.env.target_x)**2)+((self.start_pos_high[1]-self.env.target_y)**2))
            b = np.array([self.env.target_x, self.env.target_y])
            ba = self.start_pos_high - b
            bc = obs_full[:2] - b

            try:  # if 0 in  the denominator
                cosine_angle = np.dot(ba, bc) / \
                    (np.linalg.norm(ba) * np.linalg.norm(bc))
            except:
                cosine_angle = 0
            dist_to_go = cosine_angle * distance_rob_target

            rew["high_level_policy"] = old_distance_rob_target - dist_to_go
            """
            ######

        # actually never reached
        """
        if (self.env.is_in_collision([obs_full[0], obs_full[1]])):

            done["__all__"] = True
            quat_north_robot = np.concatenate(
                (obs_full[4:7], [obs_full[3]]))
            north_robot = Rotation.from_quat(
                quat_north_robot).as_euler("xyz")[2]

            maze_obs = self.env.get_current_maze_obs(
                obs_full[0], obs_full[1], north_robot)
            obs_hlc = np.concatenate(
                [maze_obs, self.cur_obs[self.obs_indices["high_level"]]])

            rew["high_level_policy"] = -50
            obs["high_level_policy"] = obs_hlc
            
            # render
            img = self.env.env.sim.render(
                width=1280, height=800, camera_name="top_fixed")

            plt.imsave("render/renderCollision_" + str(self.env_step_counter).zfill(4) +
                       '.png', img, origin='lower')

            plot_ray(reading=maze_obs, structure=self.env.maze_structure, size_scaling=self.env.maze_size_scaling, robot_xy=np.array([
                obs_full[0], obs_full[1]]), ori=north_robot, sensor_range=self.env.sensor_range, sensor_span=self.env.sensor_span, n_bins=self.env.n_bins, xy_goal=np.array([self.env.target_x, self.env.target_y]), plot_file="plot/plotCollision_" + str(self.env_step_counter).zfill(4) + '.png')
            print("collision")
        """

        if (self.env.is_in_first_goal((obs_full[0], obs_full[1]))):
            rew["high_level_policy"] = 1
            obs["high_level_policy"] = self.cur_obs[self.obs_indices["high_level"]]

        if (self.env.is_in_second_goal((obs_full[0], obs_full[1]))):
            rew["high_level_policy"] = 1
            obs["high_level_policy"] = self.cur_obs[self.obs_indices["high_level"]]

        if (self.env.is_in_goal((obs_full[0], obs_full[1]))):

            done["__all__"] = True
            # reset for the next target

            quat_north_robot = np.concatenate(
                (obs_full[4:7], [obs_full[3]]))
            north_robot = Rotation.from_quat(
                quat_north_robot).as_euler("xyz")[2]

            maze_obs = self.env.get_current_maze_obs(
                obs_full[0], obs_full[1], north_robot)
            obs_hlc = np.concatenate(
                [maze_obs, self.cur_obs[self.obs_indices["high_level"]]])

            self.goalReachedCounter += 1
            # 0.05 * (self.target_reward-self.env_step_counter)
            # 0.01 *  (self.target_reward-self.env_step_counter)
            rew["high_level_policy"] = 1#7*(1 -(self.env_step_counter/self.target_reward))
            # obs_hlc
            # self.cur_obs[self.obs_indices["high_level"]]
            # self.cur_obs[self.obs_indices["high_level"]]
            obs["high_level_policy"] = self.cur_obs[self.obs_indices["high_level"]]

            """
            # render
            img = self.env.env.sim.render(
                width=1280, height=800, camera_name="top_fixed")
            plt.imsave("render/renderReached_" + str(self.env_step_counter).zfill(4) +
                       '.png', img, origin='lower')

            plot_ray(reading=maze_obs, structure=self.env.maze_structure, size_scaling=self.env.maze_size_scaling, robot_xy=np.array([
                obs_full[0], obs_full[1]]), ori=north_robot, sensor_range=self.env.sensor_range, sensor_span=self.env.sensor_span, n_bins=self.env.n_bins, xy_goal=np.array([self.env.target_x, self.env.target_y]), plot_file="plot/plotReached_" + str(self.env_step_counter).zfill(4) + '.png')
            print("reached")
            """

            distance_x = np.sqrt(
                np.sum((self.env.sim.data.qpos[0:2] - self.start_pos)**2))
            com_vel = distance_x/self.env_step_counter

            # mujoco_py.functions.mj_getTotalmass(env.env.model) is equal to the robot mass and the target in the env or even the maze because their geo is in the body of the robot
            # the weight is directly set to: 8.78710174560547

            weight_rob = 8.78710174560547
            self.cost_of_transport = (
                self.power_total/self.env_step_counter) / (weight_rob * com_vel)

            info_w["CoT"] = self.cost_of_transport

        info = {"low_level_policy": info_w}

        return obs, rew, done, info


class QuantrupedHierarchicalFullyDecentralizedEnv(QuantrupedHierarchicalControllerSuperEnv):
    """ Derived environment for control of the four-legged agent.
        Uses 5 different, control units (policies) : 1 HLC and 4 LLC
        each instantiated as a single agent. 

        Input scope of each controller: give goal and act.

        Class defines 
        - policy_mapping_fn: defines names of the distributed controllers
        - distribute_observations: how to distribute observations towards these controllers
            Is defined in the obs_indices for each leg.
    """

    # This is ordering of the policies as applied here:
    policy_names = ["high_level_policy", "low_level_FL_policy",
                    "low_level_HL_policy", "low_level_HR_policy", "low_level_FR_policy"]

    @staticmethod
    def policy_mapping_fn(agent_id):
        if agent_id.startswith("low_level_FL"):
            return "low_level_FL_policy"
        elif agent_id.startswith("low_level_HL"):
            return "low_level_HL_policy"
        elif agent_id.startswith("low_level_HR"):
            return "low_level_HR_policy"
        elif agent_id.startswith("low_level_FR"):
            return "low_level_FR_policy"
        else:
            return "high_level_policy"

    def __init__(self, config):
        self.obs_indices = {}
        # First global information:
        # exclude_current_positions_from_observation=False
        # the positions are calculated in the stat episode position coordinate system (goal is always in 0-0)
        # 0-2: x, y, z, 3-6: quaternion orientation torso
        # 7: hip FL angle, 8: knee FL angle
        # 9: hip HL angle, 10: knee HL angle
        # 11: hip HR angle, 12: knee HR angle
        # 13: hip FR angle, 14: knee FR angle
        # Velocities follow same ordering, but have in addition x and y vel.
        # 15-17: vel, 18-20: rotational velocity
        # 21: hip FL angle, 22: knee FL angle
        # 23: hip HL angle, 24: knee HL angle
        # 25: hip HR angle, 26: knee HR angle
        # 27: hip FR angle, 28: knee FR angle
        # Passive forces same ordering, only local information used
        # 29: hip FL angle, 30: knee FL angle
        # 31: hip HL angle, 32: knee HL angle
        # 33: hip HR angle, 34: knee HR angle
        # 35: hip FR angle, 36: knee FR angle
        # Last: control signals (clipped) from last time step
        # Unfortunately, different ordering (as the action spaces...)
        # 39: hip FL angle, 40: knee FL angle
        # 41: hip HL angle, 42: knee HL angle
        # 43: hip HR angle, 44: knee HR angle
        # 37: hip FR angle, 38: knee FR angle
        # The goal vector
        # 45-46: distance, direction_head_to_target
        # The central policy gets all observations

        self.obs_indices["high_level"] = [
            0, 1, 2, 3, 4, 5, 6, 15, 16, 17, 18, 19, 20, 45, ]

        self.obs_indices["low_level_FL_policy"] = [
            7, 8, 21, 22, 29, 30, 39, 40]
        self.obs_indices["low_level_HL_policy"] = [
            9, 10, 23, 24, 31, 32, 41, 42]
        self.obs_indices["low_level_HR_policy"] = [
            11, 12, 25, 26, 33, 34, 43, 44]
        self.obs_indices["low_level_FR_policy"] = [
            13, 14, 27, 28, 35, 36, 37, 38]
        super().__init__(config)

    @staticmethod
    def return_policies():
        # For each agent the policy interface has to be defined.

        latent_space_len = 8

        obs_space = spaces.Box(-np.inf, np.inf, (14,), np.float64)
        LL_obs_space = spaces.Box(-np.inf, np.inf, (8,), np.float64)
        policies = {
            QuantrupedHierarchicalFullyDecentralizedEnv.policy_names[0]: (None,
                                                                          obs_space, spaces.Box(np.array([-1, ]*latent_space_len), np.array([+1, ]*latent_space_len)), {"model": {"custom_model": "fc_glorot_uniform_init_lstm", 'fcnet_hiddens': [64, 64, ], "lstm_cell_size": 64}}),
            QuantrupedHierarchicalFullyDecentralizedEnv.policy_names[1]: (None,
                                                                          spaces.Tuple([LL_obs_space, spaces.Box(np.array([-1, ]*2), np.array([+1, ]*2))]), spaces.Box(np.array([-1., -1.]), np.array([+1., +1.])), {"model": {"custom_model": "fc_glorot_uniform_init", 'fcnet_hiddens': [64, 64], }}),
            QuantrupedHierarchicalFullyDecentralizedEnv.policy_names[2]: (None,
                                                                          spaces.Tuple([LL_obs_space, spaces.Box(np.array([-1, ]*2), np.array([+1, ]*2))]), spaces.Box(np.array([-1., -1.]), np.array([+1., +1.])), {"model": {"custom_model": "fc_glorot_uniform_init", 'fcnet_hiddens': [64, 64], }}),
            QuantrupedHierarchicalFullyDecentralizedEnv.policy_names[3]: (None,
                                                                          spaces.Tuple([LL_obs_space, spaces.Box(np.array([-1, ]*2), np.array([+1, ]*2))]), spaces.Box(np.array([-1., -1.]), np.array([+1., +1.])), {"model": {"custom_model": "fc_glorot_uniform_init", 'fcnet_hiddens': [64, 64], }}),
            QuantrupedHierarchicalFullyDecentralizedEnv.policy_names[4]: (None,
                                                                          spaces.Tuple([LL_obs_space, spaces.Box(np.array([-1, ]*2), np.array([+1, ]*2))]), spaces.Box(np.array([-1., -1.]), np.array([+1., +1.])), {"model": {"custom_model": "fc_glorot_uniform_init", 'fcnet_hiddens': [64, 64], }}),
        }
        return policies

    def concatenate_actions(self, action_dict):
        # Return actions in the (DIFFERENT in Mujoco) order FR - FL - HL - HR
        actions = np.concatenate((action_dict[self.policy_names[4]],
                                  action_dict[self.policy_names[1]],
                                  action_dict[self.policy_names[2]],
                                  action_dict[self.policy_names[3]]))
        return actions

    def distribute_reward(self, info, action_dict, obs_full):
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
        for policy in list(action_dict.keys()):
            rew[policy] = fw_reward / len(list(action_dict.keys())) \
                - self.env.ctrl_cost_weight * np.sum(np.square(action_dict[policy])) \
                - contact_costs[policy]
            sum_act += np.sum(np.square(action_dict[policy]))
            sum_contact = contact_costs[policy]
        # save rewards for later metrices:
        info["velocity_reward"] = velocity_reward
        info["direction_reward"] = -  direction_reward
        info["ctrl_reward"] = sum_act
        info["contact_reward"] = (sum_contact)

        return rew, info

    # Distribute the contact costs into the decentralized policies.
    # In the trivial case, only a single policy is used that gets all information.

    def distribute_contact_cost(self):
        contact_cost = {}
        #from mujoco_py import functions
        #functions.mj_rnePostConstraint(self.env.model, self.env.data)
        raw_contact_forces = self.env.sim.data.cfrc_ext
        contact_forces = np.clip(raw_contact_forces, -1., 1.)
        contact_costs = self.env.contact_cost_weight * \
            np.square(contact_forces)
        global_contact_costs = np.sum(contact_costs[0:2])/4.
        contact_cost[self.policy_names[1]
                     ] = global_contact_costs + np.sum(contact_costs[2:5])
        contact_cost[self.policy_names[2]
                     ] = global_contact_costs + np.sum(contact_costs[5:8])
        contact_cost[self.policy_names[3]] = global_contact_costs + \
            np.sum(contact_costs[8:11])
        contact_cost[self.policy_names[4]
                     ] = global_contact_costs + np.sum(contact_costs[11:14])
        return contact_cost

    def reset(self):
        self.power_total = 0.0
        self.cost_of_transport = 0.0

        self.cur_obs = self.env.reset()
        self.start_pos_high = self.cur_obs[:2]
        self.start_pos = self.cur_obs[:2]

        self.steps_remaining_at_level = 0
        self.env_step_counter = 0

        # w,x,y,z are pu as x,y,z,w
        quat_north_robot = np.concatenate(
            (self.cur_obs[4:7], [self.cur_obs[3]]))
        north_robot = Rotation.from_quat(quat_north_robot).as_euler("xyz")[2]

        maze_obs = self.env.get_current_maze_obs(
            self.cur_obs[0], self.cur_obs[1], north_robot)
        obs_hlc = np.concatenate(
            [maze_obs, self.cur_obs[self.obs_indices["high_level"]]])

        #self.num_high_level_steps = 0
        obs = {}
        # self.cur_obs[self.obs_indices["high_level"]]
        obs["high_level_policy"] = self.cur_obs[self.obs_indices["high_level"]]
        return obs

    def step(self, action_dict):
        #print("action_dict: ", action_dict)
        #assert len(action_dict) == 1, action_dict
        if "high_level_policy" in action_dict:
            return self._high_level_step(action_dict["high_level_policy"])
        else:
            return self._low_level_step(action_dict)

    def _high_level_step(self, action):
        self.latentSpace = action
        # each 10 steps HLC executes an action
        self.steps_remaining_at_level = self.frequency

        self.start_pos_high = self.cur_obs[:2]

        # without the last goal informations
        obs = {"low_level_FL_policy": [
            self.cur_obs[self.obs_indices["low_level_FL_policy"]], self.latentSpace[0:2]]}
        obs["low_level_HL_policy"] = [
            self.cur_obs[self.obs_indices["low_level_HL_policy"]], self.latentSpace[2:4]]
        obs["low_level_HR_policy"] = [
            self.cur_obs[self.obs_indices["low_level_HR_policy"]], self.latentSpace[4:6]]
        obs["low_level_FR_policy"] = [
            self.cur_obs[self.obs_indices["low_level_FR_policy"]], self.latentSpace[6:8]]
        rew = {"low_level_FL_policy": 0, "low_level_HL_policy": 0,
               "low_level_HR_policy": 0, "low_level_FR_policy": 0}
        done = {"__all__": False}

        return obs, rew, done, {}

    def _low_level_step(self, action_dict):
        # Stepping the environment.
        self.steps_remaining_at_level -= 1
        # Use with mujoco 2.
        #functions.mj_rnePostConstraint(self.env.model, self.env.data)

        self.env_step_counter += 1
        # Combine actions from all agents and step environment.
        obs_full, rew_w, done_w, info_w = self.env.step(
            self.concatenate_actions(action_dict))

        current_power = np.sum(
            np.abs(np.roll(self.env.sim.data.ctrl, -2) * self.env.sim.data.qvel[6:]))
        self.power_total += current_power

        # Reward for only low level:
        rew_val, info_w = self.distribute_reward(info_w, action_dict, obs_full)

        self.cur_obs = obs_full

        # save rew and obs of low agent for later run

        obs = {"low_level_FL_policy": [
            self.cur_obs[self.obs_indices["low_level_FL_policy"]], self.latentSpace[0:2]]}
        obs["low_level_HL_policy"] = [
            self.cur_obs[self.obs_indices["low_level_HL_policy"]], self.latentSpace[2:4]]
        obs["low_level_HR_policy"] = [
            self.cur_obs[self.obs_indices["low_level_HR_policy"]], self.latentSpace[4:6]]
        obs["low_level_FR_policy"] = [
            self.cur_obs[self.obs_indices["low_level_FR_policy"]], self.latentSpace[6:8]]

        rew = {"low_level_FL_policy": 0, "low_level_HL_policy": 0,
               "low_level_HR_policy": 0, "low_level_FR_policy": 0}

        # Handle env termination & transitions back to higher level
        done = {"__all__": False}

        # check if the robot reached the target :
        distance_rob_target = np.sqrt(
            ((obs_full[0]-self.env.target_x)**2)+((obs_full[1]-self.env.target_y)**2))
        """
        # if the agent gets upside-down stop the episode
        rot = Rotation.from_quat(np.concatenate(
            (obs_full[4:7], [obs_full[3]])))
        if rot.as_matrix()[2][2] < 0:
            done_w = True
        """

        # done without reaching the goal (reaching is after)
        if done_w:
            done["__all__"] = True
            quat_north_robot = np.concatenate(
                (obs_full[4:7], [obs_full[3]]))
            north_robot = Rotation.from_quat(
                quat_north_robot).as_euler("xyz")[2]

            maze_obs = self.env.get_current_maze_obs(
                obs_full[0], obs_full[1], north_robot)
            obs_hlc = np.concatenate(
                [maze_obs, self.cur_obs[self.obs_indices["high_level"]]])

            rew["high_level_policy"] = 0
            # obs_hlc
            # self.cur_obs[self.obs_indices["high_level"]]
            obs["high_level_policy"] = self.cur_obs[self.obs_indices["high_level"]]

            distance_x = np.sqrt(
                np.sum((self.env.sim.data.qpos[0:2] - self.start_pos)**2))
            com_vel = distance_x/self.env_step_counter

            # mujoco_py.functions.mj_getTotalmass(env.env.model) is equal to the robot mass and the target in the env or even the maze because their geo is in the body of the robot
            # the weight is directly set to: 8.78710174560547

            weight_rob = 8.78710174560547
            self.cost_of_transport = (
                self.power_total/self.env_step_counter) / (weight_rob * com_vel)

            info_w["CoT"] = self.cost_of_transport

        elif self.steps_remaining_at_level == 0:
            quat_north_robot = np.concatenate(
                (obs_full[4:7], [obs_full[3]]))
            north_robot = Rotation.from_quat(
                quat_north_robot).as_euler("xyz")[2]

            maze_obs = self.env.get_current_maze_obs(
                obs_full[0], obs_full[1], north_robot)
            obs_hlc = np.concatenate(
                [maze_obs, self.cur_obs[self.obs_indices["high_level"]]])

            rew["high_level_policy"] = 0
            # obs_hlc
            # self.cur_obs[self.obs_indices["high_level"]]
            obs["high_level_policy"] = self.cur_obs[self.obs_indices["high_level"]]

        if (self.env.is_in_first_goal((obs_full[0], obs_full[1]))):
            rew["high_level_policy"] = 1
            obs["high_level_policy"] = self.cur_obs[self.obs_indices["high_level"]]

        if (self.env.is_in_second_goal((obs_full[0], obs_full[1]))):
            rew["high_level_policy"] = 1
            obs["high_level_policy"] = self.cur_obs[self.obs_indices["high_level"]]

        if (self.env.is_in_goal((obs_full[0], obs_full[1]))):
            """
            # render
            img = self.env.env.sim.render(
                width=1280, height=800, camera_name="top_fixed")
            plt.imsave("render/renderReached_" + str(self.env_step_counter).zfill(4) +
                       '.png', img, origin='lower')

            plot_ray(reading=maze_obs, structure=self.env.maze_structure, size_scaling=self.env.maze_size_scaling, robot_xy=np.array([
                obs_full[0], obs_full[1]]), ori=north_robot, sensor_range=self.env.sensor_range, sensor_span=self.env.sensor_span, n_bins=self.env.n_bins, xy_goal=np.array([self.env.target_x, self.env.target_y]), plot_file="plot/plotReached_" + str(self.env_step_counter).zfill(4) + '.png')
            print("reached")
            """
            done["__all__"] = True

            quat_north_robot = np.concatenate(
                (obs_full[4:7], [obs_full[3]]))
            north_robot = Rotation.from_quat(
                quat_north_robot).as_euler("xyz")[2]

            maze_obs = self.env.get_current_maze_obs(
                obs_full[0], obs_full[1], north_robot)
            obs_hlc = np.concatenate(
                [maze_obs, self.cur_obs[self.obs_indices["high_level"]]])

            self.goalReachedCounter += 1
            rew["high_level_policy"] = 1#7*(1 -(self.env_step_counter/self.target_reward))
            # self.cur_obs[self.obs_indices["high_level"]]
            obs["high_level_policy"] = self.cur_obs[self.obs_indices["high_level"]]

            distance_x = np.sqrt(
                np.sum((self.env.sim.data.qpos[0:2] - self.start_pos)**2))
            com_vel = distance_x/self.env_step_counter

            # mujoco_py.functions.mj_getTotalmass(env.env.model) is equal to the robot mass and the target in the env or even the maze because their geo is in the body of the robot
            # the weight is directly set to: 8.78710174560547

            weight_rob = 8.78710174560547
            self.cost_of_transport = (
                self.power_total/self.env_step_counter) / (weight_rob * com_vel)

            info_w["CoT"] = self.cost_of_transport

        info = {"low_level_FL_policy": info_w, "low_level_HL_policy": info_w,
                "low_level_HR_policy": info_w, "low_level_FR_policy": info_w}

        return obs, rew, done, info
