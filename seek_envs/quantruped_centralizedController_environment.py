import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import mujoco_py
from gym import spaces

from seek_envs import QuantrupedMultiPoliciesEnv


class Quantruped_Centralized_Env(QuantrupedMultiPoliciesEnv):
    """ Derived environment for control of the four-legged agent.
        Allows to instantiate multiple agents for control.

        Centralized approach: Single agent (as standard DRL approach)
        controls all degrees of freedom of the agent.

        Class defines 
        - policy_mapping_fn: defines names of the distributed controllers
        - distribute_observations: how to distribute observations towards these controllers
        - distribute_contact_cost: how to distribute (contact) costs individually to controllers 
        - concatenate_actions: how to integrate the control signals from the controllers
    """

    # This is ordering of the policies as applied here:
    policy_names = ["central_policy"]

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
        # 45: direction_head_to_target
        # The central policy gets all observations

        self.obs_indices["central_policy"] = [0, 1, 2, 3, 4, 5, 6, 15, 16, 17, 18, 19, 20,  45,
                                              7, 8, 21, 22, 29, 30, 39, 40,
                                              9, 10, 23, 24, 31, 32, 41, 42,
                                              11, 12, 25, 26, 33, 34, 43, 44,
                                              13, 14, 27, 28, 35, 36, 37, 38]
        super().__init__(config)

    def distribute_observations(self, obs_full):
        """ 
        Construct dictionary that routes to each policy only the relevant
        information.
        """
        obs_distributed = {}
        for policy_name in self.policy_names:
            obs_distributed[policy_name] = obs_full[self.obs_indices[policy_name], ]
        return obs_distributed

    @staticmethod
    def policy_mapping_fn(agent_id):
        # Each derived class has to define all agents by name.
        return Quantruped_Centralized_Env.policy_names[0]

    @staticmethod
    def return_policies():
        # For each agent the policy interface has to be defined.
        obs_space = spaces.Box(-np.inf, np.inf, (46,), np.float64)

        policies = {
            Quantruped_Centralized_Env.policy_names[0]: (None,
                                                         obs_space, spaces.Box(np.array([-1., -1., -1., -1., -1., -1., -1., -1.]), np.array([+1., +1., +1., +1., +1., +1., +1., +1.])), {}),
        }
        return policies
