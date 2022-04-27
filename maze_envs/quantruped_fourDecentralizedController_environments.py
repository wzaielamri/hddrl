import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import mujoco_py
from gym import spaces

from maze_envs import QuantrupedMultiPoliciesHierarchicalEnv


class QuantrupedFourControllerSuperEnv(QuantrupedMultiPoliciesHierarchicalEnv):
    """ Derived environment for control of the four-legged agent.
        Allows to instantiate multiple agents for control.

        Super class for all decentralized controller - control is split
        into four different, concurrent control units (policies) 
        each instantiated as a single agent. 

        Class defines 
        - policy_mapping_fn: defines names of the distributed controllers
        - distribute_observations: how to distribute observations towards these controllers
            Is defined in derived classes and differs between the different architectures.
        - distribute_contact_cost: how to distribute (contact) costs individually to controllers 
        - concatenate_actions: how to integrate the control signals from the controllers
    """

    def distribute_observations(self, obs_full):
        """ 
        Construct dictionary that routes to each policy only the relevant
        local information.
        """
        obs_distributed = {}
        for policy_name in self.policy_names:
            obs_distributed[policy_name] = obs_full[self.obs_indices[policy_name], ]
        return obs_distributed

    def concatenate_actions(self, action_dict):
        # Return actions in the (DIFFERENT in Mujoco) order FR - FL - HL - HR
        actions = np.concatenate((action_dict[self.policy_names[3]],
                                  action_dict[self.policy_names[0]],
                                  action_dict[self.policy_names[1]],
                                  action_dict[self.policy_names[2]]))
        return actions

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
        contact_cost[self.policy_names[0]
                     ] = global_contact_costs + np.sum(contact_costs[2:5])
        contact_cost[self.policy_names[1]
                     ] = global_contact_costs + np.sum(contact_costs[5:8])
        contact_cost[self.policy_names[2]] = global_contact_costs + \
            np.sum(contact_costs[8:11])
        contact_cost[self.policy_names[3]
                     ] = global_contact_costs + np.sum(contact_costs[11:14])
        return contact_cost

    @staticmethod
    def policy_mapping_fn(agent_id):
        # Each derived class has to define all agents by name.
        if agent_id.startswith("policy_FL"):
            return "policy_FL"
        elif agent_id.startswith("policy_HL"):
            return "policy_HL"
        elif agent_id.startswith("policy_HR"):
            return "policy_HR"
        else:
            return "policy_FR"


class QuantrupedFullyDecentralizedEnv(QuantrupedFourControllerSuperEnv):
    """ Derived environment for control of the four-legged agent.
        Uses four different, concurrent control units (policies) 
        each instantiated as a single agent. 

        Input scope of each controller: only the controlled leg.

        Class defines 
        - policy_mapping_fn: defines names of the distributed controllers
        - distribute_observations: how to distribute observations towards these controllers
            Is defined in the obs_indices for each leg.
    """

    # This is ordering of the policies as applied here:
    policy_names = ["policy_FL", "policy_HL", "policy_HR", "policy_FR"]

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
        self.obs_indices["policy_FL"] = [
            0, 1, 2, 3, 4, 5, 6, 15, 16, 17, 18, 19, 20, 45,
            7, 8, 21, 22, 29, 30, 39, 40]
        self.obs_indices["policy_HL"] = [
            0, 1, 2, 3, 4, 5, 6, 15, 16, 17, 18, 19, 20, 45,
            9, 10, 23, 24, 31, 32, 41, 42]
        self.obs_indices["policy_HR"] = [
            0, 1, 2, 3, 4, 5, 6, 15, 16, 17, 18, 19, 20, 45,
            11, 12, 25, 26, 33, 34, 43, 44]
        self.obs_indices["policy_FR"] = [
            0, 1, 2, 3, 4, 5, 6, 15, 16, 17, 18, 19, 20, 45,
            13, 14, 27, 28, 35, 36, 37, 38]
        super().__init__(config)

    @staticmethod
    def return_policies():
        # For each agent the policy interface has to be defined.
        obs_space = spaces.Box(-np.inf, np.inf, (22,), np.float64)
        policies = {
            QuantrupedFullyDecentralizedEnv.policy_names[0]: (None,
                                                              obs_space, spaces.Box(np.array([-1., -1.]), np.array([+1., +1.])), {"model": {"custom_model": "fc_glorot_uniform_init_lstm", 'fcnet_hiddens': [64, 64, ], "lstm_cell_size": 64}}),
            QuantrupedFullyDecentralizedEnv.policy_names[1]: (None,
                                                              obs_space, spaces.Box(np.array([-1., -1.]), np.array([+1., +1.])), {"model": {"custom_model": "fc_glorot_uniform_init_lstm", 'fcnet_hiddens': [64, 64, ], "lstm_cell_size": 64}}),
            QuantrupedFullyDecentralizedEnv.policy_names[2]: (None,
                                                              obs_space, spaces.Box(np.array([-1., -1.]), np.array([+1., +1.])), {"model": {"custom_model": "fc_glorot_uniform_init_lstm", 'fcnet_hiddens': [64, 64, ], "lstm_cell_size": 64}}),
            QuantrupedFullyDecentralizedEnv.policy_names[3]: (None,
                                                              obs_space, spaces.Box(np.array([-1., -1.]), np.array([+1., +1.])), {"model": {"custom_model": "fc_glorot_uniform_init_lstm", 'fcnet_hiddens': [64, 64, ], "lstm_cell_size": 64}}),
        }
        return policies
