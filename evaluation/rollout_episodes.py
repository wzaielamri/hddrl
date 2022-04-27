import collections

from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.env import MultiAgentEnv

import numpy as np
import mujoco_py
from mujoco_py import functions

import copy
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from maze_envs.maze_env_utils import plot_ray

"""
    Running a learned (multiagent) controller,
    for evaluation or visualisation.
    This is adapted from rllib's rollout.py
    (github.com/ray/rllib/rollout.py)
"""


class DefaultMapping(collections.defaultdict):
    """ Provides a default mapping.
    """

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


def rollout_episodes(env, agent, num_episodes=1, num_steps=1000, render=True, camera_name="side_fixed", plot=False, save_images=None, explore_during_rollout=None, tvel=None):
    """
        Rollout an episode:
        step through an episode, using the 
        - agent = trained policies (is a multiagent consisting of a dict of agents)
        - env = in the given environment
        for num_steps control steps and running num_episodes episodes.
        render: shows OpenGL window
        save_images: save individual frames (can be combined to video)
        tvel: set target velocity
    """
    if tvel:
        env.target_velocity_list = [tvel]
    # Setting up the agent for running an episode.
    multiagent = isinstance(env, MultiAgentEnv)
    if agent.workers.local_worker().multiagent:
        policy_agent_mapping = agent.config["multiagent"]["policy_mapping_fn"]
    policy_map = agent.workers.local_worker().policy_map
    state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
    use_lstm = {p: len(s) > 0 for p, s in state_init.items()}

    action_init = {
        p: flatten_to_single_ndarray(m.action_space.sample())
        for p, m in policy_map.items()
    }

    # if save_images:
    #   viewer = mujoco_py.MjRenderContextOffscreen(env.env.sim, 0)

    # Collecting statistics over episodes.
    reward_eps = []
    cot_eps = []
    vel_eps = []
    dist_eps = []
    steps_eps = []
    power_total_eps = []
    avg_power_success = []
    total_power_success = []

    vel_rew_list = []
    for episodes in range(0, num_episodes):
        vel_list = []
        # Reset all values for this episode.
        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        #    saver.begin_rollout()
        agent_states = DefaultMapping(
            lambda agent_id: state_init[mapping_cache[agent_id]])
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]])
        prev_rewards = collections.defaultdict(lambda: 0.)
        done = False
        reward_total = 0.0
        power_total = 0.0
        steps = 0
        done = False
        env.env.create_new_random_hfield()
        obs = env.reset()

        #print("angle: ", math.atan2(env.env.target_y, env.env.target_x)*180/np.pi)
        start_pos = copy.deepcopy(env.env.sim.data.qpos[0:2])
        oldGoal = env.return_GoalReached()

        # Control stepping:
        while not done and steps < num_steps:
            #print("steps: ", steps)
            multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
            action_dict = {}
            for agent_id, a_obs in multi_obs.items():
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id))
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        a_action, p_state, _ = agent.compute_action(
                            a_obs,
                            state=agent_states[agent_id],
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id,
                            explore=explore_during_rollout)
                        agent_states[agent_id] = p_state
                    else:
                        # Sample an action for the current observation
                        # for one entry of the agent dict.

                        # for hierarchical architecture:
                        a_action = agent.compute_action(
                            a_obs,
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                    a_action = flatten_to_single_ndarray(a_action)
                    #a_action[-1] = env.observation[-1]
                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action
            action = action_dict
            action = action if multiagent else action[_DUMMY_AGENT_ID]
            # Stepping the environment.
            next_obs, reward, done, info = env.step(action)
            # if "high_level_policy" in next_obs:
            # print(next_obs["high_level_policy"])
            #    print(np.max(next_obs["high_level_policy"]))

            #print("Env Step Counter: ", env.env_step_counter)
            # print(info)
            # print(next_obs)
            # if info:
            #    vel_list.append(info["low_level_FL_policy"]["velocity_reward"])
            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward
            if multiagent:
                done = done["__all__"]
                reward_total += sum(reward.values())
            else:
                reward_total += reward
            if render:
                if save_images:
                    #viewer.render(1280, 800, 0)
                    if tvel:
                        env.env.model.body_pos[14][0] += tvel * 0.05

                    if "high_level_policy" not in next_obs:
                        img = env.env.sim.render(
                            width=1280, height=800, camera_name=camera_name)  # side_fixed   top_fixed
                        #data = np.asarray(viewer.read_pixels(800, 1280, depth=False)[::-1, :, :], dtype=np.uint8)
                        #img_array = env.env.render('rgb_array')
                        plt.imsave(save_images + str(steps).zfill(4) +
                                   '.png', img, origin='lower')
                    # """
                    if plot:
                        quat_north_robot = np.concatenate(
                            (env.cur_obs[4:7], [env.cur_obs[3]]))
                        north_robot = Rotation.from_quat(
                            quat_north_robot).as_euler("xyz")[2]
                        # print("orientation: ", Rotation.from_quat(
                        #    quat_north_robot).as_euler("xyz")*180/math.pi)
                        maze_obs = env.env.get_current_maze_obs(
                            env.cur_obs[0], env.cur_obs[1], north_robot)
                        plot_ray(reading=maze_obs, structure=env.env.maze_structure, size_scaling=env.env.maze_size_scaling, robot_xy=np.array([
                            env.cur_obs[0], env.cur_obs[1]]), ori=north_robot, sensor_range=env.env.sensor_range, sensor_span=env.env.sensor_span, n_bins=env.env.n_bins, xy_goal=np.array([env.env.target_x, env.env.target_y]), plot_file="plot/plot_" + str(steps).zfill(4) + '.png')
                else:
                    env.render()
            #saver.append_step(obs, action, next_obs, reward, done, info)
            steps += 1
            obs = next_obs
            # Calculated as torque (during last time step - or in this case sum of
            # proportional control signal (clipped to [-1,1], multiplied by 150 to torque)
            # multiplied by joint velocity for each joint.
            # Important: unfortunately there is a shift in the ctrl signals - therefore use roll
            # (control signals start with front right leg, front left leg starts at index 2)
            if not policy_id.startswith("high_level_"):
                current_power = np.sum(
                    np.abs(np.roll(env.env.sim.data.ctrl, -2) * env.env.sim.data.qvel[6:14]))
                power_total += current_power
        # saver.end_rollout()

        # adjust steps count (steps in the environement not policy steps):
        steps = env.env_step_counter
        distance_x = np.sqrt(
            np.sum((env.env.sim.data.qpos[0:2] - start_pos)**2))
        com_vel = distance_x/steps

        # mujoco_py.functions.mj_getTotalmass(env.env.model) is equal to the robot mass and the target in the env or even the maze because their geo is in the body of the robot
        # the weight is directly set to: 8.78710174560547

        weight_rob = 8.78710174560547
        cost_of_transport = (
            power_total/steps) / (weight_rob * com_vel)
        # Weight is 8.78710174560547
        #print(steps, " - ", power_total, " / ", power_total/steps, "; CoT: ", cost_of_transport)
        cot_eps.append(cost_of_transport)
        reward_eps.append(reward_total)
        vel_eps.append(com_vel)
        dist_eps.append(distance_x)
        steps_eps.append(steps)
        power_total_eps.append(power_total)
        if int(env.return_GoalReached()-oldGoal) == 1:  # success
            avg_power_success.append(power_total/steps)
            total_power_success.append(power_total)
        # vel_rew_list.append(np.mean(vel_list))
    #print("reward_total: ", reward_total)

    goal_reached = env.return_GoalReached()
    #np.save("reward_eps_2808", reward_eps)
    #np.save("steps_eps_2808", steps_eps)
    #np.save("cot_eps_2808", cot_eps)
    #np.save("vel_rew_eps_2808", vel_rew_list)

    # Return collected information from episode.
    return (reward_eps, steps_eps, dist_eps, power_total_eps, vel_eps, cot_eps, goal_reached, avg_power_success, total_power_success)
