from gym.envs.mujoco.ant_v3 import AntEnv
import numpy as np
import os
from ray.rllib.utils.annotations import override
from scipy import ndimage
from scipy.signal import convolve2d
from mujoco_py import modder, MjSim
import random
from scipy.spatial.transform import Rotation
import math
import numpy as np
import xml.etree.ElementTree as ET
from core.serializable import Serializable

from maze_envs.maze_env_utils import ray_segment_intersect, point_distance, MazeEnv


DEFAULT_CAMERA_CONFIG = {
    'distance': 15.0,
    'type': 1,  # 1 = Tracking camera, 2 = Fixed
    'trackbodyid': 1,
    'elevation': -20.0,
}


class GoalTextureModder(modder.TextureModder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_checker(self, name, rgb1, rgb2, pos):
        bitmap = self.get_texture(name).bitmap
        cbd1, cbd2 = self.get_checker_matrices(name)

        # print(bitmap)
        rgb1 = np.asarray(rgb1).reshape([1, 1, -1])
        rgb2 = np.asarray(rgb2).reshape([1, 1, -1])
        bitmap[45:50, 45:50] = rgb1  # * cbd1  # + rgb2 * cbd2

        self.upload_texture(name)
        return bitmap


def create_new_hfield(mj_model, smoothness=0.15, bump_scale=2.):
    # Generation of the shape of the height field is taken from the dm_control suite,
    # see dm_control/suite/quadruped.py in the escape task (but we don't use the bowl shape).
    # Their parameters are TERRAIN_SMOOTHNESS = 0.15  # 0.0: maximally bumpy; 1.0: completely smooth.
    # and TERRAIN_BUMP_SCALE = 2  # Spatial scale of terrain bumps (in meters).
    res = mj_model.hfield_ncol[0]
    row_grid, col_grid = np.ogrid[-1:1:res*1j, -1:1:res*1j]
    # Random smooth bumps.
    terrain_size = 2 * mj_model.hfield_size[0, 0]
    bump_res = int(terrain_size / bump_scale)
    bumps = np.random.uniform(smoothness, 1, (bump_res, bump_res))
    smooth_bumps = ndimage.zoom(bumps, res / float(bump_res))
    # Terrain is elementwise product.
    hfield = (smooth_bumps - np.min(smooth_bumps)
              )[0:mj_model.hfield_nrow[0], 0:mj_model.hfield_ncol[0]]
    # Clears a patch shaped like box, assuming robot is placed in center of hfield.
    # Function was implemented in an old rllab version.
    h_center = int(0.5 * hfield.shape[0])
    w_center = int(0.5 * hfield.shape[1])
    patch_size = 8
    fromrow, torow = h_center - \
        int(0.5*patch_size), h_center + int(0.5*patch_size)
    fromcol, tocol = w_center - \
        int(0.5*patch_size), w_center + int(0.5*patch_size)
    # convolve to smoothen edges somewhat, in case hills were cut off
    K = np.ones((patch_size, patch_size)) / patch_size**2
    s = convolve2d(hfield[fromrow-(patch_size-1):torow+(patch_size-1), fromcol-(
        patch_size-1):tocol+(patch_size-1)], K, mode='same', boundary='symm')
    hfield[fromrow-(patch_size-1):torow+(patch_size-1),
           fromcol-(patch_size-1):tocol+(patch_size-1)] = s
    # Last, we lower the hfield so that the centre aligns at zero height
    # (importantly, we use a constant offset of -0.5 for rendering purposes)
    # print(np.min(hfield), np.max(hfield))
    hfield = hfield - np.max(hfield[fromrow:torow, fromcol:tocol])
    mj_model.hfield_data[:] = hfield.ravel()
    # print("Smoothness set to: ", smoothness)


# maze code adapted from rllab: https://rllab.readthedocs.io/en/latest/index.html#


class MazeEnv(Serializable):

    def __init__(
            self,
            maze_height=0.5,
            maze_size_scaling=2,
            *args,
            **kwargs):
        Serializable.quick_init(self, locals())

        xml_path = os.path.join(os.path.dirname(
            __file__), 'assets', 'ant_hfield.xml')
        tree = ET.parse(xml_path)
        worldbody = tree.find(".//worldbody")

        self.maze_make_contact = True

        self.maze_height = maze_height
        self.maze_size_scaling = maze_size_scaling
        self.maze_structure = [
            [1, 1, 1, 1, 1, 1],
            [1, 'r', 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 'g', 1],
            [1, 1, 1, 1, 1, 1],
        ]

        torso_x, torso_y = self._find_robot()
        self._init_torso_x = torso_x
        self._init_torso_y = torso_y

        for i in range(len(self.maze_structure)):
            for j in range(len(self.maze_structure[0])):
                if str(self.maze_structure[i][j]) == '1':
                    # offset all coordinates so that robot starts at the origin
                    ET.SubElement(
                        worldbody, "geom",
                        name="block_%d_%d" % (i, j),
                        pos="%f %f %f" % (j * self.maze_size_scaling - torso_x,
                                          -(i * self.maze_size_scaling - torso_y),
                                          self.maze_height / 2 * self.maze_size_scaling),
                        size="%f %f %f" % (0.5 * self.maze_size_scaling,
                                           0.5 * self.maze_size_scaling,
                                           self.maze_height / 2 * self.maze_size_scaling),
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="0.4 0.4 0.4 0.5"
                    )

        torso = tree.find(".//body[@name='torso']")
        goal = tree.find(".//body[@name='goal']")
        geoms = torso.findall(".//geom")
        goal_geom = goal.findall(".//geom")

        cameras = worldbody.findall(".//camera")
        for camera in cameras:
            if camera.get("name") == "top_fixed":
                cam_x = int((len(self.maze_structure)//2) *
                            self.maze_size_scaling-torso_x)
                cam_y = int(
                    (-2*len(self.maze_structure[0]))*self.maze_size_scaling+torso_y)

                camera_pos = str(cam_x)+" "+str(cam_y)+" 25"
                camera.set("pos", camera_pos)

        for geom in geoms:
            if 'name' not in geom.attrib:
                raise Exception("Every geom of the torso must have a name "
                                "defined")

        goal_x, goal_y = self._find_goal()
        pos_goal = str(goal_x-torso_x)+" " + str(-(goal_y-torso_y))+" .05"
        goal_geom[0].set("pos", pos_goal)

        if self.maze_make_contact:
            contact = ET.SubElement(
                tree.find("."), "contact"
            )
            for i in range(len(self.maze_structure)):
                for j in range(len(self.maze_structure[0])):
                    if str(self.maze_structure[i][j]) == '1':
                        for geom in geoms:
                            ET.SubElement(
                                contact, "pair",
                                geom1=geom.attrib["name"],
                                geom2="block_%d_%d" % (i, j)
                            )

        # adapt camera position

        #_, file_path = tempfile.mkstemp(text=True)
        # here we write a temporal file with the robot specifications. Why not the original one??
        self.maze_xmlfile_path = os.path.join(os.path.dirname(
            __file__), 'assets', 'ant_hfield_maze.xml')
        tree.write(self.maze_xmlfile_path)

    def _find_robot(self):
        structure = self.maze_structure
        size_scaling = self.maze_size_scaling
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == 'r':
                    return j * size_scaling, i * size_scaling
        assert False

    def _find_goal(self):
        structure = self.maze_structure
        size_scaling = self.maze_size_scaling
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == 'g':
                    return j * size_scaling, i * size_scaling
        assert False


class QuAntrupedEnv(AntEnv):
    """ Environment with a quadruped walker - derived from the ant_v3 environment

        Uses a different observation space compared to the ant environment (less inputs).
        Per default, healthy reward is turned of (unnecessary).

        The environment introduces a heightfield which allows to test or train
        the system in uneven terrain (generating new heightfields has to be explicitly
        called, ideally before a reset of the system).
    """

    def __init__(self, ctrl_cost_weight=0.5, contact_cost_weight=5e-4, healthy_reward=0., hf_smoothness=1.):

        # for the first init (_get_obs())

        self.target_x = 0
        self.target_y = 0

        # maze hyperparameter
        self.n_bins = 20
        self.sensor_range = 10.
        self.sensor_span = 2*math.pi
        self.maze_height = 0.5
        self.maze_size_scaling = 4
        self.maze_structure = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 'r', 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 'g', 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]

        self.firstRawDoor = None
        self.secondRawDoor = None
        self.thirdRawDoor = None

        maze = MazeEnv()
        super().__init__(xml_file=os.path.join(os.path.dirname(
            __file__), 'assets', 'ant_hfield_maze.xml'),
            # super().__init__(xml_file=maze.maze_xmlfile_path,


            ctrl_cost_weight=ctrl_cost_weight, contact_cost_weight=contact_cost_weight, exclude_current_positions_from_observation=False)
        self.ctrl_cost_weight = self._ctrl_cost_weight
        self.contact_cost_weight = self._contact_cost_weight

        # Heightfield
        self.hf_smoothness = hf_smoothness
        self.hf_bump_scale = 2.
        create_new_hfield(self.model, self.hf_smoothness, self.hf_bump_scale)

        # Otherwise when learning from scratch might abort
        # This allows for more collisions.
        self.model.nconmax = 500
        self.model.njmax = 2000

    def create_new_random_hfield(self):
        create_new_hfield(self.model, self.hf_smoothness, self.hf_bump_scale)

    def _get_obs(self):
        """
        Observation space for the QuAntruped model.

        Following observation spaces are used:
        * position information
        * velocity information
        * passive forces acting on the joints
        * last control signal

        Unfortunately, the numbering schemes are different for the legs depending on the
        specific case: actions and measurements use each their own scheme.

        For actions (action_space and .sim.data.ctrl) ordering is
        (front means x direction, in rendering moving to the right; rewarded direction)
            Front right: 0 = hip joint - positive counterclockwise (from top view),
                         1 = knee joint - negative is up
            Front left: 2 - pos. ccw., 3 - neg. is up
            Hind left: 4 - pos. ccw., 5 - pos. is up
            Hind right: 6 - pos. ccw., 7 - pos. is up

        For measured observations (basically everything starting with a q) ordering is:
            FL: 0, 1
            HL: 2, 3
            HR: 4, 5
            FR: 6, 7
        """
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        # contact_force = self.contact_forces.flat.copy()
        # Provide passive force instead -- in joint reference frame = eight dimensions
        # joint_passive_forces = self.sim.data.qfrc_passive.flat.copy()[6:]
        # Sensor measurements in the joint:
        # qfrc_unc is the sum of all forces outside constraints (passive, actuation, gravity, applied etc)
        # qfrc_constraint is the sum of all constraint forces.
        # If you add up these two quantities you get the total force acting on each joint
        # which is what a torque sensor should measure.
        # See note in http://www.mujoco.org/forum/index.php?threads/best-way-to-represent-robots-torque-sensors.4181/
        joint_sensor_forces = self.sim.data.qfrc_unc[6:] + \
            self.sim.data.qfrc_constraint[6:]

        # Provide actions from last time step (as used in the simulator = clipped)
        last_control = self.sim.data.ctrl.flat.copy()
        # self.sim.data.get_body_xpos("goal")
        target_pos = [self.target_x, self.target_y]

        distance = [np.sqrt((position[0]-target_pos[0]) **
                            2 + (position[1]-target_pos[1])**2)]

        # w,x,y,z are pu as x,y,z,w
        quat_north_robot = np.concatenate((position[4:7], [position[3]]))
        north_robot = Rotation.from_quat(quat_north_robot).as_euler("xyz")

        # avoid zero division
        try:
            direction_target = (math.atan2(
                target_pos[1]-position[1], target_pos[0]-position[0]))
        except:
            direction_target = (math.atan2(
                target_pos[1]-position[1], (target_pos[0]-position[0])+1e-12))

        direction_head_to_target = direction_target - north_robot[2]

        # direction_head_to_target between [-180,+180]
        if direction_head_to_target >= math.pi:
            direction_head_to_target -= 2*math.pi
        elif direction_head_to_target <= -math.pi:
            direction_head_to_target += 2*math.pi

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        # , last_control)) #, contact_force))
        observations = np.concatenate(
            (position, velocity, joint_sensor_forces, last_control,  [direction_head_to_target]))

        return observations

    def set_hf_parameter(self, smoothness, bump_scale=None):
        # Setting the parameters for the height field.
        self.hf_smoothness = smoothness
        if bump_scale:
            self.hf_bump_scale = bump_scale

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def reset_model(self):
        # reset maze struct
        self.maze_structure = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 'r', 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 'g', 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
        firstRawID = [13, 14, 15, 16, 17, 18, 19, ]
        secondRawID = [24, 25, 26, 27, 28, 29, 30, ]
        thirdRawID = [35, 36, 37, 38, 39, 40, 41, ]

        # reset the last doors:
        if self.firstRawDoor != None:
            self.model.geom_rgba[firstRawID[self.firstRawDoor]] = [
                0.4, 0.4, 0.4, 1]
            self.model.geom_pos[firstRawID[self.firstRawDoor]][2] = 1
        if self.secondRawDoor != None:
            self.model.geom_rgba[secondRawID[self.secondRawDoor]] = [
                0.4, 0.4, 0.4, 1]
            self.model.geom_pos[secondRawID[self.secondRawDoor]][2] = 1
        if self.thirdRawDoor != None:
            self.model.geom_rgba[thirdRawID[self.thirdRawDoor]] = [
                0.4, 0.4, 0.4, 1]
            self.model.geom_pos[thirdRawID[self.thirdRawDoor]][2] = 1

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)

        offset_rot = random.uniform(-180, 180)

        # Create a rotation object from Euler angles specifying axes of rotation
        rot = Rotation.from_euler('xyz', [0, 0, offset_rot], degrees=True)

        # Convert to quaternions and print
        rot_quat = rot.as_quat()

        # because the rot_quat function gives: x,y,z,w and not w,x,y,z
        qpos[3] = rot_quat[3]
        qpos[4:7] = rot_quat[0:3]

        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        # TargetSameRow
        # """
        # random target position
        y_target_rand = random.choice(range(4))
        x_target_rand = random.choice(range(7))
        y_rob_rand = y_target_rand
        x_rob_rand = random.choice(range(7))
        # target and robot are not in the same position
        while ((x_target_rand == x_rob_rand)):
            x_target_rand = random.choice(range(7))
        # """

        # Two Row Maze
        """
        y_target_rand = random.choice(range(2))
        x_target_rand = random.choice(range(7))
        y_rob_rand = random.choice(range(2))
        x_rob_rand = random.choice(range(7))
        # target and robot are not in the same position
        while ((x_target_rand == x_rob_rand) and (y_target_rand == y_rob_rand)):
            y_target_rand = random.choice(range(2))
            x_target_rand = random.choice(range(7))
        """

        # Full Maze
        """ 
        y_target_rand = random.choice(range(4)) 
        x_target_rand = random.choice(range(7))
        y_rob_rand = random.choice(range(4)) 
        x_rob_rand = random.choice(range(7))
        # target and robot are not in the same position
        # 
        while ((x_target_rand == x_rob_rand) and (y_target_rand == y_rob_rand)):
            y_target_rand = random.choice(range(4))
            x_target_rand = random.choice(range(7))
        """

        # + self.np_random.uniform(low=noise_low, high=noise_high)
        self.target_x = x_target_rand * self.maze_size_scaling

        # - self.np_random.uniform(low=noise_low, high=noise_high)
        self.target_y = -(y_target_rand*2 * self.maze_size_scaling)

        qpos[0] = (x_rob_rand * self.maze_size_scaling) + \
            self.np_random.uniform(low=noise_low, high=noise_high)

        qpos[1] = -(y_rob_rand*2 * self.maze_size_scaling) - \
            self.np_random.uniform(low=noise_low, high=noise_high)

        # set target
        self.model.body_pos[14][0] = self.target_x
        self.model.body_pos[14][1] = self.target_y
        # set start (only usefull for the visualization)
        self.model.body_pos[15][0] = x_rob_rand * self.maze_size_scaling
        self.model.body_pos[15][1] = -(y_rob_rand*2 * self.maze_size_scaling)

        # adjust the maze doors

        self.firstRawDoor = random.choice(range(7))
        self.secondRawDoor = random.choice(range(7))
        self.thirdRawDoor = random.choice(range(7))

        # adjust doors
        self.model.geom_rgba[firstRawID[self.firstRawDoor]] = [
            0.4, 0.4, 0.4, 0.5]
        self.model.geom_pos[firstRawID[self.firstRawDoor]][2] = 0.5 * 8
        self.model.geom_rgba[secondRawID[self.secondRawDoor]] = [
            0.4, 0.4, 0.4, 0.5]
        self.model.geom_pos[secondRawID[self.secondRawDoor]][2] = 0.5 * 8
        self.model.geom_rgba[thirdRawID[self.thirdRawDoor]] = [
            0.4, 0.4, 0.4, 0.5]
        self.model.geom_pos[thirdRawID[self.thirdRawDoor]][2] = 0.5 * 8

        # for j in range(1, 54):
        #    self.model.geom_pos[j][2] = 0.5 * 16
        #    self.model.geom_rgba[j] = [0.4, 0.4, 0.4, 0.1]

        # update the struct
        self.maze_structure[2][self.firstRawDoor+1] = 0
        self.maze_structure[4][self.secondRawDoor+1] = 0
        self.maze_structure[6][self.thirdRawDoor+1] = 0
        self.maze_structure[1][1] = 0  # robot
        self.maze_structure[7][7] = 0  # goal
        self.maze_structure[y_rob_rand*2+1][x_rob_rand+1] = 'r'
        self.maze_structure[y_target_rand*2+1][x_target_rand+1] = 'g'
        # print(np.array(self.maze_structure))

        self.set_state(qpos, qvel)
        observation = self._get_obs()

        return observation

    def get_observation(self):
        return self._get_obs()

    def get_z_range(self):
        return self._healthy_z_range

    def get_current_maze_obs(self, robot_x, robot_y, ori):
        # The observation would include both information about the robot itself as well as the sensors around its
        # environment

        structure = self.maze_structure
        size_scaling = self.maze_size_scaling

        segments = []
        # compute the distance of all segments

        # Get all line segments of the goal and the obstacles
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == 1:  # or structure[i][j] == 'g':
                    cx = (j - 1) * size_scaling
                    cy = -((i-1) * size_scaling)
                    x1 = cx - 0.5 * size_scaling
                    x2 = cx + 0.5 * size_scaling
                    y1 = cy - 0.5 * size_scaling
                    y2 = cy + 0.5 * size_scaling
                    struct_segments = [
                        ((x1, y1), (x2, y1)),
                        ((x2, y1), (x2, y2)),
                        ((x2, y2), (x1, y2)),
                        ((x1, y2), (x1, y1)),
                    ]
                    for seg in struct_segments:
                        segments.append(dict(
                            segment=seg,
                            type=structure[i][j],
                        ))

        wall_readings = np.ones(self.n_bins)
        # goal_readings = np.ones(self.n_bins)

        # print("ori: ", ori)

        for ray_idx in range(self.n_bins):
            ray_ori = ori - self.sensor_span * 0.5 + 1.0 * \
                (2 * ray_idx + 1) / (2 * self.n_bins) * self.sensor_span
            ray_segments = []
            for seg in segments:
                p = ray_segment_intersect(
                    ray=((robot_x, robot_y), ray_ori), segment=seg["segment"])
                if p is not None:
                    ray_segments.append(dict(
                        segment=seg["segment"],
                        type=seg["type"],
                        ray_ori=ray_ori,
                        distance=point_distance(p, (robot_x, robot_y)),
                    ))
            if len(ray_segments) > 0:
                first_seg = sorted(
                    ray_segments, key=lambda x: x["distance"])[0]
                # print first_seg
                if first_seg["type"] == 1:
                    # Wall -> add to wall readings
                    if first_seg["distance"] <= self.sensor_range:
                        wall_readings[ray_idx] = 1 - ((
                            self.sensor_range - first_seg["distance"]) / self.sensor_range)
                # elif first_seg["type"] == 'g':
                    # Goal -> add to goal readings
                #    if first_seg["distance"] <= self.sensor_range:
                #        goal_readings[ray_idx] = 1 - ((
                #            self.sensor_range - first_seg["distance"]) / self.sensor_range)
                else:
                    assert False

        # obs = np.concatenate([
        #    wall_readings,
        #    goal_readings
        # ])
        obs = wall_readings

        return obs

    def is_in_collision(self, pos):
        x, y = pos
        structure = self.maze_structure
        size_scaling = self.maze_size_scaling

        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == 1:
                    cx = (j - 1) * size_scaling
                    cy = -((i-1) * size_scaling)
                    minx = cx - size_scaling * 0.5
                    maxx = cx + size_scaling * 0.5
                    miny = cy - size_scaling * 0.5
                    maxy = cy + size_scaling * 0.5
                    if minx <= x <= maxx and miny <= y <= maxy:
                        return True
        return False

    def is_in_goal(self, pos):
        x, y = pos
        structure = self.maze_structure
        size_scaling = self.maze_size_scaling

        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == 'g':
                    cx = (j - 1) * size_scaling
                    cy = -((i-1) * size_scaling)
                    minx = cx - size_scaling * 0.5
                    maxx = cx + size_scaling * 0.5
                    miny = cy - size_scaling * 0.5
                    maxy = cy + size_scaling * 0.5
                    if minx <= x <= maxx and miny <= y <= maxy:
                        return True
        return False
