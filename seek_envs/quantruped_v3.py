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
    #print(np.min(hfield), np.max(hfield))
    hfield = hfield - np.max(hfield[fromrow:torow, fromcol:tocol])
    mj_model.hfield_data[:] = hfield.ravel()
    #print("Smoothness set to: ", smoothness)

    """
    # update goal position
    from mujoco_py import modder, MjSim
    sim = MjSim(mj_model)
    
    #modder = GoalTextureModder(sim)
    #modder.whiten_materials()  # ensures materials won't impact colors
    #modder.set_checker('floor', (255, 0, 0), (0, 0, 0), (h_center, w_center))
    # modder.rand_all('floor')
    
    idx = mj_model.site_name2id("goal")
    sim.data.site_xpos[idx][: 3] = [10, 10, 1]
    """


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

        super().__init__(xml_file=os.path.join(os.path.dirname(__file__), 'assets', 'ant_hfield.xml'),
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
        #contact_force = self.contact_forces.flat.copy()
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
        # distance,
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

        # set target pos:
        max_distance = 10
        min_distance = 0.5

        # delete this
        """
        # Exp1: FixFarTarget
        random_angle = np.pi/2  # random.uniform(-180, 180)*np.pi / 180

        self.target_x = math.cos(random_angle) * 100
        self.target_y = math.sin(random_angle) * 100

        
        """
        """
        # Exp2: DifferentAngleFixFarTarget
        random_angle = random.uniform(-180, 180)*np.pi / 180

        self.target_x = math.cos(random_angle) * 100
        self.target_y = math.sin(random_angle) * 100
        """
        """
        # Exp3: FixAnglNearTarget
        random_angle = np.pi/2  # random.uniform(-180, 180)*np.pi / 180

        self.target_x = math.cos(random_angle) * 100
        self.target_y = random.uniform(-max_distance, max_distance)
        distance = np.sqrt((qpos[0]-self.target_x) **
                           2+(qpos[1]-self.target_y)**2)

        while ((distance < min_distance) or (distance > max_distance)):
            self.target_x = self.target_x = math.cos(random_angle) * 100
            self.target_y = random.uniform(-max_distance, max_distance)
            distance = np.sqrt((qpos[0]-self.target_x)
                               ** 2+(qpos[1]-self.target_y)**2)
        """
        # """
        # Exp4: DifferentAngleNear/FarTarget

        self.target_x = random.uniform(-max_distance, max_distance)
        self.target_y = random.uniform(-max_distance, max_distance)
        distance = np.sqrt((qpos[0]-self.target_x) **
                           2+(qpos[1]-self.target_y)**2)

        while ((distance < min_distance) or (distance > max_distance)):
            self.target_x = random.uniform(-max_distance, max_distance)
            self.target_y = random.uniform(-max_distance, max_distance)
            distance = np.sqrt((qpos[0]-self.target_x)
                               ** 2+(qpos[1]-self.target_y)**2)
        # """
        ####

        self.model.body_pos[14][0] = self.target_x
        self.model.body_pos[14][1] = self.target_y

        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def get_z_range(self):
        return self._healthy_z_range
