from gym.envs.registration import registry, register, make, spec
from ray.tune.registry import register_env
from gym.wrappers.time_limit import TimeLimit
from seek_envs.quantruped_v3 import QuAntrupedEnv
from seek_envs.ant_v3_mujoco_2 import AntEnvMujoco2

# Importing the different multiagent environments.
from seek_envs.quantruped_adaptor_multi_environment import QuantrupedMultiPoliciesEnv
from seek_envs.quantruped_centralizedController_environment import Quantruped_Centralized_Env
from seek_envs.quantruped_fourDecentralizedController_environments import QuantrupedFullyDecentralizedEnv
from seek_envs.quantruped_hierarchicalController_environments import QuantrupedHierarchicalEnv
from seek_envs.quantruped_hierarchicalController_environments import QuantrupedHierarchicalFullyDecentralizedEnv
from seek_envs.quantruped_decentralizedHierarchicalController_environments import QuantrupedFullyDecentralizedHierarchicalEnv
from seek_envs.quantruped_decentralizedHierarchicalController_environments import QuantrupedFullyDecentralizedHierarchicalFullyDecentralizedEnv

# Register Gym environment.
register(
    id='QuAntrupedSeek-v3',
    entry_point='seek_envs.quantruped_v3:QuAntrupedEnv',
    max_episode_steps=1000,  # 1000
    # reward_threshold=6000.0,
)

register(
    id='Ant_Muj2_Seek-v3',
    entry_point='seek_envs.ant_v3_mujoco_2:AntEnvMujoco2',
    max_episode_steps=1000,  # 1000
    # reward_threshold=6000.0,
)

# Register single agent ray environment (wrapping gym environment).
register_env("Ant_Muj2_Seek-v3",
             lambda config: TimeLimit(AntEnvMujoco2(), max_episode_steps=1000))  # 1000
register_env("QuAntrupedSeek-v3",
             lambda config: TimeLimit(QuAntrupedEnv(), max_episode_steps=1000))  # 1000

# Register multiagent environments (allowing individual access to individual legs).
register_env("QuantrupedMultiEnv_Centralized_Seek",
             lambda config: Quantruped_Centralized_Env(config))

register_env("QuantrupedMultiEnv_FullyDecentral_Seek",
             lambda config: QuantrupedFullyDecentralizedEnv(config))

register_env("QuantrupedMultiEnv_Hierarchical_Seek",
             lambda config: QuantrupedHierarchicalEnv(config))
register_env("QuantrupedMultiEnv_HierarchicalFullyDecentralized_Seek",
             lambda config: QuantrupedHierarchicalFullyDecentralizedEnv(config))

register_env("QuantrupedMultiEnv_FullyDecentralizedHierarchical_Seek",
             lambda config: QuantrupedFullyDecentralizedHierarchicalEnv(config))
register_env("QuantrupedMultiEnv_FullyDecentralizedHierarchicalFullyDecentralized_Seek",
             lambda config: QuantrupedFullyDecentralizedHierarchicalFullyDecentralizedEnv(config))
