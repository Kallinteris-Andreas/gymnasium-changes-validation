import gymnasium
from gymnasium.utils.env_match import check_environments_match

import mujoco_old
import mujoco_new
NUM_STEPS = int(1e6)

old_env = mujoco_old.AntEnv(render_mode="rgb_array")
new_env = mujoco_new.AntEnv(render_mode="rgb_array")
check_environments_match(old_env, new_env, NUM_STEPS)

old_env = mujoco_old.AntEnv(render_mode="depth_array")
new_env = mujoco_new.AntEnv(render_mode="depth_array")
check_environments_match(old_env, new_env, NUM_STEPS)

"""
#old_env = mujoco_old.HalfCheetahEnv()
#new_env = mujoco_new.HalfCheetahEnv()
#check_environments_match(old_env, new_env, NUM_STEPS)

old_env = mujoco_old.HopperEnv()
new_env = mujoco_new.HopperEnv()
check_environments_match(old_env, new_env, NUM_STEPS)

old_env = mujoco_old.HumanoidEnv()
new_env = mujoco_new.HumanoidEnv()
check_environments_match(old_env, new_env, NUM_STEPS)

old_env = mujoco_old.HumanoidStandupEnv()
new_env = mujoco_new.HumanoidStandupEnv()
check_environments_match(old_env, new_env, NUM_STEPS)

old_env = mujoco_old.InvertedDoublePendulumEnv()
new_env = mujoco_new.InvertedDoublePendulumEnv()
check_environments_match(old_env, new_env, NUM_STEPS)

old_env = mujoco_old.InvertedPendulumEnv()
new_env = mujoco_new.InvertedPendulumEnv()
check_environments_match(old_env, new_env, NUM_STEPS)

old_env = mujoco_old.PusherEnv()
new_env = mujoco_new.PusherEnv()
check_environments_match(old_env, new_env, NUM_STEPS)

old_env = mujoco_old.ReacherEnv()
new_env = mujoco_new.ReacherEnv()
check_environments_match(old_env, new_env, NUM_STEPS)

old_env = mujoco_old.SwimmerEnv()
new_env = mujoco_new.SwimmerEnv()
check_environments_match(old_env, new_env, NUM_STEPS)

old_env = mujoco_old.Walker2dEnv()
new_env = mujoco_new.Walker2dEnv()
check_environments_match(old_env, new_env, NUM_STEPS)
"""
