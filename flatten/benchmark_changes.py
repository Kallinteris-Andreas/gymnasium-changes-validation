from gymnasium.utils.performance import benchmark_step

import mujoco_old
import mujoco_new

old_env = mujoco_old.AntEnv()
new_env = mujoco_new.AntEnv()
print(f"ant speedup = {benchmark_step(old_env)/ benchmark_step(new_env)}")

old_env = mujoco_old.HalfCheetahEnv()
new_env = mujoco_new.HalfCheetahEnv()
print(f"half cheetah speedup = {benchmark_step(old_env)/ benchmark_step(new_env)}")

old_env = mujoco_old.HopperEnv()
new_env = mujoco_new.HopperEnv()
print(f"hopper speedup = {benchmark_step(old_env)/ benchmark_step(new_env)}")

old_env = mujoco_old.HumanoidEnv()
new_env = mujoco_new.HumanoidEnv()
print(f"humanoid speedup = {benchmark_step(old_env)/ benchmark_step(new_env)}")

old_env = mujoco_old.HumanoidStandupEnv()
new_env = mujoco_new.HumanoidStandupEnv()
print(f"humanoid standup speedup = {benchmark_step(old_env)/ benchmark_step(new_env)}")

old_env = mujoco_old.InvertedDoublePendulumEnv()
new_env = mujoco_new.InvertedDoublePendulumEnv()
print(f"double pendulum speedup = {benchmark_step(old_env)/ benchmark_step(new_env)}")

old_env = mujoco_old.InvertedPendulumEnv()
new_env = mujoco_new.InvertedPendulumEnv()
print(f"pendulum speedup = {benchmark_step(old_env)/ benchmark_step(new_env)}")

old_env = mujoco_old.PusherEnv()
new_env = mujoco_new.PusherEnv()
print(f"pusher speedup = {benchmark_step(old_env)/ benchmark_step(new_env)}")

old_env = mujoco_old.ReacherEnv()
new_env = mujoco_new.ReacherEnv()
print(f"reacher speedup = {benchmark_step(old_env)/ benchmark_step(new_env)}")

old_env = mujoco_old.SwimmerEnv()
new_env = mujoco_new.SwimmerEnv()
print(f"swimmer speedup = {benchmark_step(old_env)/ benchmark_step(new_env)}")

old_env = mujoco_old.Walker2dEnv()
new_env = mujoco_new.Walker2dEnv()
print(f"walker speedup = {benchmark_step(old_env)/ benchmark_step(new_env)}")
