import gymnasium
import gymnasium_robotics
import kitchen_env
from gymnasium.utils.env_match import check_environments_match
from gymnasium.wrappers import TimeLimit, OrderEnforcing, PassiveEnvChecker

NUM_STEPS = 10_000

# check behavior of the environment
old_env = gymnasium.make('FrankaKitchen-v1')
new_env = TimeLimit(OrderEnforcing(PassiveEnvChecker(kitchen_env.KitchenEnv())), max_episode_steps=280)
check_environments_match(old_env, new_env, NUM_STEPS)
print("Behavior matches")

# check render of the environment
old_env = gymnasium.make('FrankaKitchen-v1', render_mode='rgb_array')
new_env = TimeLimit(OrderEnforcing(PassiveEnvChecker(kitchen_env.KitchenEnv(render_mode='rgb_array'))), max_episode_steps=280)
check_environments_match(old_env, new_env, 1_000)
print("Rendering matches")
