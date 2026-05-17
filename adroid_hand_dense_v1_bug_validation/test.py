import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC
from gymnasium.wrappers import RecordVideo

env_id = "AdroitHandHammer-v2"

# Create the training environment
train_env = gym.make(env_id)

# Initialize and train the SAC agent for 1 million steps
model = SAC("MlpPolicy", train_env, verbose=1)
model.learn(total_timesteps=1_000_000)
train_env.close()

# Create the evaluation environment with video recording enabled
eval_env = gym.make(env_id, render_mode="rgb_array")
eval_env = RecordVideo(eval_env, video_folder="./videos", name_prefix=f"sac-{env_id}", episode_trigger=lambda x: True)

# Generate and save a video of the trained policy for one episode
obs, info = eval_env.reset()
done = False
truncated = False

while not (done or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = eval_env.step(action)

eval_env.close()
