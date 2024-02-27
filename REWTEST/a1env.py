import gymnasium as gym
import time as time
# Creation of environment
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False,render_mode="human")


observation = env.reset()
time.sleep(5)
# Rendering the environment 
env.render()


#changes made to this file 
#arguments depends upon gym version