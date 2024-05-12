import gymnasium as gym

# Creation of environment 
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False,render_mode="human")
#env=gym.make("NChain-v1",render_mode="human")

observation = env.reset()

# Rendering the environment 
env.render()
