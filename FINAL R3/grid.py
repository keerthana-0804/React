
import gym
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
print(env.observation_space.n)
print(env.action_space.n)
def transition(env):
    for state in range(env.observation_space.n):
        for action in range(env.action_space.n):
            transitions = env.P[state][action]
            print(state,action)
            print(transitions)
transition(env)

