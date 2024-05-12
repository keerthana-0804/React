from searchstr import mira_graph
import gymnasium as gym
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
action_map = {
    (state, next_state): action
    for state, neighbors in mira_graph.items()
    for next_state, action_info in zip(neighbors, env.P[state])
    for action, _ in action_info
}
