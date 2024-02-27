

import gym
import networkx as nx
from tabulate import tabulate

def convert_graph_to_state_transition_table(graph):
    state_transition_table = {}

    for edge in graph.edges(data=True):
        source, target, data = edge
        action = data['action']

        if source not in state_transition_table:
            state_transition_table[source] = {}

        if action not in state_transition_table[source]:
            state_transition_table[source][action] = []

        if source == target:
            # Self-loop represents staying in the same state
            state_transition_table[source][action].append((1.0, target, 100, False))
        else:
            # Regular transition to the next state
            state_transition_table[source][action].append((1.0, target, 0, False))

    return state_transition_table

# Example usage:
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
reward_nodes = [15]
graph1 = convert_env_to_graph(env, reward_nodes)

state_transition_table = convert_graph_to_state_transition_table(graph1)

# Display the state transition table
for state, actions in state_transition_table.items():
    print(f"State {state}:")
    for action, transitions in actions.items():
        table_data = [(f"{prob * 100:.0f}%", next_state, reward, done) for prob, next_state, reward, done in transitions]
        headers = ["Probability", "Next State", "Reward", "Done"]
        print(f"  Action {action}:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print()
