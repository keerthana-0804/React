

import gym
import networkx as nx
import matplotlib.pyplot as plt
from a2rewardnodes import find_reward_nodes,convert_env_to_graph1
from tabulate import tabulate

# Example usage with FrozenLake
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
graph1 = convert_env_to_graph1(env)
reward_nodes = find_reward_nodes(graph1)
l=[]
for i in reward_nodes:
    l.append(i)
#print(l)

def convert_env_to_graph(env, reward_nodes):
    G = nx.DiGraph()

    for state in range(env.observation_space.n):
        G.add_node(state)

    for state in range(env.observation_space.n):
        for action in range(env.action_space.n):
            transitions = env.P[state][action]
            for prob, next_state, reward, done in transitions:
                if state == next_state and state in reward_nodes:
                    # Add self-loops only for specific reward nodes
                    G.add_edge(state, state, action=action)
                elif state != next_state:
                    # Add edges without weights for all transitions
                    G.add_edge(state, next_state, action=action)

    return G

def draw(env, reward_nodes):
    graph1 = convert_env_to_graph(env, reward_nodes)

    # Visualize the graph with arrows and a spring layout
    pos = nx.spring_layout(graph1,seed=23)

    plt.figure(figsize=(15, 15))
    nx.draw(graph1, pos, with_labels=True, font_weight='bold', node_size=100, node_color='skyblue', font_size=8, arrowsize=20)

    # Add rewards on top of arrows
    edge_labels = {}
    for edge in graph1.edges():
        source, target = edge
        label = None
        if target in reward_nodes:
            label = '100'
        else:
            label = '0'
        edge_labels[edge] = label

    nx.draw_networkx_edge_labels(graph1, pos, edge_labels=edge_labels, font_color='red', font_size=8)

    plt.title("Directed Graph of Frozenlake Environment")
    plt.show()






if __name__ == "__main__":
    print("reward nodes:", reward_nodes)
    draw(env, reward_nodes)