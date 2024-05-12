import gymnasium as gym
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from testrewardnodes import r as rew


def mira_lists(state, next_state):
    mira_l = [[state, next_state, next_state]]
    return mira_l

def graphenv(env, rewnodes, cflag,ip):
    G = nx.DiGraph()
    mira_list = []
    expressions=[]
    num_states = env.observation_space.n
    reward_matrix = np.full((num_states, num_states), -1, dtype=int)
    for state in range(env.observation_space.n):
        G.add_node(state)
    env = env.unwrapped # added this line for the error Timelimit
    for state in range(env.observation_space.n):
        for action in range(env.action_space.n):
            transitions = env.P[state][action]
            for prob, next_state, reward, done in transitions:
                if state == next_state and state in rewnodes :
                    # Add self-loops only for specific reward nodes
                    G.add_edge(state, state, action=action)
                    reward_matrix[state, next_state] = 100
                    mira_list.extend(mira_lists(state, next_state))
                elif next_state in rewnodes:
                    G.add_edge(state, next_state, action=action)
                    reward_matrix[state, next_state] = 100
                    mira_list.extend(mira_lists(state, next_state))
                elif state != next_state and next_state not in ip:
                    # Add edges without weights for all transitions
                    G.add_edge(state, next_state, action=action)
                    reward_matrix[state, next_state] = 0
                    mira_list.extend(mira_lists(state, next_state))

    return G,reward_matrix,mira_list


env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
ip=[5,7,11,12]
g,rm,mlist = graphenv(env, rew, False,ip)

if __name__ == "__main__":
    pos = nx.spring_layout(g)
    
    # Get edge labels from the graph
    #edge_labels = {(u, v): d['reward'] for u, v, d in g.edges(data=True)}
    
    nx.draw(g, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_size=8, arrowsize=10)

    # Highlight reward nodes in red
    nx.draw_networkx_nodes(g, pos, nodelist=rew, node_color='red', node_size=700)

    # Draw edge labels
    #nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=6, font_family='sans-serif')

    # Display the graph
    # Add a title to the graph
    plt.title("Frozenlake Environment Graph", fontsize=16, fontweight='bold')
    # Display the graph
    plt.savefig("frozengraph.png")
    plt.show()
    print("Reward nodes:", rew)
    print("Reward matrix:")
    print(rm)
    print(mlist)
    print(len(mlist))
