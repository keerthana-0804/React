import gymnasium as gym
import networkx as nx
import matplotlib.pyplot as plt



def convert_env_to_graph1(env):
    G = nx.DiGraph()
    reward_nodes = []
    print("Observation Space",env.observation_space.n)
    print("Action space",env.action_space.n)
    env = env.unwrapped # added this line for the error Timelimit
    for state in range(env.observation_space.n):
        G.add_node(state)

    for state in range(env.observation_space.n):
        for action in range(env.action_space.n):
            transitions = env.P[state][action]
            #print("Transition:",transitions,"for State: ", state)
            for prob, next_state, reward, done in transitions:
                #print(prob, next_state, reward, done)
                G.add_edge(state, next_state, reward=reward)
                if reward > 0:
                    if next_state not in reward_nodes:
                        reward_nodes.append(next_state)



    return G,reward_nodes


env=gym.make("FrozenLake-v1",map_name="4x4",render_mode="human")
g,r=convert_env_to_graph1(env)
    
    # Print edge attributes
    #for edge in g.edges(data=True):
        #print("Edge:", edge)
    # Draw the graph
if __name__=='__main__':
    pos = nx.spring_layout(g,seed=23)
    nx.draw(g, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_size=8, arrowsize=10)
    
    # Highlight reward nodes in red
    nx.draw_networkx_nodes(g, pos, nodelist=r, node_color='red', node_size=700)

    # Display the graph
    plt.show()
    print("Reward nodes:",r)
    
    
