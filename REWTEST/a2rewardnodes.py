import gym
import networkx as nx
import matplotlib.pyplot as plt

def convert_env_to_graph1(env):
    G = nx.DiGraph()

    for state in range(env.observation_space.n):
        G.add_node(state)

    for state in range(env.observation_space.n):
        for action in range(env.action_space.n):
            transitions = env.P[state][action]
            for prob, next_state, reward, done in transitions:
                G.add_edge(state, next_state, action=action, reward=reward)

    return G

def find_reward_nodes(graph):
    reward_nodes = []
    for edge in graph.edges(data=True):
        #print(edge[2])
        if edge[2].get('reward', 0) > 0:
            #print(edge[1])
            reward_nodes.append(edge[1])
    return reward_nodes



def reward_matrix(env, reward_nodes):
    num_actions = env.action_space.n

    # Initialize a matrix to store rewards
    reward_matrix = [[0] * num_actions for _ in reward_nodes]

    # Populate the reward matrix
    for idx, state in enumerate(reward_nodes):
        for action in range(num_actions):
            transitions = env.P[state][action]
            for prob, next_state, reward, _ in transitions:
                reward_matrix[idx][action] = reward

    # Print the reward matrix as a table
    print("Reward Matrix for Reward Nodes:")
    print("  |", end="")
    for action in range(num_actions):
        print(f"  {action}  |", end="")
    print("\n" + "-" * (6 * (num_actions + 1)))

    for idx, state in enumerate(reward_nodes):
        print(f"{state} |", end="")
        for action in range(num_actions):
            print(f"  {reward_matrix[idx][action]}  |", end="")
        print("\n" + "-" * (6 * (num_actions + 1)))


if __name__ == "__main__":
 # Example usage with FrozenLake
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
    graph1 = convert_env_to_graph1(env)
    reward_nodes = find_reward_nodes(graph1)
    print(" rewards:", reward_nodes) 


    rm=reward_matrix(env,reward_nodes)