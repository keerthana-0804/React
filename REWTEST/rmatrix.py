import gym
import numpy as np
from a3graph import l,convert_env_to_graph

def print_reward_expression(state, next_state):
    return f"room{state} ---> r(go_room{next_state}) ---> p(room{next_state})"

def mira_lists(state, next_state):
    mira_l = [[state, next_state, next_state]]
    return mira_l
def create_reward_matrix(env, graph):
    num_states = env.observation_space.n
    reward_matrix = np.full((num_states, num_states), -1, dtype=int)
    expressions = []
    mira_list = []

    custom_rewards = {
        (4, 8): 20,
        (8, 4): 20,
        (6, 10): 10,
        (10, 6): 10,
    }
    
    for state in range(num_states):
        for action in range(env.action_space.n):
            transitions = env.P[state][action]
            for prob, next_state, reward, done in transitions:
                edge = (state, next_state)
                
                if edge in custom_rewards:
                    expressions.append(print_reward_expression(state, next_state))
                    mira_list.extend(mira_lists(state, next_state))
                    reward_matrix[state, next_state] = custom_rewards[edge]
                elif state not in l and next_state in l:
                    expressions.append(print_reward_expression(state, next_state))
                    mira_list.extend(mira_lists(state, next_state))
                    reward_matrix[state, next_state] = 100
                elif state not in l:
                    print(state,next_state)
                    if state != next_state and next_state not in [5, 7, 11, 12]:
                        expressions.append(print_reward_expression(state, next_state))
                        mira_list.extend(mira_lists(state, next_state))
                        reward_matrix[state, next_state] = 0
                else:
                    print(state,next_state)
                    expressions.append(print_reward_expression(state, next_state))
                    mira_list.extend(mira_lists(state, next_state))
                    reward_matrix[state, next_state] = 100

    return reward_matrix, expressions, mira_list


    
# Create FrozenLake environment
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)

# Convert environment to graph
g = convert_env_to_graph(env,l)

# Create reward matrix and get expressions
reward_matrix, all_expressions, mira_list = create_reward_matrix(env, g)
mira_l=mira_list
if __name__ == "__main__":
    # Display the reward matrix
    print("Reward Matrix:")
    print(reward_matrix)
    
    print("Total",len(mira_l))
    # Display mira_list
    print("\Mira List:")
    for l in mira_list:
        print(l)
        
    #print(mira_l)

"""     # Display all expressions
    print("\nReward Expressions:")
    for expression in all_expressions:
        print(expression) """

    