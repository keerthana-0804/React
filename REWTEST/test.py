import gym
import matplotlib.pyplot as plt
import REWTEST.a3graph as a3graph
import networkx as nx
env = gym.make("Taxi-v3")
graph1 = a3graph.convert_env_to_graph(env)

# Visualize the graph without weights and values
pos = nx.spring_layout(graph1, seed=25)

plt.figure(figsize=(12, 8))
nx.draw(graph1, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_size=8, arrowsize=10)
plt.title("Directed Graph of FrozenLake Environment")
plt.show()
