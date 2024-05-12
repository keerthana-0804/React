
import networkx as nx
from testrewardnodes import r as rew
from finalgraphlayout import mlist
""" 
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
g,rm,mlist = graphenv(env, rew, False) """


#print(mlist)
exps=[]
for l in mlist:
    #print(l)
    exps.append(f"state{l[0]} ---> i(at_state{l[1]}) ---> p(state{l[2]})")

#printing mira expressions 
for exp in exps:
    print(exp)
