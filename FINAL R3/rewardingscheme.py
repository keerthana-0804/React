from searchstr import mira_graph,dfs
from finalgraphlayout import rm


# Define tables for enjoining goals, reasons, and actions
tables = {
    'goal': {
        ('S', 'T'): 'S',
        ('S', '⊥'): 'N',
        ('V', 'T'): 'V',
        ('V', '⊥'): 'N',
    },
    'reason': {
        ('⊤', 'T'): 'S',
        ('⊤', '⊥'): 'N',
        ('T', 'S'): 'S',
        ('T', 'V'): 'V',
        ('T', 'N'): 'N',
        ('⊥', 'S'): 'N',
        ('⊥', 'V'): 'N',
        ('⊥', 'N'): 'N',
    },
    'action': {
        ('S', 'S'): 'S',
        ('S', 'V'): 'V',
        ('V', 'S'): 'V',
        ('V', 'V'): 'V',
        ('N', 'S'): 'N',
        ('N', 'V'): 'N',
        ('N', 'N'): 'N',
    },
    'disjunction': {
        ('S', 'S'): 'V',
        ('S', 'V'): 'S',
        ('S', 'N'): 'N',
        ('V', 'S'): 'S',
        ('V', 'V'): 'V',
        ('V', 'N'): 'N',
        ('N', 'S'): 'N',
        ('N', 'V'): 'N',
        ('N', 'N'): 'N',
    }
}

""" initial_state = int(input("Enter initial state: "))
goal_state = int(input("Enter goal state: "))

    # Find all paths using Depth-First Search
all_paths = dfs(mira_graph, initial_state, goal_state)

    # Display all paths
if all_paths:
    print(f"All possible paths from {initial_state} to {goal_state}:")
    for path in all_paths:
            print(path)
else:
    print("No valid paths found.") """



# Function to evaluate the MIRA formula for a given path
def evaluate(path):
    n = len(path)
    #print(n)
    currentc = None
    dict1={}
    for action in range(len(path)):
        
        if action == 0:
            #print("Node=",path[action] ,"and",path[action+1],end=" ")
            condition = ('T', 'S')
            currentc = tables['reason'].get(condition)
            dict1[path[action],path[action+1]]=currentc
            #print("pass", action, currentc)
        elif action == n - 1:
            #print("Node=",path[action] ,end=" ")
            condition = (currentc, 'S')
            currentc = tables['goal'].get(condition)
            #print("pass", action, currentc)
            #print("final output", currentc)
        else:
            #print("Node=",path[action] ,"and",path[action+1],end=" ")
            condition = (currentc, 'S')
            currentc = tables['action'].get(condition)
            dict1[path[action],path[action+1]]=currentc
            #print("pass", action, currentc)
            result=currentc
    
    #printing dictionary
    print(dict1)
    tot=0
    for key,value in dict1.items():            
        if value=="S":
            tot+=rm[key]
            #print("tot iter-----",tot)
    

    return result,tot

""" if __name__ == '__main__':
    for path in all_paths:
        result = evaluate(path)
        print(f"Path {path}: Result = {result}") """

