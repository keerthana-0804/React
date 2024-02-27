from search import all_paths

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



# Function to evaluate the MIRA formula for a given path
def evaluate(path):
    n = len(path)
    print(n)
    currentc = None
    for action in range(len(path)):
        if action == 0:
            print("Node=",path[action] ,"and",path[action+1],end=" ")
            condition = ('T', 'S')
            currentc = tables['reason'].get(condition)
            print("pass", action, currentc)
        elif action == n - 1:
            print("Node=",path[action] ,end=" ")
            condition = (currentc, 'S')
            currentc = tables['goal'].get(condition)
            print("pass", action, currentc)
            print("final output", currentc)
        else:
            print("Node=",path[action] ,"and",path[action+1],end=" ")
            condition = (currentc, 'S')
            currentc = tables['action'].get(condition)
            print("pass", action, currentc)
            result=currentc
    return result
for path in all_paths:
    result = evaluate(path)
    print(f"Path {path}: Result = {result}")
