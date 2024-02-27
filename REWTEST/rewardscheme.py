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

# Define the MIRA formula
mira_formula = "(room2 →r (goto3 →i ((goto1 →p room5)⊕(goto4 →p part5)))"

# Define all possible paths
all_paths = [
    [2, 3, 1, 5],
    [2, 3, 4, 5]
]

# Function to evaluate the MIRA formula for a given path
def evaluate_mira_for_path(path):
    current = path[0]
    print("current path:",current)
    for action in path[1:]:
        print(action,path[1:])
        if action == 3:  # Assuming action 3 corresponds to "goto3"
            condition = ('T', 'S')  # Assuming the condition for successful navigation is true
            current_room = tables['reason'].get((condition))
            k=tables['reason'].get((condition))
            print("k=",k)
        elif action == 1:  # Assuming action 1 corresponds to "goto1"
            condition = (k, 'S')  # Assuming the condition for successful action is true
            current_room = tables['goal'].get((current_room, condition))
            k1=tables['goal'].get((current_room, condition))
        elif action == 4:  # Assuming action 4 corresponds to "goto4"
            condition = ('V', 'S')  # Assuming the condition for successful action is true
            current_room = tables['goal'].get((current_room, condition), 'Unknown')
        else:
            # Handle other actions if needed
            pass

    return current_room

# Evaluate the MIRA formula for all paths
for path in all_paths:
    result = evaluate_mira_for_path(path)
    print(f"Path {path}: Result = {result}")
