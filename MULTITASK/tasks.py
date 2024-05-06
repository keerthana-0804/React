from sympy import symbols, And, Or, simplify_logic

# Define symbols for each item
wood, iron, grass, toolshed, gold = symbols('wood iron grass toolshed gold')


def evaluate_task(simplified_task, items_list):
    # Split the simplified expression into terms based on logical operators
    terms = str(simplified_task).split('&')

    for term in terms:
        # Remove whitespace and parentheses
        term = term.strip().replace('(', '').replace(')', '')

        # If term contains '|', it indicates OR condition
        if '|' in term:
            # Split the term based on '|'
            sub_terms = term.split('|')
            # Check if any sub-term is already present in items_list
            found = False
            for sub_term in sub_terms:
                if sub_term.strip() in items_list:
                    found = True
                    break
            # If none of the sub-terms are present in items_list, add the first sub-term
            if not found:
                items_list.append(sub_terms[0].strip())
        else:
            # Add the term to the items_list
            items_list.append(term.strip())

    return items_list

# Example usage:
task1 = "iron & toolshed & wood"
task2 = "gold & (grass | wood)"

# Initialize an empty list to store items
items_list = []

# Parse and evaluate Task 1
parsed_task1 = And(iron, toolshed, wood)
simplified_task1 = simplify_logic(parsed_task1)
items_list = evaluate_task(simplified_task1, items_list)

# Parse and evaluate Task 2
parsed_task2 = Or(And(wood, gold), And(grass, gold))
simplified_task2 = simplify_logic(parsed_task2)
items_list = evaluate_task(simplified_task2, items_list)

print("Items to be picked for Task 1:", items_list)
