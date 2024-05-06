from sympy import symbols, And

#mira tables
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



import re

def parse_input_expression(input_exp):
    # Pattern to match the input expression
    pattern = r"(\w+)\s*=\s*(\w+)\s*&\s*\((\w+)\s*\|\s*(\w+)\)"
    match = re.match(pattern, input_exp)
    
    if match:
        destination = match.group(2)
        source1 = match.group(3)
        source2 = match.group(4)
        
        output_exp = f"T->(i({source1})->p({destination}) ⊕ (i({source2})->p({destination}))"
        return output_exp
    else:
        return "Invalid input expression format."







wood, iron, grass, toolshed, gold = symbols('wood iron grass toolshed gold')
def generate_output_expression(input_expression):
    # Split the input expression into symbols
    symbols_list = input_expression.replace("And(", "").replace(")", "").split(", ")

    # Separate items from the destination
    destination = None
    items = []
    for symbol in symbols_list:
        if "toolshed" in symbol or "furnace" in symbol:  # Assuming the destination contains "toolshed" or "furnace"
            destination = symbol
        else:
            items.append(symbol)

    # Construct the output expression
    output_expr = "T -> ("
    for item in items:
        output_expr += f"i({item})->"
    output_expr += f"p({destination}))"
    return output_expr



if __name__ == "__main__":
        
    # Example usage
    input_expression = "And(iron, toolshed, wood)"
    output_expression = generate_output_expression(input_expression)
    print(output_expression)

