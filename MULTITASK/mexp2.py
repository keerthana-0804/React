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

# Test the function
input_exp = "EXP=gold & (grass | wood)"
output_exp = parse_input_expression(input_exp)
print("OUTPUT EXPRESSION:", output_exp)


