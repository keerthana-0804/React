def transform_expression(expression):
    # Split the expression by "&" and strip any whitespace
    items = [item.strip() for item in expression.split('&')]
    
    # Identify destination places
    destinations = [item for item in items if item in ["toolshed", "gold", "furnace"]]
    
    # Remove destination places from the items
    remaining_items = [item for item in items if item not in destinations]
    
    # Connect the remaining items
    connected_items = "->".join(["i({})".format(item) for item in remaining_items])
    
    # Add destination places at the end
    transformed_expression = "{}->p({})".format(connected_items, "->".join(destinations))
    
    return transformed_expression

# Example
original_expression = "iron & wood & toolshed"
transformed_expression = transform_expression(original_expression)
print("Original expression:", original_expression)
print("Transformed expression:", transformed_expression)
