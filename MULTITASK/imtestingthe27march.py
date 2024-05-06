from marchtwentyseven import *

def execute_final_path(environment, final_path, items):
    for index, step in enumerate(final_path):
        collected_item = None
        if index == len(final_path) - 1:
            collected_item = "goal"
        elif index < len(items):
            collected_item = items[index]
        environment.step_through_path(step, collected_item)


# Example usage
items_to_collect = ["wood", "iron", "toolshed", "grass"]

# Assuming you have the final path stored in a variable final_path
final_path = [(0, 0), (1, 0), (2, 0), (2, 1), (3, 1), (3, 2), (3, 3)]

# Assuming you have an instance of your environment stored in a variable environment
env = GridWorldEnv(render_mode="human", size=5)

# Execute the final path dynamically
execute_final_path(env, final_path, items_to_collect)



