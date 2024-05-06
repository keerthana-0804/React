import pygame
import sys
import random
import numpy as np

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Set the dimensions of the screen and grid size
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
GRID_SIZE = 100  # Adjusted to create a 5x5 grid with 25 boxes

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("LTL Task Completion")

# Load images for items and agent
wood_img = pygame.image.load('minewood.png')
iron_img = pygame.image.load('mineiron.png')
gold_img = pygame.image.load('gold.png')
toolshed_img = pygame.image.load('mineshed.png')
grass_img = pygame.image.load('minegrass.png')
furnace_img = pygame.image.load('minegrass.png')
agent_img = pygame.image.load('mineplayer.png')

# Resize images to fit grid size
wood_img = pygame.transform.scale(wood_img, (GRID_SIZE, GRID_SIZE))
iron_img = pygame.transform.scale(iron_img, (GRID_SIZE, GRID_SIZE))
gold_img = pygame.transform.scale(gold_img, (GRID_SIZE, GRID_SIZE))
toolshed_img = pygame.transform.scale(toolshed_img, (GRID_SIZE, GRID_SIZE))
grass_img = pygame.transform.scale(grass_img, (GRID_SIZE, GRID_SIZE))
furnace_img = pygame.transform.scale(furnace_img, (GRID_SIZE, GRID_SIZE))
agent_img = pygame.transform.scale(agent_img, (GRID_SIZE, GRID_SIZE))

# Define constants for items
WOOD = 0
IRON = 1
GOLD = 2
TOOLSHED = 3
GRASS = 4
FURNACE = 5

# Define atomic propositions
WoodCollected = False
IronCollected = False
GoldCollected = False
AtToolshed = False
FurnaceBuilt = False

# Define LTL formulas
phi1 = lambda: WoodCollected and IronCollected
phi2 = lambda: GoldCollected and AtToolshed

# Define the agent
agent_x = 50
agent_y = 50

# Define the items grid
items = [[random.randrange(0, SCREEN_WIDTH - GRID_SIZE, GRID_SIZE), 
          random.randrange(0, SCREEN_HEIGHT - GRID_SIZE, GRID_SIZE), WOOD],
         [random.randrange(0, SCREEN_WIDTH - GRID_SIZE, GRID_SIZE), 
          random.randrange(0, SCREEN_HEIGHT - GRID_SIZE, GRID_SIZE), IRON],
         [random.randrange(0, SCREEN_WIDTH - GRID_SIZE, GRID_SIZE), 
          random.randrange(0, SCREEN_HEIGHT - GRID_SIZE, GRID_SIZE), GOLD],
         [random.randrange(0, SCREEN_WIDTH - GRID_SIZE, GRID_SIZE), 
          random.randrange(0, SCREEN_HEIGHT - GRID_SIZE, GRID_SIZE), TOOLSHED],
         [random.randrange(0, SCREEN_WIDTH - GRID_SIZE, GRID_SIZE), 
          random.randrange(0, SCREEN_HEIGHT - GRID_SIZE, GRID_SIZE), GRASS],
         [random.randrange(0, SCREEN_WIDTH - GRID_SIZE, GRID_SIZE), 
          random.randrange(0, SCREEN_HEIGHT - GRID_SIZE, GRID_SIZE), FURNACE]]

# Function to draw agent and items
def draw_items():
    for item in items:
        if item[2] == WOOD:
            screen.blit(wood_img, (item[0], item[1]))
        elif item[2] == IRON:
            screen.blit(iron_img, (item[0], item[1]))
        elif item[2] == GOLD:
            screen.blit(gold_img, (item[0], item[1]))
        elif item[2] == TOOLSHED:
            screen.blit(toolshed_img, (item[0], item[1]))
        elif item[2] == GRASS:
            screen.blit(grass_img, (item[0], item[1]))
        elif item[2] == FURNACE:
            screen.blit(furnace_img, (item[0], item[1]))

    screen.blit(agent_img, (agent_x, agent_y))

# Initialize Q-table
num_actions = 4  # Up, Down, Left, Right
num_states = (SCREEN_WIDTH // GRID_SIZE, SCREEN_HEIGHT // GRID_SIZE)
q_table = np.zeros((num_states[0], num_states[1], num_actions))

# Hyperparameters
learning_rate = 0.1
discount_factor = 0.99
exploration_rate = 1.0
max_exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay_rate = 0.01

print("TRAINING STARTED")
# Training loop
for episode in range(1, 25):
    print(f"Episode: {episode}")  # Print episode number before starting the episode

    # Reset environment
    agent_x = random.randrange(0, SCREEN_WIDTH - GRID_SIZE, GRID_SIZE)
    agent_y = random.randrange(0, SCREEN_HEIGHT - GRID_SIZE, GRID_SIZE)
    WoodCollected = False
    IronCollected = False
    GoldCollected = False
    AtToolshed = False
    FurnaceBuilt = False
    items = [[random.randrange(0, SCREEN_WIDTH - GRID_SIZE, GRID_SIZE), 
              random.randrange(0, SCREEN_HEIGHT - GRID_SIZE, GRID_SIZE), WOOD],
             [random.randrange(0, SCREEN_WIDTH - GRID_SIZE, GRID_SIZE), 
              random.randrange(0, SCREEN_HEIGHT - GRID_SIZE, GRID_SIZE), IRON],
             [random.randrange(0, SCREEN_WIDTH - GRID_SIZE, GRID_SIZE), 
              random.randrange(0, SCREEN_HEIGHT - GRID_SIZE, GRID_SIZE), GOLD],
             [random.randrange(0, SCREEN_WIDTH - GRID_SIZE, GRID_SIZE), 
              random.randrange(0, SCREEN_HEIGHT - GRID_SIZE, GRID_SIZE), TOOLSHED],
             [random.randrange(0, SCREEN_WIDTH - GRID_SIZE, GRID_SIZE), 
              random.randrange(0, SCREEN_HEIGHT - GRID_SIZE, GRID_SIZE), GRASS],
             [random.randrange(0, SCREEN_WIDTH - GRID_SIZE, GRID_SIZE), 
              random.randrange(0, SCREEN_HEIGHT - GRID_SIZE, GRID_SIZE), FURNACE]]
    print("Episode started")  # Print when the episode starts

    # Explore environment and learn
    for step in range(100):
        # Display environment
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Exploration-exploitation trade-off
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            # Exploitation: Choose the action with the highest Q-value for the current state
            action = np.argmax(q_table[agent_x // GRID_SIZE, agent_y // GRID_SIZE])
        else:
            # Exploration: Choose a random action
            action = random.randint(0, num_actions - 1)

        # Perform the action
        if action == 0:  # Up
            agent_y -= GRID_SIZE
        elif action == 1:  # Down
            agent_y += GRID_SIZE
        elif action == 2:  # Left
            agent_x -= GRID_SIZE
        elif action == 3:  # Right
            agent_x += GRID_SIZE

        # Clip agent position to screen boundaries
        agent_x = max(0, min(agent_x, SCREEN_WIDTH - GRID_SIZE))
        agent_y = max(0, min(agent_y, SCREEN_HEIGHT - GRID_SIZE))
                # Adjust the speed of the agent
        #agent_speed = 1000  # Increase the value to make the agent slower

        # Inside the training loop after each movement of the agent
        #pygame.time.delay(agent_speed)  # Add this line to introduce a delay

        # Check for collisions with items
        for item in items:
            if agent_x < item[0] + GRID_SIZE and agent_x + GRID_SIZE > item[0] \
                    and agent_y < item[1] + GRID_SIZE and agent_y + GRID_SIZE > item[1]:
                if item[2] == WOOD:
                    WoodCollected = True
                elif item[2] == IRON:
                    IronCollected = True
                elif item[2] == GOLD:
                    GoldCollected = True
                elif item[2] == TOOLSHED:
                    AtToolshed = True
                elif item[2] == FURNACE:
                    FurnaceBuilt = True
                
                # Remove the collected item
                items.remove(item)

        # Check LTL formulas
        if phi1():
            print("Task 1: Wood and Iron collected.")
        if phi2():
            print("Task 2: Gold collected and at toolshed.")

        # Update Q-values
        if not WoodCollected and not IronCollected:
            reward = -1
        elif not GoldCollected and not AtToolshed:
            reward = -1
        else:
            reward = 0

        new_state = (agent_x // GRID_SIZE, agent_y // GRID_SIZE)
        max_future_q = np.max(q_table[new_state])
        current_q = q_table[agent_x // GRID_SIZE, agent_y // GRID_SIZE, action]

        new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_factor * max_future_q)
        q_table[agent_x // GRID_SIZE, agent_y // GRID_SIZE, action] = new_q

        print("Q-Table:")
        print(q_table)  # Print the Q-table after each update

        print(f"Reward: {reward}")  # Print the reward obtained in the current step

        # Decay exploration rate
        exploration_rate = min_exploration_rate + \
                           (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate)

        # Clear the screen
        screen.fill(WHITE)

        # Draw grid lines
        for x in range(0, SCREEN_WIDTH, GRID_SIZE):
            pygame.draw.line(screen, BLACK, (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, GRID_SIZE):
            pygame.draw.line(screen, BLACK, (0, y), (SCREEN_WIDTH, y))

        # Draw items and agent
        draw_items()

        # Update the display
        pygame.display.flip()
    
    print("Episode ended")  # Print when the episode ends
    # Testing episode skipped for brevity

pygame.quit()
sys.exit()