import pygame
import sys
import random

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Initialize Pygame
pygame.init()

# Set the dimensions of the screen
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("LTL Task Completion")

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
agent_width = 20
agent_height = 20
agent_speed = 5

# Define the items grid
items = [[random.randint(0, SCREEN_WIDTH - 50), random.randint(0, SCREEN_HEIGHT - 50), WOOD],
         [random.randint(0, SCREEN_WIDTH - 50), random.randint(0, SCREEN_HEIGHT - 50), IRON],
         [random.randint(0, SCREEN_WIDTH - 50), random.randint(0, SCREEN_HEIGHT - 50), GOLD],
         [random.randint(0, SCREEN_WIDTH - 50), random.randint(0, SCREEN_HEIGHT - 50), TOOLSHED],
         [random.randint(0, SCREEN_WIDTH - 50), random.randint(0, SCREEN_HEIGHT - 50), GRASS],
         [random.randint(0, SCREEN_WIDTH - 50), random.randint(0, SCREEN_HEIGHT - 50), FURNACE]]


# Function to draw agent and items
def draw_items():
    for item in items:
        if item[2] == WOOD:
            pygame.draw.rect(screen, (139, 69, 19), pygame.Rect(item[0], item[1], 30, 30))
        elif item[2] == IRON:
            pygame.draw.rect(screen, (169, 169, 169), pygame.Rect(item[0], item[1], 30, 30))
        elif item[2] == GOLD:
            pygame.draw.rect(screen, (255, 215, 0), pygame.Rect(item[0], item[1], 30, 30))
        elif item[2] == TOOLSHED:
            pygame.draw.rect(screen, (128, 128, 128), pygame.Rect(item[0], item[1], 30, 30))
        elif item[2] == GRASS:
            pygame.draw.rect(screen, (0, 128, 0), pygame.Rect(item[0], item[1], 30, 30))
        elif item[2] == FURNACE:
            pygame.draw.rect(screen, (178, 34, 34), pygame.Rect(item[0], item[1], 30, 30))
    
    pygame.draw.rect(screen, (0, 0, 255), pygame.Rect(agent_x, agent_y, agent_width, agent_height))


# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        agent_x -= agent_speed
    if keys[pygame.K_RIGHT]:
        agent_x += agent_speed
    if keys[pygame.K_UP]:
        agent_y -= agent_speed
    if keys[pygame.K_DOWN]:
        agent_y += agent_speed

    # Check for collisions with items
    for item in items:
        if agent_x < item[0] + 30 and agent_x + agent_width > item[0] and agent_y < item[1] + 30 and agent_y + agent_height > item[1]:
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

    # Clear the screen
    screen.fill(WHITE)

    # Draw items and agent
    draw_items()

    # Update the display
    pygame.display.flip()

    # Limit frames per second
    pygame.time.Clock().tick(30)

pygame.quit()
sys.exit()
