import pygame
import sys
import random
import numpy as np

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Set the dimensions of the screen and grid size
SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500
GRID_SIZE = 100

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Q-Learning Example")

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

# Define the agent class
class Agent:
    def __init__(self):
        self.x = random.randrange(0, SCREEN_WIDTH - GRID_SIZE, GRID_SIZE)
        self.y = random.randrange(0, SCREEN_HEIGHT - GRID_SIZE, GRID_SIZE)
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.q_table = np.zeros((SCREEN_WIDTH // GRID_SIZE, SCREEN_HEIGHT // GRID_SIZE, len(self.actions)))
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.exploration_rate = 1.0
        self.max_exploration_rate = 1.0
        self.min_exploration_rate = 0.01
        self.exploration_decay_rate = 0.01

    def choose_action(self):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(self.actions)
        else:
            return self.actions[np.argmax(self.q_table[self.x // GRID_SIZE, self.y // GRID_SIZE])]

    def update_q_table(self, action, reward, next_x, next_y):
        current_q = self.q_table[self.x // GRID_SIZE, self.y // GRID_SIZE, self.actions.index(action)]
        max_future_q = np.max(self.q_table[next_x // GRID_SIZE, next_y // GRID_SIZE])
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * max_future_q)
        self.q_table[self.x // GRID_SIZE, self.y // GRID_SIZE, self.actions.index(action)] = new_q

    def move(self, action):
        if action == 'UP':
            self.y = max(0, self.y - GRID_SIZE)
        elif action == 'DOWN':
            self.y = min(SCREEN_HEIGHT - GRID_SIZE, self.y + GRID_SIZE)
        elif action == 'LEFT':
            self.x = max(0, self.x - GRID_SIZE)
        elif action == 'RIGHT':
            self.x = min(SCREEN_WIDTH - GRID_SIZE, self.x + GRID_SIZE)

# Define the environment class
class Environment:
    def __init__(self):
        self.agent = Agent()
        self.items = []

    def reset(self):
        self.agent = Agent()
        self.items = []

        item_positions = set()  # Set to store item positions to check for collisions

        for item_type in [WOOD, IRON, GOLD, TOOLSHED, GRASS, FURNACE]:
            while True:
                x = random.randrange(0, SCREEN_WIDTH - GRID_SIZE, GRID_SIZE)
                y = random.randrange(0, SCREEN_HEIGHT - GRID_SIZE, GRID_SIZE)
                # Check if the current item position overlaps with any existing item
                if (x, y) not in item_positions:
                    item_positions.add((x, y))
                    self.items.append([x, y, item_type])
                    break

    def step(self, action):
        reward = 0
        next_x, next_y = self.agent.x, self.agent.y
        for item in self.items:
            if (self.agent.x < item[0] + GRID_SIZE and self.agent.x + GRID_SIZE > item[0]
                    and self.agent.y < item[1] + GRID_SIZE and self.agent.y + GRID_SIZE > item[1]):
                reward += 1  # Reward for collecting an item
                self.items.remove(item)
                if item[2] == GOLD and (next_x, next_y) == (100, 300):  # Task completion condition
                    reward += 10
        self.agent.move(action)
        self.agent.update_q_table(action, reward, self.agent.x, self.agent.y)
        return reward

    def check_ltl_task1(self):
        # Check if LTL task 1 is satisfied: "(F wood) && (F iron) && (F toolshed)"
        wood_collected = False
        iron_collected = False
        toolshed_reached = False

        for item in self.items:
            if item[2] == WOOD:
                wood_collected = True
            elif item[2] == IRON:
                iron_collected = True
            elif item[2] == TOOLSHED:
                toolshed_reached = True

        ltl_task1_satisfied = wood_collected and iron_collected and toolshed_reached
        return ltl_task1_satisfied

    def check_ltl_task2(self):
        # Check if LTL task 2 is satisfied: "(G (iron && !gold))"
        iron_collected = False
        gold_not_collected = True

        for item in self.items:
            if item[2] == IRON:
                iron_collected = True
            elif item[2] == GOLD:
                gold_not_collected = False

        ltl_task2_satisfied = iron_collected and gold_not_collected
        return ltl_task2_satisfied

# Training loop
def train():
    env = Environment()
    env.reset()
    
    for episode in range(1, 50):
        print("EPISODE NUMBER:", episode)
        print(f"Training Episode: {episode}")
        env.reset()

        for step in range(100):
            action = env.agent.choose_action()
            reward = env.step(action)
            print(f"Step: {step}, Action: {action}, Reward: {reward}")

            # Check LTL tasks after each step
            ltl_task1_satisfied = env.check_ltl_task1()
            ltl_task2_satisfied = env.check_ltl_task2()
            print(f"LTL Task 1 Satisfied: {ltl_task1_satisfied}")
            print(f"LTL Task 2 Satisfied: {ltl_task2_satisfied}")

            # Render the environment (draw items, agent, etc.)
            screen.fill((200, 230, 255))
            for x in range(0, SCREEN_WIDTH, GRID_SIZE):
                pygame.draw.line(screen, BLACK, (x, 0), (x, SCREEN_HEIGHT))
            for y in range(0, SCREEN_HEIGHT, GRID_SIZE):
                pygame.draw.line(screen, BLACK, (0, y), (SCREEN_WIDTH, y))
            for item in env.items:
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
            screen.blit(agent_img, (env.agent.x, env.agent.y))
            pygame.display.flip()

            pygame.time.delay(90)  # Reduced delay for faster movement

        env.agent.exploration_rate = max(env.agent.min_exploration_rate,
                                         env.agent.exploration_rate * env.agent.exploration_decay_rate)

    print("TRAINING FINISHED")

# Testing loop
def test():
    env = Environment()
    env.reset()

    for episode in range(1, 6):  # Test for 5 episodes
        print(f"Testing Episode: {episode}")
        env.reset()

        for step in range(100):
            action = env.agent.choose_action()
            reward = env.step(action)
            print(f"Step: {step}, Action: {action}, Reward: {reward}")

            # Check LTL tasks after each step
            ltl_task1_satisfied = env.check_ltl_task1()
            ltl_task2_satisfied = env.check_ltl_task2()
            print(f"LTL Task 1 Satisfied: {ltl_task1_satisfied}")
            print(f"LTL Task 2 Satisfied: {ltl_task2_satisfied}")

            # Render the environment (draw items, agent, etc.)
            screen.fill((173, 216, 230)) 
            for x in range(0, SCREEN_WIDTH, GRID_SIZE):
                pygame.draw.line(screen, BLACK, (x, 0), (x, SCREEN_HEIGHT))
            for y in range(0, SCREEN_HEIGHT, GRID_SIZE):
                pygame.draw.line(screen, BLACK, (0, y), (SCREEN_WIDTH, y))
            for item in env.items:
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
            screen.blit(agent_img, (env.agent.x, env.agent.y))
            pygame.display.flip()

            pygame.time.delay(100)  # Reduced delay for faster movement

    print("TESTING FINISHED")

# Main function
def main():
    train()
    test()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
