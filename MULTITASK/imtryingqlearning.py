import gym
from gym import spaces
import numpy as np

class CustomEnv(gym.Env):
    def __init__(self):
        self.grid_size = 5
        self.observation_space = spaces.Discrete(self.grid_size * self.grid_size)
        self.action_space = spaces.Discrete(4)  # 4 possible actions: up, down, left, right
        self.start_pos = (0, 0)
        self.goal_pos = (self.grid_size - 1, self.grid_size - 1)
        self.obstacle_pos = [(1, 1), (2, 2), (3, 3)]  # Example obstacle positions
        self.agent_pos = self.start_pos
        self.max_steps = 10
        self.current_step = 0

    def reset(self):
        self.agent_pos = self.start_pos
        self.current_step = 0
        return self._get_obs()

    def _get_obs(self):
        return self.agent_pos[0] * self.grid_size + self.agent_pos[1]

    def step(self, action):
        self.current_step += 1
        if action == 0:  # Up
            new_pos = (self.agent_pos[0] - 1, self.agent_pos[1])
        elif action == 1:  # Down
            new_pos = (self.agent_pos[0] + 1, self.agent_pos[1])
        elif action == 2:  # Left
            new_pos = (self.agent_pos[0], self.agent_pos[1] - 1)
        elif action == 3:  # Right
            new_pos = (self.agent_pos[0], self.agent_pos[1] + 1)

        if self._is_valid_move(new_pos):
            self.agent_pos = new_pos

        reward = self._calculate_reward()
        done = self.agent_pos == self.goal_pos or self.current_step >= self.max_steps
        return self._get_obs(), reward, done, {}

    def _is_valid_move(self, pos):
        if pos[0] < 0 or pos[0] >= self.grid_size or pos[1] < 0 or pos[1] >= self.grid_size:
            return False
        if pos in self.obstacle_pos:
            return False
        return True

    def _calculate_reward(self):
        if self.agent_pos == self.goal_pos:
            return 1
        else:
            return -0.1

    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size))
        grid[self.start_pos] = 0.5
        grid[self.goal_pos] = 0.5
        for pos in self.obstacle_pos:
            grid[pos] = -1
        grid[self.agent_pos] = 1
        print(grid)

# Q-learning algorithm
def q_learning(env, num_episodes=1000, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore action space
            else:
                action = np.argmax(q_table[state])  # Exploit learned values

            next_state, reward, done, _ = env.step(action)

            # Q-learning update rule
            best_next_action = np.argmax(q_table[next_state])
            q_table[state, action] += learning_rate * (reward + discount_factor * q_table[next_state, best_next_action] - q_table[state, action])

            state = next_state

    return q_table

# Main function
if __name__ == "__main__":
    env = CustomEnv()
    q_table = q_learning(env)

    # Testing the learned policy
    state = env.reset()
    done = False
    while not done:
        action = np.argmax(q_table[state])
        next_state, _, done, _ = env.step(action)
        env.render()
        state = next_state
