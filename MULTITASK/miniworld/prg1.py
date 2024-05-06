import gym
import gym_miniworld

def run_miniworld():
    # Create the MiniWorld environment
    env = gym.make('MiniWorld-Hallway-v0')

    # Reset the environment to start
    obs = env.reset()

    # Run for a few steps
    for _ in range(100):
        # Render the current state to the screen
        env.render()

        # Take a random action
        action = env.action_space.sample()

        # Apply the action
        obs, reward, done, info = env.step(action)

        # If the episode is done, reset the environment
        if done:
            obs = env.reset()

    # Close the environment
    env.close()

if __name__ == "__main__":
    run_miniworld()


import cv2  # Import OpenCV

def save_frames_as_video():
    env = gym.make('MiniWorld-Hallway-v0')
    obs = env.reset()
    frames = []

    for _ in range(100):
        frame = env.render(mode='rgb_array')  # Get frame in RGB array format
        frames.append(frame)

        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()

    env.close()

    # Save frames as video
    height, width, layers = frames[0].shape
    video = cv2.VideoWriter('miniworld_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))

    for frame in frames:
        video.write(frame)

    video.release()

if __name__ == "__main__":
    save_frames_as_video()
