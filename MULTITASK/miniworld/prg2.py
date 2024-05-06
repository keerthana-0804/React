import gym
import gym_miniworld
import cv2

def heuristic_policy(env):
    """
    Simple heuristic:
    - Move forward if there's no obstacle directly in front.
    - Turn left or right otherwise.
    """
    # Simplified: always tries to move forward if possible
    forward_obs = env.mini_obs(10)  # Check for obstacles in a small distance ahead
    if forward_obs is None or forward_obs > 0.1:  # No obstacle closer than threshold
        return env.actions.move_forward
    else:
        # Randomly choose to turn left or right when an obstacle is detected
        return env.actions.turn_left if np.random.rand() > 0.5 else env.actions.turn_right

def run_miniworld():
    env = gym.make('MiniWorld-Hallway-v0')
    obs = env.reset()

    frames = []  # To store frames for video

    while True:
        env.render()

        # Apply heuristic policy instead of random actions
        action = heuristic_policy(env)
        obs, reward, done, info = env.step(action)

        # Save frames for video
        frames.append(env.render(mode='rgb_array'))

        if done:
            print("Reached the goal!")
            break

    env.close()

    # Save frames as video
    height, width, layers = frames[0].shape
    video = cv2.VideoWriter('miniworld_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))

    for frame in frames:
        video.write(frame)

    video.release()
    print("Video saved as 'miniworld_video.avi'")

if __name__ == "__main__":
    run_miniworld()
