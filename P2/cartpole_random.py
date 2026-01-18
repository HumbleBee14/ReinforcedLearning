import gymnasium as gym
import time

env = gym.make("CartPole-v1", render_mode="human")
state, info = env.reset()

print("Starting Random Agent...")
episode_reward = 0
done = False

while not done:
    env.render()
    
    # Pick a Random Action (0=Left, 1=Right)
    action = env.action_space.sample()
    
    # Take the Step
    next_state, reward, terminated, truncated, info = env.step(action)
    
    episode_reward += reward
    done = terminated or truncated
    
    # Slowing it down slightly so its easier for us to see what's happening
    time.sleep(0.05)

print(f"Game Over! Total Score: {episode_reward}")
env.close()