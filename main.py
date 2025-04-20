import pygame
import matplotlib.pyplot as plt
import numpy as np

from game_environment import MainEnv, screen
from agent import Agent

NO_OF_EPISODES = 5000

def main():
    clock = pygame.time.Clock()
    env = MainEnv()
    agent = Agent((2, 20, 20), 4)
    reward_history = []

    for episode in range(NO_OF_EPISODES):
        done = False
        state = env.frame_stacker.get_stacked_frames()
        score = 0
        steps = 0
        total_reward = 0
        
        while not done:
            # Clear the screen with a background color.
            screen.fill((84, 174, 204))
            # Draw the game elements (fruit and snake).
            env.draw_elements(screen)
            pygame.display.update()
            
            # Agent picks an action.
            action = agent.choose_action(state)
            next_state, reward, done = env.play_step(action, screen)
            total_reward += reward
            score += int(reward / 10)  # Convert reward to an approximate score.
            
            agent.store_in_memory(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            steps += 1
            clock.tick(60)  # Limit the game to 60 FPS
            
        reward_history.append(total_reward)
        print(f"Episode {episode + 1} Score: {score+1} Steps: {steps} Epsilon: {agent.epsilon:.3f}")
        
        if episode % 200 == 0 and episode > 0:
            model_path = f"model_{episode}.pth"
            agent.save_model(model_path)
            plt.figure(figsize=(12,5))
            plt.subplot(1,2,1)
            plt.plot(np.arange(len(agent.losses)), agent.losses)
            plt.title("Loss Over Training Steps")
            plt.xlabel("Training Step")
            plt.ylabel("Loss")
            plt.subplot(1,2,2)
            plt.plot(np.arange(len(reward_history)), reward_history)
            plt.title("Reward Over Episodes")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.tight_layout()
            plt.show()
    
    pygame.quit()

if __name__ == '__main__':
    main()
