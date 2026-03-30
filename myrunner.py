'''
Training / Self-play Loop
- Create environment 
- Create agent/model 
- Run episodes 
- Alternate through PettingZoo turns 
- Train via self-play
- Maybe save checkpoints / print rewards 
'''

import torch
import torch.optim as optim 
import numpy as np 

from mycheckersenv import env
from myagent import ActorCritic, select_action 

def train_self_play(num_episodes=500, gamma=0.99, lr=1e-3):
    '''
    Training loop for training the Actor-Critic agent using self-play.

    Prints out: 
    - Reward per episode 
    - Winner per episode 
    - Final cumulative reward 
    '''
    environment = env(render_mode="human") # No board spam
    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    #episode_returns = []
    p0_rewards = []
    p1_rewards = []

    #cumulative_reward = 0.0 # total reward across all episodes 

    for episode in range(num_episodes):
        print(f"\n=== Episode {episode + 1} ===")
        environment.reset()
        episode_reward_p0 = 0.0
        episode_reward_p1 = 0.0
        #episode_reward = 0.0

        for agent in environment.agent_iter(max_iter=500):

            # Get las step info
            obs, reward, termination, truncation, info = environment.last()
            
            # Update reward for this episode
            #episode_reward += reward 
            #Track rewards separately
            if agent =="player_0":
                episode_reward_p0 += reward 
            else:
                episode_reward_p1 += reward

            # If game ended, pass None action 
            if termination or truncation:
                environment.step(None)
                continue
            
            # Select action using policy
            action, log_prob, value = select_action(model, obs)
            
            # Apply action
            environment.step(action)
            
            # Get next state
            next_obs, next_reward, next_termination, next_truncation, _ = environment.last()

            # Compute TD target 
            if next_termination or next_truncation:
                target = torch.tensor(next_reward, dtype=torch.float32)
            else:
                next_state = torch.tensor(
                    next_obs["Observation"].flatten(), dtype=torch.float32
                ).unsqueeze(0)
                with torch.no_grad():
                    _, next_value = model(next_state)
                target = torch.tensor(next_reward, dtype=torch.float32) + gamma * next_value.squeeze()
            
            # TD error
            advantage = target - value

            # Actor loss (policy gradient)
            actor_loss = -log_prob * advantage.detach()

            # Critic loss (value regression)
            critic_loss = advantage.pow(2)

            # Total loss
            loss = actor_loss + critic_loss 

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #episode_returns.append(episode_reward_p0)
        #episode_returns.append(episode_reward_p1)
        #cumulative_reward += episode_reward
        p0_rewards.append(episode_reward_p0)
        p1_rewards.append(episode_reward_p1)
        
        if environment.engine.winner is not None:
            print(f"Episode {episode + 1}: winner = player_{environment.engine.winner}, turns = {environment.turn_count}")
        else:
            print(f"Episode {episode + 1}: truncated, turns = {environment.turn_count}")
        
        '''
        Episode reward will likely be 0.000 since player_0 gets +1
        and player_1 gets -1 so total reward will be 0.
        '''
        print(f"Player_0 reward: {episode_reward_p0:.3f}")
        print(f"Player_1 reward: {episode_reward_p1:.3f}")

        # =========================
        # Final summary
        # =========================
        print("\n==============================")
        print("TRAINING COMPLETE")
        print("==============================")
        print(f"Average player_0 reward: {np.mean(p0_rewards):.3f}")
        print(f"Average player_1 reward: {np.mean(p1_rewards):.3f}")
        
        '''
        if (episode+1) % 50 == 0:
            avg_return = np.mean(episode_returns[-50:])
            print(f"Episode {episode + 1}, avg return: {avg_return:.3f}")
        '''
    return model, p0_rewards, p1_rewards

if __name__ == "__main__":
    train_self_play(num_episodes=10)
