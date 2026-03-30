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
    environment = env(render_mode="human") # No board spam
    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    episode_returns = []

    for episode in range(num_episodes):
        print(f"\n=== Episode {episode + 1} ===")
        environment.reset()
        episode_reward = 0.0
        
        '''
        last_value = None
        last_log_prob = None 
        last_agent = None 
        '''

        for agent in environment.agent_iter(max_iter=500):
            obs, reward, termination, truncation, info = environment.last()

            episode_reward += reward 

            if termination or truncation:
                environment.step(None)
                continue
            
            action, log_prob, value = select_action(model, obs)

            environment.step(action)

            next_obs, next_reward, next_termination, next_truncation, _ = environment.last()

            if next_termination or next_truncation:
                target = torch.tensor(next_reward, dtype=torch.float32)
            else:
                next_state = torch.tensor(
                    next_obs["Observation"].flatten(), dtype=torch.float32
                ).unsqueeze(0)
                with torch.no_grad():
                    _, next_value = model(next_state)
                target = torch.tensor(next_reward, dtype=torch.float32) + gamma * next_value.squeeze()
            
            advantage = target - value

            actor_loss = -log_prob * advantage.detach()
            critic_loss = advantage.pow(2)
            loss = actor_loss + critic_loss 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        episode_returns.append(episode_reward)
        #print(f"Episode {episode + 1} return: {episode_reward:.3f}")
        if environment.engine.winner is not None:
            print(f"Episode {episode + 1}: winner = player_{environment.engine.winner}, turns = {environment.turn_count}")
        else:
            print(f"Episode {episode + 1}: truncated, turns = {environment.turn_count}")

        if (episode+1) % 50 == 0:
            avg_return = np.mean(episode_returns[-50:])
            print(f"Episode {episode + 1}, avg return: {avg_return:.3f}")
    return model, episode_returns

if __name__ == "__main__":
    train_self_play(num_episodes=10)
