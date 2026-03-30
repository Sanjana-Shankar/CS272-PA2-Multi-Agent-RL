'''
Task 2 model: 
- Actor-Critic neural network 
- Masked action-selection 
- Value estimation 
- Update logic 

Contents: 
- ActorCritic class 
- Helper function for masked softmax 
- Helper function to choose action 
- Helper function to compute/update losses 
File should NOT contain environment rules 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):

    def __init__(self, state_dim=36, action_dim=1296, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
   
    def forward(self, x):
        x = self.shared(x)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value
    
def masked_action_distribution(logits, action_mask):
    mask = action_mask.float()
    masked_logits = logits.masked_fill(mask == 0, -1e9)
    return torch.distributions.Categorical(logits=masked_logits)

def select_action(model, obs):
    state = torch.tensor(obs["Observation"].flatten(), dtype=torch.float32).unsqueeze(0)
    action_mask = torch.tensor(obs["action_mask"], dtype=torch.float32).unsqueeze(0)

    logits, value = model(state)
    dist = masked_action_distribution(logits, action_mask)
    action = dist.sample()
    log_prob = dist.log_prob(action)

    return action.item(), log_prob.squeeze(), value.squeeze()

    