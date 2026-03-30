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
    """
    Actor-Critic neural network.
    
    Structure: 
    - Shared layers: extract features from the board state 
    - Actor head: outputs logits for all possible actions 
    - Critic head: outputs value estimate V(s)

    Input: 
        state (flattened 6x6 board -> size 36)
    
    Outputs:
        logits -> used to compute action probabilities 
        value -> estimated value of current state 
    """
    def __init__(self, state_dim=36, action_dim=1296, hidden_dim=128):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor: outputs action logits (before softmax)
        self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic: outputs scalar value estimate V(s)
        self.critic = nn.Linear(hidden_dim, 1)
   
    def forward(self, x):
        '''
        Forward pass through the network.
        Parameters:
            x: tensor of shape (batch_size, state_dim)
        Return values:
            logits: raw action scores 
            value: estimated value of the state 
        '''
        x = self.shared(x)
        logits = self.actor(x) # action logits
        value = self.critic(x) # state value
        return logits, value
    
def masked_action_distribution(logits, action_mask):
    '''
    Applies action masking to ensure illegal moves are never selected. 
    Logic: 
        - Illegal actions which have a mask equal to 0 get very large negative logits 
        - This makes their probability around 0 after softmax 
    Parameters:
        - logits: raw action logits from actor 
        - action_mask: binary mask (1 = legal, 0 = illegal)
    Returns: 
        torch.distributions.Categorical distribution over valid actions
    '''
    mask = action_mask.float()
    # Set illegal actions to -inf (approx with -1e9)
    masked_logits = logits.masked_fill(mask == 0, -1e9)
    return torch.distributions.Categorical(logits=masked_logits)

def select_action(model, obs):
    '''
    Selects an action using the policy (Actor). 
    The function follows the steps: 
    - Convert observation to tensor 
    - Pass through nthe neural network -> get logits + value 
    - Apply the action mask 
    - Sample action from distribution 
    - Return action, log_prob, and value 

    Parameters: 
        model: ActorCritic network 
        obs: Observation dictionary from environment 
    Returns:
        action (int): chosen action 
        log_prob (tensor): log probability of action 
        value (tensor): Estimated state value 
    '''
    # Flatten board state (6x6 -> 36)
    state = torch.tensor(obs["Observation"].flatten(), dtype=torch.float32).unsqueeze(0)
    
    # Action mask (legal moves only)
    action_mask = torch.tensor(obs["action_mask"], dtype=torch.float32).unsqueeze(0)
    
    #Forward pass
    logits, value = model(state)
    
    # Apply mask and create distribution 
    dist = masked_action_distribution(logits, action_mask)
    
    # Sample action from policy
    action = dist.sample()
    
    # Log probability (used for policy gradient update)
    log_prob = dist.log_prob(action)

    return action.item(), log_prob.squeeze(), value.squeeze()

    