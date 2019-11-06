import torch
import torch.nn as nn
class ActorCriticDeep(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, sigma = 0.0):
        super(ActorCriticDeep,self).__init__()
        self.Actor = nn.Sequential(nn.Linear(state_dim, hidden_dim, bias = True),
        nn.ReLU(), nn.Linear(hidden_dim, hidden_dim//2, bias = True),
        nn.ReLU(), nn.Linear(hidden_dim//2,action_dim, bias = True))
        self.Critic = nn.Sequential(nn.Linear(state_dim,hidden_dim, bias=True),
        nn.ReLU(),nn.Linear(hidden_dim,hidden_dim//2, bias=True),
        nn.ReLU(),nn.Linear(hidden_dim//2, 1, bias = True))
        self.ln_sigma = nn.Parameter(torch.ones(action_dim)*sigma)
    def forward(self, states):
        V = self.Critic(states)
        mu = self.Actor(states)
        sigma = self.ln_sigma.exp().expand_as(mu)
        distribution = torch.distributions.Normal(mu,sigma)
        return distribution, V