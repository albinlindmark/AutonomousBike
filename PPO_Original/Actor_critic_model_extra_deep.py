import torch
import torch.nn as nn
class ActorCriticXDeep(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, sigma = 0.0):
        super(ActorCriticXDeep,self).__init__()
        self.Actor = nn.Sequential(nn.Linear(state_dim, hidden_dim//2, bias = True),
        nn.ReLU(), nn.Linear(hidden_dim//2, hidden_dim//4, bias = True),
        nn.ReLU(), nn.Linear(hidden_dim//4, hidden_dim//8, bias = True),
        nn.ReLU(), nn.Linear(hidden_dim//8, hidden_dim//16, bias = True),
        nn.ReLU(), nn.Linear(hidden_dim//16, hidden_dim//32, bias = True),
        nn.ReLU(), nn.Linear(hidden_dim//32,action_dim, bias = True))

        self.Critic = nn.Sequential(nn.Linear(state_dim, hidden_dim//2, bias = True),
        nn.ReLU(), nn.Linear(hidden_dim//2, hidden_dim//4, bias = True),
        nn.ReLU(), nn.Linear(hidden_dim//4, hidden_dim//8, bias = True),
        nn.ReLU(), nn.Linear(hidden_dim//8, hidden_dim//16, bias = True),
        nn.ReLU(), nn.Linear(hidden_dim//16, hidden_dim//32, bias = True),
        nn.ReLU(), nn.Linear(hidden_dim//32,1, bias = True))
        self.ln_sigma = nn.Parameter(torch.ones(action_dim)*sigma)
    def forward(self, states):
        V = self.Critic(states)
        mu = self.Actor(states)
        sigma = self.ln_sigma.exp().expand_as(mu)
        distribution = torch.distributions.Normal(mu,sigma)
        return distribution, V