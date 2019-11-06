import torch
def GAE(model,states,rewards,masks,values,gamma_GAE,lambda_GAE):
    next_state = torch.FloatTensor(states[-1])
    _, next_value = model(next_state)
    values = values + [next_value]
    PPO_steps = len(states)
    returns = []
    A_t = 0
    advantages = []
    for t in reversed(range(PPO_steps)): #Loop backwards to fill up advantage estimate
        delta = rewards[t] + gamma_GAE*values[t + 1]*masks[t] - values[t]
        A_t = delta + gamma_GAE*lambda_GAE*masks[t]*A_t       
        returns.append(A_t + values[t])
        
    returns.reverse()
    return returns
