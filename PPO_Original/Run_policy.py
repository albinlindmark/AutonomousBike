import torch
def run_policy_old(envs,model,T,state):
    ln_probs, values, states, actions, rewards, masks = [],[],[],[],[],[]
    for _ in range(T):
        state = torch.FloatTensor(state)
        policy, value = model(state)
        action = policy.sample()
        next_state, reward , terminal, _ = envs.step(action.numpy())
        ln_prob = policy.log_prob(action)
        values.append(value)
        reward = torch.FloatTensor(reward).unsqueeze(1) 
        rewards.append(reward)
        mask = torch.FloatTensor(1-terminal).unsqueeze(1)
        masks.append(mask)
        states.append(state)
        actions.append(action)
        ln_probs.append(ln_prob)
        state = next_state
    return ln_probs, values, states, actions, rewards, masks


