import numpy as np 
import torch
def optimize_ppo(model,optimizer,states,actions,ln_probs,returns, values, K, epsilon, c1, c2, mb_size, losses):

    def generate_minibatch(states,actions,ln_probs,returns,values):
        returns   = torch.cat(returns).detach()
        ln_probs = torch.cat(ln_probs).detach()
        values    = torch.cat(values).detach()
        states    = torch.cat(states)
        actions   = torch.cat(actions)
        advantages = returns - values
        advantages -= advantages.mean()
        advantages /= (advantages.std() + 1e-8)
        
        index = np.arange(states.shape[0])
        np.random.shuffle(index)

        mb_states = states[index,:]
        mb_states = torch.split(mb_states,mb_size)

        mb_actions = actions[index,:]
        mb_actions = torch.split(mb_actions,mb_size)

        mb_ln_probs = ln_probs[index,:]
        mb_ln_probs = torch.split(mb_ln_probs,mb_size)

        mb_returns = returns[index,:]
        mb_returns = torch.split(mb_returns,mb_size)

        mb_advantages = advantages[index,:]
        mb_advantages = torch.split(mb_advantages,mb_size)

        chunks = states.size(0)//mb_size
        return mb_states, mb_actions, mb_ln_probs, mb_returns, mb_advantages, chunks

    for _ in range(K):

        mb_states, mb_actions, mb_ln_probs, mb_returns, mb_advantages, chunks = generate_minibatch(states,actions,ln_probs,returns,values)

        for chunk in range(chunks):
            _states = mb_states[chunk]
            _actions = mb_actions[chunk]
            _ln_probs = mb_ln_probs[chunk]
            _returns = mb_returns[chunk]
            _advantages = mb_advantages[chunk]

            policy , value = model(_states)
            entropy = policy.entropy().mean()
            new_ln_probs = policy.log_prob(_actions)

            ratio = (new_ln_probs - _ln_probs).exp()
            surrogate_1 = ratio*_advantages
            surrogate_2 = torch.clamp(ratio, 1.0-epsilon,1.0+epsilon)*_advantages
            L_clip = torch.min(surrogate_1,surrogate_2).mean()
            L_vf = (_returns-value).pow(2).mean()

            loss = -L_clip + c1*L_vf - c2*entropy
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()