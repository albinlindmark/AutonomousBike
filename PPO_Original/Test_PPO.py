import torch
import gym
def run_test(env,model, do_render):
    state = env.reset()
    terminal = False
    rewards = 0
    while not terminal:
        state = torch.FloatTensor(state)
        policy, value = model(state)
        action = policy.mean.detach()
        next_state, reward , terminal, _ = env.step(action.numpy())
        if do_render:
            env.render()
        rewards += reward
        state = next_state
    if do_render:
        env.env.close()
    return rewards