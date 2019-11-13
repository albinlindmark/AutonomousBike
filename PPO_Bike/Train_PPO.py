import torch
import gym
import Compute_GAE
import Run_policy
import Actor_critic_model
import Actor_critic_model_deep
import Actor_critic_model_extra_deep
import PPO_optim
import Make_MP_envs
import Test_PPO
import Save_metrics

#PPO parameters
env_id = 'BikeLRQ-v0'
num_envs = 64
hidden_size = 256
learning_rate = 0.00005
finished = False
T = 2000
gamma_GAE = 0.99
lambda_GAE = 0.95
K = 10
epsilon = 0.2
c1 = 0.5
c2 = 0.001
mb_size = 64
episode_nr = 0
reward_threshold = 300

test_rewards = [-5000]
losses = []
test_length = 15
mean_losses = []
do_render = False
deep = 2
#Create envs, model and optimizer
env_list, state_dim, action_dim = Make_MP_envs.Make_envs(env_id,num_envs)
test_env = gym.make(env_id)
if deep == 0:
    model = Actor_critic_model.ActorCritic(state_dim,action_dim,hidden_size)
elif deep == 1:
    model = Actor_critic_model_deep.ActorCriticDeep(state_dim,action_dim,hidden_size)
elif deep == 2:
    print('Deep2')
    model = Actor_critic_model_extra_deep.ActorCriticXDeep(state_dim,action_dim,hidden_size)
optimizer = torch.optim.Adam(model.parameters(), learning_rate)
state = env_list.reset()
print('state')
#Algorithm: PPO, Actor critic style
while not finished:
    episode_nr += 1
    print('Episode: ',episode_nr)
    # PPO algorithm
    ln_probs, values, states, actions, rewards, masks = Run_policy.run_policy_old(env_list,model, T, state)
    returns = Compute_GAE.GAE(model,states,rewards,masks,values,gamma_GAE,lambda_GAE)
    PPO_optim.optimize_ppo(model,optimizer,states,actions,ln_probs,returns,values,K,epsilon,c1,c2,mb_size,losses)
    #Run test and save model metrics
    test_reward = [Test_PPO.run_test(test_env, model, do_render) for tests in range(test_length)]
    mean_test_reward = sum(test_reward)/len(test_reward)
    test_rewards.append(mean_test_reward)
    mean_loss = sum(losses[-num_envs:])/num_envs
    mean_losses.append(mean_loss)
    Save_metrics.save_model_metrics(model,test_rewards, episode_nr, mean_losses, deep)
    print('Mean loss for ',num_envs,' environments: ',mean_loss, 'Mean test reward over', test_length, ' runs:', mean_test_reward)
    if mean_test_reward >= reward_threshold:
        print('Test threshold reached, saving model paramenters.')
        finished = True
