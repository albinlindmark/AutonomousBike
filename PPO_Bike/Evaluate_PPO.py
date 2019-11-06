import torch
import gym
import Actor_critic_model
import Actor_critic_model_deep
import Actor_critic_model_extra_deep
import Test_PPO
import os
import matplotlib.pyplot as plt

have_path = False
#file_path = '/home/jacob/Documents/LunarLander/AlbinJacob/Saves_PPO/model_params_253.50440692121083' #Vanilla model with 1 layer
#file_path = '/home/jacob/Documents/LunarLander/AlbinJacob/Saves_PPO/model_params_251.6410028433216_DEEP' #"Deeper" model with 2 layers
#file_path = '/home/jacob/Documents/LunarLander/AlbinJacob/Saves_PPO/model_params_253.52047213033865_EXTRA_DEEP' #Even deeper model, with 5 layers
file_path = 'model_params_294.9360994385101_EXTRA_DEEP' #Deeper model, with 5 layers, run longer
env = gym.make('LunarLanderContinuous-v2')

do_render = True
hidden_size = 256
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

if have_path: 
    path = '/home/jacob/Documents/LunarLander/AlbinJacob/Saves_PPO/'
    file_list = os.listdir(path)
    file_list.sort()
    file_path = path + file_list[-1]
    print('Opening file path: ',file_path)
#model = Actor_critic_model.ActorCritic(state_dim,action_dim,hidden_size)
#model = Actor_critic_model_deep.ActorCriticDeep(state_dim,action_dim,hidden_size)
model = Actor_critic_model_extra_deep.ActorCriticXDeep(state_dim,action_dim,hidden_size)
print(model)
checkpoint = torch.load(file_path)
model.load_state_dict(checkpoint['model_state_dict'])
episodes = checkpoint['episodes']
loss = checkpoint['loss']
test_rewards = checkpoint['test_rewards']
test_rewards = test_rewards[1:]
#plots
max_mean_test_reward = round(max(test_rewards),2)
fig, ax = plt.subplots()
line, = ax.plot(test_rewards)
bbox_args = dict(boxstyle="round", fc="0.8")
arrow_args = dict(arrowstyle="->", connectionstyle="arc,angleA=0,armA=40,angleB=-90,armB=30,rad=7")
ax.annotate('Best reward: ' + str(max_mean_test_reward) , xy=(len(test_rewards)-1,max_mean_test_reward+1), xycoords='data', xytext=(-20, -100), textcoords='offset points', ha="right", va="top", bbox=bbox_args, arrowprops=arrow_args)
ax.set_ylabel('Mean test reward')
ax.set_xlabel('Episodes')
ax.set_title('Actor Critic with PPO Algorithm for Lunar Lander Continous')
ax.legend(['Mean test reward per episode'], loc = 'lower right')
fig.savefig('mean_reward_history.png')
plt.show()

plt.figure()
plt.plot(loss)
plt.ylabel('Mean loss over all environments')
plt.xlabel('Episodes')
plt.show()


#Run model
for i in range(20):
    #vid_name = "./vid" + str(i)
    #env = gym.wrappers.Monitor(env, vid_name,force=True) #To record
    reward = Test_PPO.run_test(env,model,do_render)
    print('Reward: ', reward)
