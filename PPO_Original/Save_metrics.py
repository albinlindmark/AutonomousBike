import torch
def save_model_metrics(model, test_rewards, episodes, loss, deep):
    if test_rewards[-1] >= max(test_rewards[:-1]):
        if deep == 0:
            file_name = '/home/jacob/Documents/LunarLander/AlbinJacob/Saves_PPO/model_params_' + str(test_rewards[-1])
        elif deep == 1:
            file_name = '/home/jacob/Documents/LunarLander/AlbinJacob/Saves_PPO/model_params_' + str(test_rewards[-1])  + '_DEEP'
        elif deep == 2:
            file_name = '/home/jacob/Documents/LunarLander/AlbinJacob/Saves_PPO/model_params_' + str(test_rewards[-1])  + '_EXTRA_DEEP'
        torch.save({'episodes' : episodes,'model_state_dict' : model.state_dict(),'loss' : loss,'test_rewards' : test_rewards}, file_name)
        print('The best reward: ',max(test_rewards[:-1]),' was replaced by: ', test_rewards[-1])
        

