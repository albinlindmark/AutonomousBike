import gym
import multiprocessing_env
def Make_envs(env_id, num_envs):
    def make_environment():
    # returns a function which creates a single environment
        def function_handle():
            env = gym.make(env_id)
            return env
        return function_handle
    env_list = [make_environment() for i in range(num_envs)]
    env_list = multiprocessing_env.SubprocVecEnv(env_list)
    state_dim = env_list.observation_space.shape[0]
    action_dim = env_list.action_space.shape[0]
    return env_list, state_dim, action_dim