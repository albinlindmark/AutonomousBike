''' Simple script to test if the BikeLQR environment seems to work as intended.
The LQR gain K for the speed v = 5 m/s (the same speed used for to calculate the state space
matrices in the BikeLQR environment) is gotten from the matlab script LQR_measure_all_no_kalman.
So this python script uses the -K*x as action to the environment and then plots
how phi changes with this action. Then you can compare this plot to the simulation
plot in matlab/simulink. The cumulative reward is also printed for each time step,
which can be compared to the total cost in the matlab/simulink simulation. '''

import gym
import numpy as np
import matplotlib.pyplot as plt

K = np.array([2.153841222845478, 0.298295153292258], dtype=np.float32)
Ts = 0.04
nr_time_steps = 50

env = gym.make("BikeLQR-v0")
state = env.reset()
print('state_0', state)
cumulative_reward = 0
phi_list = [state[0]*180/np.pi]
for i in range(nr_time_steps):
    state, reward, done, _ = env.step([-K@state])
    #state, reward, done, _ = env.step(np.array([0], dtype=np.float32))
    phi_list.append(state[0].item()*180/np.pi)
    
    cumulative_reward += reward
    print('Cumulative reward', cumulative_reward)
    print('Done:', done)
    
    if done:
        break


t = np.linspace(0, len(phi_list)*Ts, len(phi_list))
fig, axes = plt.subplots(figsize=(14,8))
axes.plot(t, phi_list)