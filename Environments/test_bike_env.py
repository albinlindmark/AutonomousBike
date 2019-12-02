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
import scipy.linalg

def dlqr(A,B,Q,R):
    """Solve the discrete time lqr controller.
    
    x[k+1] = A x[k] + B u[k]
    
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    #ref Bertsekas, p.151
    
    #first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
    
    #compute the LQR gain
    K = np.matrix(scipy.linalg.inv(B.T*X*B+R)*(B.T*X*A))
    
    eigVals, eigVecs = scipy.linalg.eig(A-B*K)
    
    return K, X, eigVals


A = np.array([[1.015144907891091, 0.070671622176451], [0.431844962338814, 1.015144907891091]], dtype=np.float32)
B_k_wo_v = np.array([0.036491277663333, 0.047719661231268], dtype=np.float32)

env = gym.make("BikeLQR-v0")
state = env.reset()
init_state = state.copy()
v = state[2]

B_k = B_k_wo_v * np.array([v, v**2], dtype=np.float32)
B_k = B_k.reshape((2,1))
#K = np.array([2.153841222845478, 0.298295153292258], dtype=np.float32)
Q = np.array([[10, 0], [0, 0]])
R = 1
K, X, eigVals = dlqr(A,B_k,Q,R)
K = np.squeeze(np.asarray(K))

Ts = 0.04
nr_time_steps = 30
cumulative_reward = 0
phi_list = [state[0]*180/np.pi]
delta_list = []
for i in range(nr_time_steps):
    action = -K@state[0:2]
    state, reward, done, _ = env.step([action])
    print('reward:', reward)
    print('phi:', state[0].item()*180/np.pi)
    #state, reward, done, _ = env.step(np.array([0], dtype=np.float32))
    phi_list.append(state[0].item()*180/np.pi)
    delta_list.append(np.rad2deg(action))
    
    #cumulative_reward += reward
    cumulative_reward += reward
    print('Cumulative reward', cumulative_reward)
    print('Done:', done)
    
    if done:
        break


#t = np.linspace(0, len(phi_list)*Ts, len(phi_list))
t = np.arange(0, len(phi_list)*Ts, Ts)
fig, axes = plt.subplots(figsize=(14,8))
axes.plot(t, phi_list)
#axes.plot(t[1:], delta_list)
axes.set_title('v = ' + str(v))

plt.show()