import gym
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
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

#K = np.array([2.153841222845478, 0.298295153292258], dtype=np.float32)
Q = np.array([[10, 0], [0, 0]])
R = 1

Ts = 0.04
nr_time_steps = 30

cumulative_reward = np.zeros(64)
for i in range(len(cumulative_reward)):
    state = env.reset()
    v = state[2]
    B_k = B_k_wo_v * np.array([v, v**2], dtype=np.float32)
    B_k = B_k.reshape((2,1))
    K, X, eigVals = dlqr(A,B_k,Q,R)
    K = np.squeeze(np.asarray(K))
    for j in range(nr_time_steps):
        state, reward, done, _ = env.step([-K@state[0:2]])
        cumulative_reward[i] += reward
        
        if done:
            break


print('Average comulative reward: {:.6f}'.format(np.mean(cumulative_reward)))