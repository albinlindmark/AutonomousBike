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
nr_time_steps = 100

initial_v_to_test = [0.75, 5.5, 9.5]
initial_roll_angles_to_test = [np.deg2rad(5), np.deg2rad(-5)]
nr_of_sequences = len(initial_v_to_test) * len(initial_roll_angles_to_test)
idx = 0
cumulative_rewards = np.zeros(nr_of_sequences)
for v_0 in initial_v_to_test:
    for phi_0 in initial_roll_angles_to_test:
        init_state = np.array([phi_0, 0, v_0], dtype=np.float32)
        state = env.reset(init_state=init_state)
        B_k = B_k_wo_v * np.array([v_0, v_0**2], dtype=np.float32)
        B_k = B_k.reshape((2,1))
        K, X, eigVals = dlqr(A,B_k,Q,R)
        K = np.squeeze(np.asarray(K))
        for j in range(nr_time_steps):
            state, reward, done, _ = env.step([-K@state[0:2]])
            cumulative_rewards[idx] += reward
            if done:
                break
        idx += 1

mean_of_cumulative_rewards = cumulative_rewards.mean()
print('Average comulative reward: {:.6f}'.format(mean_of_cumulative_rewards))