import scipy.linalg
import numpy as np

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

def get_optimal_sequence(init_state, env):
    
    v = init_state[2]
    A = env.A
    B_k_wo_v = env.B_k_wo_v
    Q = env.Q
    R = env.R

    B_k = B_k_wo_v * np.array([v, v**2], dtype=np.float32)
    K, X, eigVals = dlqr(A, B_k.reshape((2,1)), Q, R)
    K = np.squeeze(np.asarray(K))
    
    phi_list = [init_state[0]]
    delta_list = []
    done = False
    state = env.reset(init_state=init_state)
    while not done:
        action = -K@state[0:2]
        state, reward, done, _ = env.step([action])
        phi_list.append(state[0].item())
        delta_list.append(action)
    
    return phi_list, delta_list
