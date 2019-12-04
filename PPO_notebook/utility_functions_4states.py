import scipy.linalg
import numpy as np
import pandas as pd
Gains = pd.read_excel('Klist_xls.xls', header=None).values

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

def get_K(v):
    #Get K-values for the given v using interpolation
    v_table = Gains[:,0]
    gains_col1 = Gains[:,1]
    gains_col2 = Gains[:, 2]

    K1 = np.interp(v, v_table, gains_col1)
    K2 = np.interp(v, v_table, gains_col2)
    K_interp = np.array([K1, K2], dtype=np.float32)

    # index = np.argwhere(np.isclose(Gains[:,0], round(v, 1))).item()
    # K = Gains[index,1:3]
    
    return K_interp

def get_optimal_sequence(init_state, env, changing_speed = False):
    
    v = init_state[2]
    A = env.A
    Q = env.Q
    R = env.R
    B_c_wo_v = env.B_c_wo_v
    inv_Ac = env.inv_Ac

    
    B_c = B_c_wo_v * np.array([v, v**2], dtype=np.float32)
    B_k = inv_Ac @ (A - np.eye(2)) @ B_c
    K, X, eigVals = dlqr(A, B_k.reshape((2,1)), Q, R)
    K = np.squeeze(np.asarray(K))
    
    phi_list = [init_state[0]]
    delta_list = [init_state[3]]
    done = False
    state = env.reset(init_state=init_state, changing_speed = changing_speed)
    while not done:
        
        if changing_speed:
            v = state[2]
            K = get_K(v)
        
        action = -K@state[0:2]
        state, reward, done, _ = env.step([action])
        phi_list.append(state[0].item())
        delta_list.append(state[3])
    
    return phi_list, delta_list
