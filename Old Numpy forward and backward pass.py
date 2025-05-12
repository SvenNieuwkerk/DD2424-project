import numpy as np

def SoftMax(input):
    expInput = np.exp(input)
    return expInput / np.sum(expInput, axis=1, keepdims=True)

def forwardPassUsingNumpy(X,h0,c0, numpy_network):
    fp_data = {}
    fp_data['X'] = X #(tau x K)
    tau = X.shape[0]
    m = numpy_network['W'].shape[0]/4

    A = np.zeros((tau, 4 * m))
    # If we want to do it individually
    I = np.zeros((tau, m))  # (tau x m)
    F = np.zeros((tau, m))  # (tau x m)
    O = np.zeros((tau, m))  # (tau x m)
    C_wiggle = np.zeros((tau, m))  # (tau x m)
    C = np.zeros((tau + 1, m))  # ((tau+1) x m)
    H = np.zeros((tau + 1, m))  # ((tau+1) x m)

    XU = X@numpy_network['U'] #(tau x K)@(K x 4m) = (tau x 4m)
    #If we want to do it individually
    #XUi = X @ network['Ui']  # (tau x K)@(K x m) = (tau x m)
    #XUf = X @ network['Uf']  # (tau x K)@(K x m) = (tau x m)
    #XUo = X @ network['Uo']  # (tau x K)@(K x m) = (tau x m)
    #XUc = X @ network['Uc']  # (tau x K)@(K x m) = (tau x m)

    ht = h0 #(1 x m)
    ct = c0
    H[0,:] = ht
    C[0,:] = ct
    for t in range(tau):
        #We use b as bias, when taking gradient b for c_wiggle_t has different activation so different gradients
        at = ht@numpy_network['W'] + XU[t,:] + numpy_network['b'] #(1 x m)@(m x 4m) + (1 x m) + (1 x 4m) = (1 x 4m)
        A[t,:] = at

        ft = SoftMax(at@numpy_network['E1'])
        F[t,:] = ft
        it = SoftMax(at@numpy_network['E2'])
        I[t,:]=it
        ot = SoftMax(at@numpy_network['E3'])
        O[t,:] = ot
        c_wiggle_t = np.tanh(at@numpy_network['E4'])
        C_wiggle[t,:] = c_wiggle_t

        ct = ft*C[t]+ it*c_wiggle_t
        C[t+1,:] = ct

        ht = ot * np.tanh(ct)
        H[t+1,:] = ht

    fp_data['I'] = I
    fp_data['F'] = F
    fp_data['O'] = O
    fp_data['C'] = C
    fp_data['C_wiggle'] = C_wiggle
    fp_data['H'] = H

    Z = H[1:,:]@numpy_network['V']+ numpy_network['c'] #(tau x m) @ (m x K) + (1xK) = (tau x K)
    fp_data['Z'] = Z
    P = SoftMax(Z) #(tau x K)
    fp_data['P'] = P

    return fp_data

def backwardPassUsingNumpy(Y, fp_data, numpy_network):
    #Not finished, just for RNN
    grads = {}
    tau = fp_data['X'].shape[0]
    m = numpy_network['W'].shape[0]/4
    G = -(Y-fp_data['P']).T #((tau x K) - (tau x K)).T=(K x tau)
    grads['V']= (1 / tau) * fp_data['H'][1:,:].T@G.T #(tau x m).T @ (K x tau).T = (m, K)
    grads['c'] = (1 / tau) * np.ones((1, tau))@G.T #(1 x tau) @ (K x tau).T = (1, K)

    dLdHt_part_1 = numpy_network['V'] @ G  # (m x K)@(K x tau) = (m x tau)
    dHtdAt = 1-fp_data['H'][1:,]**2 #(tau x m)

    dLdat = dHtdAt[tau-1,:]*dLdHt_part_1[:,tau-1] #(m,)*(m,) = (m,) #Only for tau
    GNew = np.zeros((tau,m)) #(tau x m)
    GNew[tau-1,:] = dLdat #(1 x m)
    for t in range(tau - 1, 0, -1):
        dLdht = dLdHt_part_1[:,t-1] + numpy_network['W']@dLdat #(m x 1) + (m x m)@(m,) = (m x 1)
        dLdat = dHtdAt[t-1,:]*dLdht #(m,)*(m,) = (m,)
        GNew[t-1,:] = dLdat #(1 x m)
    G= GNew

    grads['W'] = (1 / tau) *fp_data['H'][:tau,:].T@G #(tau x m).T @ (tau x m) = (m x m)
    grads['U'] = (1 / tau) *fp_data['X'].T@G #(tau x K).T @ (tau x m) = (K x m)

    grads['b'] = (1 / tau) * np.ones((1, tau))@G #(1 x tau) @ (tau x m) = (1 x m)

    return grads