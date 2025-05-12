import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from torch_gradient_computations_row_wise import ComputeGradsWithTorch
import copy
import matplotlib.pyplot as plt

def read_in_data(seq_length = -1):
    book_dir = r'C:\Users\svenr\OneDrive - KTH\Deep Learning in Data Science\Datasets\\'
    book_fname = book_dir + 'goblet_book.txt'
    fid = open(book_fname, "r")
    book_data = fid.read()
    fid.close()

    unique_chars = list(set(book_data))
    K = len(unique_chars)

    char_to_ind = {ch: i for i, ch in enumerate(unique_chars)}
    ind_to_char = {i: ch for i, ch in enumerate(unique_chars)}

    return book_data, char_to_ind, ind_to_char, K

def initializeRNG():
    rng = np.random.default_rng()
    # get the BitGenerator used by default rng
    BitGen = type(rng.bit_generator)
    # use the state from a fresh bit generator
    seed = 42
    rng.bit_generator.state = BitGen(seed).state
    return rng

def initialiseRNN(K,m,rng):
    RNN={}
    RNN['b'] = np.zeros((1,m))
    RNN['c'] = np.zeros((1,K))
    RNN['U'] = (1 / np.sqrt(2 * K)) * rng.standard_normal(size=(K, m))
    RNN['W'] = (1 / np.sqrt(2 * m)) * rng.standard_normal(size=(m, m))
    RNN['V'] = (1 / np.sqrt(m)) * rng.standard_normal(size=(m, K))
    return RNN

def SoftMax(input):
    expInput = np.exp(input)
    return expInput / np.sum(expInput, axis=1, keepdims=True)

def synthesiseTextFromRNN(h0,n, RNN, rng):
    K = RNN['V'].shape[1]

    start_char_ind = np.random.randint(0, K)

    x0 = np.zeros((1, K))
    x0[:, start_char_ind] = 1

    Y = np.zeros((n+1,K))
    Y[0,:]=x0
    ht = h0
    for i in range(n):
        xt = Y[i,:]
        at= ht@RNN['W'] + xt@RNN['U'] + RNN['b']
        ht = np.tanh(at)
        ot = ht@RNN['V']+RNN['c']
        pt = SoftMax(ot)
        cp = np.cumsum(pt, axis=1)
        a = rng.uniform(size=1)
        nextx = np.argmax(cp - a > 0)
        Y[i+1,nextx]=1
    return Y

def getTextFromY(Ygen, index_to_character):
    indices = np.argmax(Ygen, axis=1)
    chars = [index_to_character[i] for i in indices]
    return ''.join(chars)

def ComputeLoss(fp_data,y):
    P = fp_data['P']
    tau = P.shape[0]
    probs = P[np.arange(tau),y]
    logProbs = -np.log(probs)
    return (1/tau)*np.sum(logProbs)

def forwardPass(X,h0, network):
    fp_data = {}
    fp_data['X'] = X #(tau x K)
    tau = X.shape[0]
    m = network['W'].shape[0]
    A = np.zeros((tau,m)) #(tau x m)
    H = np.zeros((tau+1,m)) #((tau+1) x m)

    XU = X@network['U'] #(tau x K)@(K x m) = (tau x m)
    #ht = np.zeros((1, m))
    ht = h0 #(1 x m)
    H[0,:] = ht
    for t in range(tau):
        at = ht@network['W'] + XU[t,:] + network['b'] #(1 x m)@(m x m) + (1 x m) + (1 x m) = (1 x m)
        A[t,:] = at
        ht = np.tanh(at)
        H[t+1,:] = ht

    fp_data['A'] = A
    fp_data['H'] = H

    O = H[1:,:]@network['V']+ network['c'] #(tau x m) @ (m x K) + (1xK) = (tau x K)
    fp_data['O'] = O
    P = SoftMax(O) #(tau x K)
    fp_data['P'] = P

    return fp_data

def backwardPass(Y, fp_data, network):
    grads = {}
    tau = fp_data['X'].shape[0]
    m = network['W'].shape[0]
    G = -(Y-fp_data['P']).T #((tau x K) - (tau x K)).T=(K x tau)
    grads['V']= (1 / tau) * fp_data['H'][1:,:].T@G.T #(tau x m).T @ (K x tau).T = (m, K)
    grads['c'] = (1 / tau) * np.ones((1, tau))@G.T #(1 x tau) @ (K x tau).T = (1, K)

    grad_to_h_t_part_1 = network['V'] @ G  # (m x K)@(K x tau) = (m x tau)
    grad_h_to_a = 1-fp_data['H'][1:,]**2 #(tau x m)

    grad_L_to_a_t = grad_h_to_a[tau-1,:]*grad_to_h_t_part_1[:,tau-1] #(m,)*(m,) = (m,)
    GNew = np.zeros((tau,m)) #(tau x m)
    GNew[tau-1,:] = grad_L_to_a_t #(1 x m)
    for t in range(tau - 1, 0, -1):
        grad_to_h_t = grad_to_h_t_part_1[:,t-1] + network['W']@grad_L_to_a_t #(m x 1) + (m x m)@(m,) = (m x 1)
        grad_L_to_a_t = grad_h_to_a[t-1,:]*grad_to_h_t #(m,)*(m,) = (m,)
        GNew[t-1,:] = grad_L_to_a_t #(1 x m)
    G= GNew

    grads['W'] = (1 / tau) *fp_data['H'][:tau,:].T@G #(tau x m).T @ (tau x m) = (m x m)
    grads['U'] = (1 / tau) *fp_data['X'].T@G #(tau x K).T @ (tau x m) = (K x m)

    grads['b'] = (1 / tau) * np.ones((1, tau))@G #(1 x tau) @ (tau x m) = (1 x m)

    return grads

def encode_sequence(sequence, char_to_ind):
    K = len(char_to_ind)           # Number of unique characters
    n = len(sequence)              # Sequence length
    X = np.zeros((n, K))           # One-hot encoded matrix of shape (K, n)
    indices = np.zeros(n, dtype=int)

    for t, char in enumerate(sequence):
        index = char_to_ind[char]
        X[t,index] = 1
        indices[t] = index

    return X, indices

def plot_smooth_loss(smooth_losses):
    update_steps = range(len(smooth_losses))

    plt.figure(figsize=(6, 4))
    # Plot smooth loss
    plt.plot(update_steps, smooth_losses, label='Smooth Loss')
    plt.xlabel('Update Steps')
    plt.ylabel('Smooth Loss')
    plt.title('Smooth Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def AdamOptimizer(book_data, char_to_ind, ind_to_char, GDparams, init_net, rng, print_loss = False, print_text_before = 10000):
    seq_length = GDparams['seq_length']
    num_iterations = GDparams['num_iterations']
    eta = GDparams['eta']
    beta_1 = GDparams['beta_1']
    beta_2 = GDparams['beta_2']
    epsilon = GDparams['epsilon']

    RNN = copy.deepcopy(init_net)
    m = RNN['W'].shape[0]

    AdamParams = {}
    for key, parameter in RNN.items():
        newParams = {}
        newParams['m'] = np.zeros_like(parameter)
        newParams['v'] = np.zeros_like(parameter)
        AdamParams[key] = newParams

    hprev = np.zeros((1,m))

    smooth_loss = 0
    smooth_losses = []

    e = 0
    epoch = 1

    print("Synthesised text before first update step")
    print(getTextFromY(synthesiseTextFromRNN(hprev, 200, RNN, rng), ind_to_char))

    for iter in range(num_iterations):
        if e>len(book_data)-seq_length-1:
            e =0
            hprev = np.zeros((1,m))
            epoch+=1
            #print("Entering epoch "+ str(epoch))

        X, _ = encode_sequence(book_data[e:e + seq_length], char_to_ind)
        Y, y = encode_sequence(book_data[e + 1:e + seq_length + 1], char_to_ind)

        fp_data_sequence = forwardPass(X,hprev,RNN)
        hprev = fp_data_sequence['H'][seq_length,:]
        grads = backwardPass(Y,fp_data_sequence,RNN)
        if iter == 0:
            smooth_loss = ComputeLoss(fp_data_sequence,y)
        else:
            smooth_loss = 0.999*smooth_loss + 0.001*ComputeLoss(fp_data_sequence,y)
        smooth_losses.append(smooth_loss)

        for kk in grads.keys():
            AdamParams[kk]['m'] = beta_1*AdamParams[kk]['m']+(1-beta_1)*grads[kk]
            AdamParams[kk]['v'] = beta_2*AdamParams[kk]['v'] + (1-beta_2)*(grads[kk]**2)
            update_m = AdamParams[kk]['m'] / (1 - beta_1 ** (iter + 1))
            update_v = AdamParams[kk]['v']/(1-beta_2**(iter+1))
            RNN[kk] -=  eta /(np.sqrt(update_v)+epsilon) * update_m

        e+=seq_length

        if iter%100==0 and print_loss:
            print("After " + str(iter) + " iterations the smooth loss is " +str(smooth_loss))

        if (iter+1)%print_text_before==0:
            print("Synthesised text before " + str((iter+1)) + " update steps")
            print(getTextFromY(synthesiseTextFromRNN(hprev,200, RNN, rng), ind_to_char))


    plot_smooth_loss(smooth_losses)

    return RNN


def test_part_3():
    book_data, char_to_ind, ind_to_char, K = read_in_data()
    rng = initializeRNG()

    m= 100 #Hidden layer size
    seq_length = 25
    eta = 0.001
    init_RNN = initialiseRNN(K,m,rng)

    h0 = np.zeros((1,m))

    Ygen = synthesiseTextFromRNN(h0,seq_length,init_RNN,rng)
    text = getTextFromY(Ygen, ind_to_char)
    print(text)

def test_part_4():
    book_data, char_to_ind, ind_to_char, K = read_in_data()
    rng = initializeRNG()

    m = 10  # Hidden layer size
    seq_length = 25
    init_RNN = initialiseRNN(K, m, rng)

    X,_ = encode_sequence(book_data[0:seq_length], char_to_ind)
    Y,y = encode_sequence(book_data[1:seq_length+1], char_to_ind)
    h0 = np.zeros((1,m))

    print("Doing forward pass")
    fp_data = forwardPass(X,h0, init_RNN)
    print("Doing backward pass")
    grads = backwardPass(Y, fp_data, init_RNN)
    print("Doing torch")
    torch_grads = ComputeGradsWithTorch(X, y, h0, init_RNN)

    if np.allclose(grads['V'], torch_grads['V'], rtol=0, atol=1e-10):
        print("No big difference found in V")
    else:
        print("Big difference found in V")

    if np.allclose(grads['c'], torch_grads['c'], rtol=0, atol=1e-10):
        print("No big difference found in c")
    else:
        print("Big difference found in c")

    if np.allclose(grads['U'], torch_grads['U'], rtol=0, atol=1e-10):
        print("No big difference found in U")
    else:
        print("Big difference found in U")

    if np.allclose(grads['W'], torch_grads['W'], rtol=0, atol=1e-10):
        print("No big difference found in W")
    else:
        print("Big difference found in W")

    if np.allclose(grads['b'], torch_grads['b'], rtol=0, atol=1e-10):
        print("No big difference found in b")
    else:
        print("Big difference found in b")

def test_part_5():
    book_data, char_to_ind, ind_to_char, K = read_in_data()
    rng = initializeRNG()

    m = 100
    seq_length = 25
    eta = 0.001

    init_RNN = initialiseRNN(K, m, rng)
    GD_params ={"seq_length": seq_length,"num_iterations": 300001, "eta": eta,"beta_1":0.9,"beta_2":0.999, "epsilon":1e-8}

    trained_RNN = AdamOptimizer(book_data, char_to_ind, ind_to_char, GD_params, init_RNN, rng, print_loss=True, print_text_before=1000)

def results_report():
    book_data, char_to_ind, ind_to_char, K = read_in_data()
    rng = initializeRNG()

    m = 100
    seq_length = 25
    eta = 0.001

    init_RNN = initialiseRNN(K, m, rng)
    GD_params = {"seq_length": seq_length, "num_iterations": 300001, "eta": eta, "beta_1": 0.9, "beta_2": 0.999,
                 "epsilon": 1e-8}
    trained_RNN = AdamOptimizer(book_data, char_to_ind, ind_to_char, GD_params, init_RNN, rng)

    text_string = getTextFromY(synthesiseTextFromRNN(np.zeros((1,m)),1000, trained_RNN, rng), ind_to_char)

    with open("output.txt", "w") as file:
        file.write(text_string)
    print("String saved to output.txt")

results_report()

