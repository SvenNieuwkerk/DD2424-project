import numpy as np
import torch
import copy
import matplotlib.pyplot as plt
import requests
import re
import random

from fontTools.misc.plistlib import end_data
from spellchecker import SpellChecker

def read_in_data(seq_length = -1):
    book_dir = r'C:\Users\svenr\OneDrive - KTH\Deep Learning in Data Science\Project\DD2424-project\\'
    book_fname = book_dir + 'goblet_book.txt'
    fid = open(book_fname, "r")
    book_data = fid.read()
    fid.close()

    unique_chars = list(set(book_data))
    K = len(unique_chars)

    char_to_ind = {ch: i for i, ch in enumerate(unique_chars)}
    ind_to_char = {i: ch for i, ch in enumerate(unique_chars)}

    return book_data, char_to_ind, ind_to_char, K

def load_poems():
    url = "https://www.gutenberg.org/cache/epub/12242/pg12242.txt"
    r = requests.get(url)
    r.encoding = 'utf-8'
    raw = r.text

    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK POEMS BY EMILY DICKINSON, THREE SERIES, COMPLETE ***"
    end_marker   = "*** END OF THE PROJECT GUTENBERG EBOOK POEMS BY EMILY DICKINSON, THREE SERIES, COMPLETE ***"

    start_idx = raw.find(start_marker)
    end_idx   = raw.find(end_marker)

    body = raw[start_idx + len(start_marker): end_idx]

    first_num_match = re.search(r'(?m)^[IVXLCDM]+\.\s*$', body)
    body = body[first_num_match.start():]

    poems = re.split(r'(?m)^[IVXLCDM]+\.\s*$', body)
    poems = [p.strip() for p in poems if len(p.strip().split()) > 10]

    lines = poems[280].splitlines()
    poems[280] = "\n".join(lines[:4])

    lines = poems[114].splitlines()
    poems[114] = "\n".join(lines[:9])

    lines = poems[0].splitlines()
    poems[0] = "\n".join(lines[6:])

    return poems

def initializeRNG():
    rng = np.random.default_rng()
    # get the BitGenerator used by default rng
    BitGen = type(rng.bit_generator)
    # use the state from a fresh bit generator
    seed = 42
    rng.bit_generator.state = BitGen(seed).state
    return rng

def init_network(K, m, rng, l):
    # Identity and zero matrices
    I = np.eye(m)
    Z = np.zeros((m, m))

    network = {}
    # Construct E1 and E2 by horizontal stacking
    network['E1'] = np.vstack([I, Z, Z, Z])
    network['E2'] = np.vstack([Z, I, Z, Z])
    network['E3'] = np.vstack([Z, Z, I, Z])
    network['E4'] = np.vstack([Z, Z, Z, I])

    network['b'] = np.zeros((1, 4*m))
    network['c'] = np.zeros((1, K))
    if l == 1:
        network['U'] = (1 / np.sqrt(2 * K)) * rng.standard_normal(size=(K, 4 * m))
    else:
        network['U'] = (1 / np.sqrt(2 * K)) * rng.standard_normal(size=(m, 4 * m))
    network['W'] = (1 / np.sqrt(2 * m)) * rng.standard_normal(size=(m, 4*m))
    network['V'] = (1 / np.sqrt(m)) * rng.standard_normal(size=(m, K))

    return network

def init_network_torch(K, m, rng):
    numpy_network = init_network(K, m, rng, 1)
    torch_network = {}
    for kk in numpy_network.keys():
        if kk in ('E1', 'E2', 'E3', 'E4'):
            torch_network[kk] = torch.tensor(numpy_network[kk], requires_grad=False)
        else:
            torch_network[kk] = torch.tensor(numpy_network[kk], requires_grad=True)
    return torch_network

def init_network_torch_2_layer(K, m, rng, L):
    torch_network = {}
    for i in (1,L):
        #U2 should have different shape as X has shape tau x K and h1, the input for U2 has shape tau x m
        layer = init_network(K, m, rng, i)
        for kk in layer.keys():
            if kk == 'V' or kk == 'c':
                if not kk in torch_network.keys():
                    torch_network[kk] = torch.tensor(layer[kk], requires_grad=True)
            else:
                key = f"{kk}{i}"
                if kk in ('E1', 'E2', 'E3', 'E4'):
                    torch_network[key] = torch.tensor(layer[kk], requires_grad=False)
                else:
                    torch_network[key] = torch.tensor(layer[kk], requires_grad=True)
    return torch_network


def synthesiseTextFromRNNMultilayer(h0, c0, n, RNN, rng,L, temparature = 1, nucleus_p = 1):
    K = RNN['U'+str(1)].shape[0]

    apply_tanh = torch.nn.Tanh()
    apply_softmax = torch.nn.Softmax(dim=1)

    start_char_ind = np.random.randint(0, K)

    x0 = torch.zeros((1, K), dtype = torch.float64)
    x0[:, start_char_ind] = 1

    Y = torch.zeros((n+1,K), dtype = torch.float64)
    Y[0,:]=x0
    ht = copy.deepcopy(h0)
    ct = copy.deepcopy(c0)
    for i in range(n):
        xt = Y[i,:]
        for l in range(1,L+1):
            #ht -> xt, new h0-> ht
            at= ht[l]@RNN['W'+str(l)] + xt@RNN['U'+str(l)] + RNN['b'+str(l)]

            ft = apply_softmax(at @ RNN['E1'+str(l)])
            it = apply_softmax(at @ RNN['E2'+str(l)])
            ot = apply_softmax(at @ RNN['E3'+str(l)])
            c_wiggle_t = apply_tanh(at @ RNN['E4'+str(l)])
            ct[l] = ft * ct[l] + it * c_wiggle_t
            ht[l] = ot * apply_tanh(ct[l])
            xt = ht[l]

        zt = ht[L]@RNN['V']+RNN['c']
        #Applying temperature
        zt = zt / temparature
        pt = apply_softmax(zt)

        pt_np = pt.detach().cpu().numpy().flatten()

        #Sort probabilities and cumulate the sum
        sorted_indices = np.argsort(-pt_np)
        cumulative_probs = np.cumsum(pt_np[sorted_indices])

        #Cutoff the probalities when they reach nucleus_p and renormalize the probabilities
        sorted_indices = sorted_indices[cumulative_probs <= nucleus_p]
        candidate_probs = pt_np[sorted_indices]
        candidate_probs /= candidate_probs.sum()

        #Sample from the top-nucleus_p tokens
        nextx = rng.choice(sorted_indices, p=candidate_probs)
        Y[i+1,nextx]=1

    return Y

def synthesiseTextFromRNN(h0, c0, n, RNN, rng, temparature = 1, nucleus_p = 1):
    K = RNN['U'].shape[0]

    apply_tanh = torch.nn.Tanh()
    apply_softmax = torch.nn.Softmax(dim=1)

    start_char_ind = np.random.randint(0, K)

    x0 = torch.zeros((1, K), dtype = torch.float64)
    x0[:, start_char_ind] = 1

    Y = torch.zeros((n+1,K), dtype = torch.float64)
    Y[0,:]=x0
    ht = h0
    ct = c0
    for i in range(n):
        xt = Y[i,:]

        at= ht@RNN['W'] + xt@RNN['U'] + RNN['b']

        ft = apply_softmax(at @ RNN['E1'])
        it = apply_softmax(at @ RNN['E2'])
        ot = apply_softmax(at @ RNN['E3'])
        c_wiggle_t = apply_tanh(at @ RNN['E4'])
        ct = ft * ct + it * c_wiggle_t
        ht = ot * apply_tanh(ct)

        zt = ht@RNN['V']+RNN['c']
        #Applying temperature
        zt = zt / temparature
        pt = apply_softmax(zt)

        pt_np = pt.detach().cpu().numpy().flatten()

        #Sort probabilities and cumulate the sum
        sorted_indices = np.argsort(-pt_np)
        cumulative_probs = np.cumsum(pt_np[sorted_indices])

        #Cutoff the probalities when they reach nucleus_p and renormalize the probabilities
        sorted_indices = sorted_indices[cumulative_probs <= nucleus_p]
        candidate_probs = pt_np[sorted_indices]
        candidate_probs /= candidate_probs.sum()

        #Sample from the top-nucleus_p tokens
        nextx = rng.choice(sorted_indices, p=candidate_probs)
        Y[i+1,nextx]=1

    return Y

def getTextFromY(Ygen, index_to_character):
    indices = torch.argmax(Ygen, axis=1)
    chars = [index_to_character[i.item()] for i in indices]
    return ''.join(chars)

def plot_smooth_loss(smooth_losses, validation_losses, validation_iterations):
    update_steps = range(len(smooth_losses))

    plt.figure(figsize=(6, 4))
    # Plot smooth loss
    plt.plot(update_steps, smooth_losses, label='Smooth Loss')
    plt.plot(validation_iterations, validation_losses, label='Validation Loss')
    plt.xlabel('Update Steps')
    plt.ylabel('Smooth Loss')
    plt.title('Smooth Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compute_loss(fp_data, y):
    X = fp_data['X']  # (tau x K)
    tau = X.shape[0]
    P = fp_data['P']  # (tau x K)
    loss = torch.mean(-torch.log(P[torch.arange(tau), y]))
    return loss.item()


def spelling_accuracy(generated_text):
    spell = SpellChecker()
    words = generated_text.split()
    total_words = len(words)
    misspelled = spell.unknown(words)
    num_misspelled = len(misspelled)

    correct_percentage = (total_words - num_misspelled) / total_words * 100
    return correct_percentage


def ngram_overlap(generated_text, training_text, n):
    def get_ngrams(text, n):
        tokens = text.split()
        return list(zip(*[tokens[i:] for i in range(n)]))

    gen_ngrams = get_ngrams(generated_text, n)
    print(gen_ngrams)
    train_ngrams = set(get_ngrams(training_text, n))

    match_count = sum(1 for gram in gen_ngrams if gram in train_ngrams)
    overlap_percentage = match_count / len(gen_ngrams) * 100
    return overlap_percentage

def encode_sequence(sequence, char_to_ind):
    K = len(char_to_ind)           # Number of unique characters
    n = len(sequence)              # Sequence length
    X = torch.zeros((n, K), dtype = torch.float64)        # One-hot encoded matrix of shape (K, n)
    indices = torch.zeros(n, dtype=torch.int64)

    for t, char in enumerate(sequence):
        index = char_to_ind[char]
        X[t,index] = 1
        indices[t] = index

    return X, indices

def forwardPassLayerUsingTorch(inputlayer,h0layer,c0layer, torch_network, layer_nr, fp_data):
    #Make sure X and Y are torch
    if layer_nr == 1:
        fp_data['X'] = inputlayer #(tau x K)
    tau = inputlayer.shape[0]
    m = torch_network['W'+str(layer_nr)].shape[0] #Might be the wrong parameter for the shape if the layers have different sizes

    apply_tanh = torch.nn.Tanh()
    apply_softmax = torch.nn.Softmax(dim=1)

    H_list = [h0layer]
    C_list = [c0layer]
    I_list, F_list, O_list, Cw_list, A_list = [], [], [], [], []

    XU = inputlayer@torch_network['U'+str(layer_nr)] #(tau x K)@(K x 4m) = (tau x 4m)
    #If we want to do it individually
    #XUi = X @ network['Ui']  # (tau x K)@(K x m) = (tau x m)
    #XUf = X @ network['Uf']  # (tau x K)@(K x m) = (tau x m)
    #XUo = X @ network['Uo']  # (tau x K)@(K x m) = (tau x m)
    #XUc = X @ network['Uc']  # (tau x K)@(K x m) = (tau x m)

    #ht = np.zeros((1, m))
    ht = h0layer #(1 x m)
    ct = c0layer
    for t in range(tau):
        at = ht @ torch_network['W'+str(layer_nr)] + XU[t, :] + torch_network['b'+str(layer_nr)]
        A_list.append(at.squeeze(0))

        ft = apply_softmax(at @ torch_network['E1'+str(layer_nr)])
        F_list.append(ft.squeeze(0))
        it = apply_softmax(at @ torch_network['E2'+str(layer_nr)])
        I_list.append(it.squeeze(0))
        ot = apply_softmax(at @ torch_network['E3'+str(layer_nr)])
        O_list.append(ot.squeeze(0))
        c_wiggle_t = apply_tanh(at @ torch_network['E4'+str(layer_nr)])
        Cw_list.append(c_wiggle_t.squeeze(0))

        ct = ft * C_list[-1] + it * c_wiggle_t
        C_list.append(ct.squeeze(0))

        ht = ot * apply_tanh(ct)
        H_list.append(ht.squeeze(0))

    fp_data['A'+str(layer_nr)] = torch.stack(A_list)
    fp_data['F'+str(layer_nr)] = torch.stack(F_list)
    fp_data['I'+str(layer_nr)] = torch.stack(I_list)
    fp_data['O'+str(layer_nr)] = torch.stack(O_list)
    fp_data['C'+str(layer_nr)] = torch.stack(C_list)
    fp_data['C_wiggle'+str(layer_nr)] = torch.stack(Cw_list)
    fp_data['H'+str(layer_nr)] = torch.stack(H_list)
    return fp_data

def forwardPassMultiLayer (X,h0_both_layers,c0_both_layers, torch_network, L):
    fp_data = {}
    input = X
    for i in range(1, L + 1):
        fp_data = forwardPassLayerUsingTorch(input, h0_both_layers[i],c0_both_layers[i],torch_network, i, fp_data)
        input= fp_data['H'+str(i)][1:]

    apply_softmax = torch.nn.Softmax(dim=1)

    Z = fp_data['H'+str(L)][1:] @ torch_network['V'] + torch_network['c']  # (tau x m) @ (m x K) + (1xK) = (tau x K)
    fp_data['Z'] = Z
    P = apply_softmax(Z)  # (tau x K)
    fp_data['P'] = P
    return fp_data

def forwardPassUsingTorch(X,h0,c0, torch_network):
    #Make sure X and Y are torch
    fp_data = {}
    fp_data['X'] = X #(tau x K)
    tau = X.shape[0]
    m = torch_network['W'].shape[0]

    apply_tanh = torch.nn.Tanh()
    apply_softmax = torch.nn.Softmax(dim=1)

    H_list = [h0]
    C_list = [c0]
    I_list, F_list, O_list, Cw_list, A_list = [], [], [], [], []

    XU = X@torch_network['U'] #(tau x K)@(K x 4m) = (tau x 4m)
    #If we want to do it individually
    #XUi = X @ network['Ui']  # (tau x K)@(K x m) = (tau x m)
    #XUf = X @ network['Uf']  # (tau x K)@(K x m) = (tau x m)
    #XUo = X @ network['Uo']  # (tau x K)@(K x m) = (tau x m)
    #XUc = X @ network['Uc']  # (tau x K)@(K x m) = (tau x m)

    #ht = np.zeros((1, m))
    ht = h0 #(1 x m)
    ct = c0
    for t in range(tau):
        at = ht @ torch_network['W'] + XU[t, :] + torch_network['b']
        A_list.append(at.squeeze(0))

        ft = apply_softmax(at @ torch_network['E1'])
        F_list.append(ft.squeeze(0))
        it = apply_softmax(at @ torch_network['E2'])
        I_list.append(it.squeeze(0))
        ot = apply_softmax(at @ torch_network['E3'])
        O_list.append(ot.squeeze(0))
        c_wiggle_t = apply_tanh(at @ torch_network['E4'])
        Cw_list.append(c_wiggle_t.squeeze(0))

        ct = ft * C_list[-1] + it * c_wiggle_t
        C_list.append(ct.squeeze(0))

        ht = ot * apply_tanh(ct)
        H_list.append(ht.squeeze(0))

    fp_data['A'] = torch.stack(A_list)
    fp_data['F'] = torch.stack(F_list)
    fp_data['I'] = torch.stack(I_list)
    fp_data['O'] = torch.stack(O_list)
    fp_data['C'] = torch.stack(C_list)
    fp_data['C_wiggle'] = torch.stack(Cw_list)
    fp_data['H'] = torch.stack(H_list)


    Z = fp_data['H'][1:]@torch_network['V']+ torch_network['c'] #(tau x m) @ (m x K) + (1xK) = (tau x K)
    fp_data['Z'] = Z
    P = apply_softmax(Z) #(tau x K)
    fp_data['P'] = P

    return fp_data

def backwardPassUsingTorch(y, fp_data, torch_network):
    #X needs to be torch, y can be numpy
    X = fp_data['X'] # (tau x K)
    tau = X.shape[0]
    P = fp_data['P']  #(tau x K)

    loss = torch.mean(-torch.log(P[torch.arange(tau), y]))

    loss.backward()

    # extract the computed gradients and make them numpy arrays
    grads = {}
    #What to do with E's
    for kk in torch_network.keys():
        if kk in ('E1', 'E2', 'E3', 'E4', 'E11', 'E12', 'E21', 'E22', 'E31', 'E32', 'E41', 'E42'):
            continue
        grads[kk] = torch_network[kk].grad.clone()

    return grads

def AdamOptimizerPoem(poems, validation_set, char_to_ind, ind_to_char, GDparams, L, init_net, rng, print_loss = False, print_text_before = 1000, multilayer_forward = True):
    seq_length = GDparams['seq_length']
    num_iterations = GDparams['num_iterations']
    eta = GDparams['eta']
    beta_1 = GDparams['beta_1']
    beta_2 = GDparams['beta_2']
    epsilon = GDparams['epsilon']

    RNN = copy.deepcopy(init_net)
    if multilayer_forward:
        m = RNN['W'+str(1)].shape[0]
    else:
        m = RNN['W'].shape[0]

    AdamParams = {}
    for key, parameter in RNN.items():
        newParams = {}
        newParams['m'] = torch.zeros_like(parameter)
        newParams['v'] = torch.zeros_like(parameter)
        AdamParams[key] = newParams

    #hprev = torch.zeros((1,m), dtype = torch.float64)
    #cprev = torch.zeros((1,m), dtype = torch.float64)
    if multilayer_forward:
        hprev = {}
        cprev = {}
        for i in range(1,L+1):
            hprev[i] = torch.zeros(m, dtype=torch.float64)
            cprev[i] = torch.zeros(m, dtype=torch.float64)
    else:
        hprev = torch.zeros(m, dtype=torch.float64)
        cprev = torch.zeros(m, dtype=torch.float64)

    smooth_loss = 0
    smooth_losses = []
    validation_losses = []
    validation_iterations = []
    e = 0
    poem = 0
    while poem in validation_set:
        poem += 1
    epoch = 1

    print("Synthesised text before first update step")
    if multilayer_forward:
        print(getTextFromY(synthesiseTextFromRNNMultilayer(hprev, cprev, 200, RNN, rng, L), ind_to_char))
    else:
        print(getTextFromY(synthesiseTextFromRNN(hprev, cprev, 200, RNN, rng), ind_to_char))

    for iter in range(num_iterations):
        start = e
        go_to_next_poem = False
        if e+seq_length+1 >= len(poems[poem]):
            end = len(poems[poem])-1
            go_to_next_poem = True
        else:
            end = e+seq_length
            e = end

        X, _ = encode_sequence(poems[poem][start:end], char_to_ind)
        Y, y = encode_sequence(poems[poem][(start+1):(end+1)], char_to_ind)

        if multilayer_forward:
            fp_data_sequence = forwardPassMultiLayer(X,hprev,cprev, RNN, L)
        else:
            fp_data_sequence = forwardPassUsingTorch(X,hprev, cprev, RNN)

        grads = backwardPassUsingTorch(y,fp_data_sequence,RNN)

        if iter == 0:
            smooth_loss = compute_loss(fp_data_sequence,y)
        else:
            smooth_loss = 0.999*smooth_loss + 0.001*compute_loss(fp_data_sequence,y)
        smooth_losses.append(smooth_loss)

        if iter%250==0:
            loss_validation = 0
            for validation_poem_nr in validation_set:
                if multilayer_forward:
                    h0val = {}
                    c0val = {}
                    for i in range(1, L + 1):
                        h0val[i] = torch.zeros(m, dtype=torch.float64)
                        c0val[i] = torch.zeros(m, dtype=torch.float64)
                else:
                    h0val = torch.zeros(m, dtype=torch.float64)
                    c0val = torch.zeros(m, dtype=torch.float64)


                validation_poem = poems[validation_poem_nr]
                X_val,_ = encode_sequence(validation_poem[:len(validation_poem)-1], char_to_ind)
                Y_val, y_val = encode_sequence(validation_poem[1:len(validation_poem)], char_to_ind)
                if multilayer_forward:
                    fp_validation_poem = forwardPassMultiLayer(X_val, h0val, c0val, RNN, L)
                else:
                    fp_validation_poem = forwardPassUsingTorch(X_val,h0val, c0val, RNN)
                loss_validation += compute_loss(fp_validation_poem, y_val)
            loss_validation = loss_validation/len(validation_set)
            validation_losses.append(loss_validation)
            validation_iterations.append(iter)

        for kk in grads.keys():
            AdamParams[kk]['m'] = beta_1*AdamParams[kk]['m']+(1-beta_1)*grads[kk]
            AdamParams[kk]['v'] = beta_2*AdamParams[kk]['v'] + (1-beta_2)*(grads[kk]**2)
            update_m = AdamParams[kk]['m'] / (1 - beta_1 ** (iter + 1))
            update_v = AdamParams[kk]['v']/(1-beta_2**(iter+1))
            with torch.no_grad():
                RNN[kk] -= eta / (torch.sqrt(update_v) + epsilon) * update_m

        if go_to_next_poem:
            poem += 1
            while poem in validation_set:
                poem += 1
            if poem >= len(poems):
                poem = 0
                while poem in validation_set:
                    poem += 1
            e = 0
            if multilayer_forward:
                hprev = {}
                cprev = {}
                for i in range(1, L + 1):
                    hprev[i] = torch.zeros(m, dtype=torch.float64)
                    cprev[i] = torch.zeros(m, dtype=torch.float64)
            else:
                hprev = torch.zeros(m, dtype=torch.float64)
                cprev = torch.zeros(m, dtype=torch.float64)
        else:
            if multilayer_forward:
                hprev = {}
                cprev = {}
                for i in range(1, L+1):
                    hprev[i] = fp_data_sequence['H'+str(i)][end - start, :].detach()
                    cprev[i] = fp_data_sequence['C'+str(i)][end - start, :].detach()
            else:
                hprev = fp_data_sequence['H'][end - start, :].detach()
                cprev = fp_data_sequence['C'][end - start, :].detach()

        if iter%100==0 and print_loss:
            print("After " + str(iter) + " iterations the smooth loss is " +str(smooth_loss))
        if iter%250==0 and print_loss:
            print("After " + str(iter) + " iterations the validation loss is " +str(validation_losses[-1]))
        if (iter+1)%print_text_before==0:
            print("Synthesised text before " + str((iter+1)) + " update steps")
            if multilayer_forward:
                print(getTextFromY(synthesiseTextFromRNNMultilayer(hprev, cprev, 200, RNN, rng, L), ind_to_char))
            else:
                print(getTextFromY(synthesiseTextFromRNN(hprev, cprev, 200, RNN, rng), ind_to_char))



    plot_smooth_loss(smooth_losses, validation_losses, validation_iterations)

    return RNN, validation_losses[-1]

def AdamOptimizerBook(book_data, start_validation_set, char_to_ind, ind_to_char, GDparams, L, init_net, rng, print_loss = False, print_text_before = 1000, multilayer_forward = True):
    seq_length = GDparams['seq_length']
    num_iterations = GDparams['num_iterations']
    eta = GDparams['eta']
    beta_1 = GDparams['beta_1']
    beta_2 = GDparams['beta_2']
    epsilon = GDparams['epsilon']

    RNN = copy.deepcopy(init_net)
    if multilayer_forward:
        m = RNN['W'+str(1)].shape[0]
    else:
        m = RNN['W'].shape[0]

    AdamParams = {}
    for key, parameter in RNN.items():
        newParams = {}
        newParams['m'] = torch.zeros_like(parameter)
        newParams['v'] = torch.zeros_like(parameter)
        AdamParams[key] = newParams

    #hprev = torch.zeros((1,m), dtype = torch.float64)
    #cprev = torch.zeros((1,m), dtype = torch.float64)
    if multilayer_forward:
        hprev = {}
        cprev = {}
        for i in range(1,L+1):
            hprev[i] = torch.zeros(m, dtype=torch.float64)
            cprev[i] = torch.zeros(m, dtype=torch.float64)
    else:
        hprev = torch.zeros(m, dtype=torch.float64)
        cprev = torch.zeros(m, dtype=torch.float64)

    smooth_loss = 0
    smooth_losses = []
    validation_losses = []
    validation_iterations = []
    e = 0
    epoch = 1

    print("Synthesised text before first update step")
    if multilayer_forward:
        print(getTextFromY(synthesiseTextFromRNNMultilayer(hprev, cprev, 200, RNN, rng, L), ind_to_char))
    else:
        print(getTextFromY(synthesiseTextFromRNN(hprev, cprev, 200, RNN, rng), ind_to_char))

    for iter in range(num_iterations):
        if e > start_validation_set - seq_length - 1:
            e = 0
            hprev = {}
            cprev = {}
            for i in range(1, L + 1):
                hprev[i] = torch.zeros(m, dtype=torch.float64)
                cprev[i] = torch.zeros(m, dtype=torch.float64)
            epoch += 1

        X, _ = encode_sequence(book_data[e:e + seq_length], char_to_ind)
        Y, y = encode_sequence(book_data[e + 1:e + seq_length + 1], char_to_ind)

        if multilayer_forward:
            fp_data_sequence = forwardPassMultiLayer(X,hprev,cprev, RNN, L)
        else:
            fp_data_sequence = forwardPassUsingTorch(X,hprev, cprev, RNN)

        if multilayer_forward:
            hprev = {}
            cprev = {}
            for i in range(1, L + 1):
                hprev[i] = fp_data_sequence['H' + str(i)][seq_length, :].detach()
                cprev[i] = fp_data_sequence['C' + str(i)][seq_length, :].detach()
        else:
            hprev = fp_data_sequence['H'][seq_length, :].detach()
            cprev = fp_data_sequence['C'][seq_length, :].detach()

        grads = backwardPassUsingTorch(y,fp_data_sequence,RNN)

        if iter == 0:
            smooth_loss = compute_loss(fp_data_sequence,y)
        else:
            smooth_loss = 0.999*smooth_loss + 0.001*compute_loss(fp_data_sequence,y)
        smooth_losses.append(smooth_loss)

        #print("Start validation step")
        skip_validation = True
        if iter%250==0 and not skip_validation:
            h0val = {}
            c0val = {}
            for i in range(1, L + 1):
                h0val[i] = torch.zeros(m, dtype=torch.float64)
                c0val[i] = torch.zeros(m, dtype=torch.float64)

            #validation_poem = poems[validation_poem_nr]
            X_val,_ = encode_sequence(book_data[start_validation_set:len(book_data)-1], char_to_ind)
            Y_val, y_val = encode_sequence(book_data[start_validation_set+1:len(book_data)], char_to_ind)
            if multilayer_forward:
                fp_data_validation = forwardPassMultiLayer(X_val, h0val, c0val, RNN, L)
            else:
                fp_data_validation = forwardPassUsingTorch(X_val,h0val, c0val, RNN)
            loss_validation = compute_loss(fp_data_validation,y_val)
            validation_losses.append(loss_validation)
            validation_iterations.append(iter)

        for kk in grads.keys():
            AdamParams[kk]['m'] = beta_1*AdamParams[kk]['m']+(1-beta_1)*grads[kk]
            AdamParams[kk]['v'] = beta_2*AdamParams[kk]['v'] + (1-beta_2)*(grads[kk]**2)
            update_m = AdamParams[kk]['m'] / (1 - beta_1 ** (iter + 1))
            update_v = AdamParams[kk]['v']/(1-beta_2**(iter+1))
            with torch.no_grad():
                RNN[kk] -= eta / (torch.sqrt(update_v) + epsilon) * update_m

        e+=seq_length


        if iter%100==0 and print_loss:
            print("After " + str(iter) + " iterations the smooth loss is " +str(smooth_loss))
        if iter%250==0 and print_loss and not skip_validation:
            print("After " + str(iter) + " iterations the validation loss is " +str(validation_losses[-1]))
        if (iter+1)%print_text_before==0:
            print("Synthesised text before " + str((iter+1)) + " update steps")
            if multilayer_forward:
                print(getTextFromY(synthesiseTextFromRNNMultilayer(hprev, cprev, 200, RNN, rng, L), ind_to_char))
            else:
                print(getTextFromY(synthesiseTextFromRNN(hprev, cprev, 200, RNN, rng), ind_to_char))



    plot_smooth_loss(smooth_losses, validation_losses, validation_iterations)

    return RNN, validation_losses[-1]

def main(book=True):
    """
    text = "This is a test for mispeling, should retrn seveny percent"
    print(spelling_accuracy(text))

    test_input = "This is a test sentence"
    test_text = "This is a sentence a test"
    print("With n= 2, overlap percentage is ", ngram_overlap(test_text, test_input, 2)) # This should print 60% ??? maybe
    print("With n= 3, overlap percentage is ", ngram_overlap(test_text, test_input, 3)) # This should print 25% ??? maybe
    """
    if book:
        text, char_to_ind, ind_to_char, K = read_in_data()
    else:
        text = load_poems()
        full_text = ' '.join(text)

        rng = initializeRNG()

        unique_chars = list(set(full_text))
        K = len(unique_chars)

        char_to_ind = {ch: i for i, ch in enumerate(unique_chars)}
        ind_to_char = {i: ch for i, ch in enumerate(unique_chars)}

    rng = initializeRNG()

    m = [100]
    seq_lengths = [10, 25, 50]
    seq_length = 25
    etas = [0.0001, 0.001, 0.05, 0.01, 0,1]
    eta = 0.001
    L = 2

    GD_params = {"seq_length": seq_length, "num_iterations": 10000, "eta": eta, "beta_1": 0.9, "beta_2": 0.999,
                 "epsilon": 1e-8}

    validition_losses_hl = []
    for hidden_layer_size in m:
        network = init_network_torch(K, hidden_layer_size, rng)
        #network = init_network_torch_2_layer(K, hidden_layer_size, rng, 2)
        if book:
            validation_start_index = int(len(text) * 0.90)
            trained_LSTM, last_validation_loss = AdamOptimizerBook(text, validation_start_index, char_to_ind, ind_to_char,GD_params, L, network, rng, print_loss=True, multilayer_forward=False)
        else:
            validation_indices = random.sample(range(len(text)), 44)
            trained_LSTM, last_validation_loss = AdamOptimizerPoem(text, validation_indices, char_to_ind, ind_to_char, GD_params, L, network, rng, print_loss = True)
        validition_losses_hl.append(last_validation_loss)

    for i in range(len(m)):
        print("Hidden layer size = " + str(m[i])+" had validation loss " + str(validition_losses_hl[i])+ " after " + str(GD_params["num_iterations"]) + " iterations")


if __name__ == "__main__":
    main()

    