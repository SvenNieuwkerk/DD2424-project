import numpy as np
import torch
import copy
import matplotlib.pyplot as plt
import requests
import re
import random

from fontTools.misc.plistlib import end_data
from spellchecker import SpellChecker

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

def init_network(K, m, rng):
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
    network['U'] = (1 / np.sqrt(2 * K)) * rng.standard_normal(size=(K, 4*m))
    network['W'] = (1 / np.sqrt(2 * m)) * rng.standard_normal(size=(m, 4*m))
    network['V'] = (1 / np.sqrt(m)) * rng.standard_normal(size=(m, K))

    return network

def init_network_torch(K, m, rng):
    numpy_network = init_network(K, m, rng)
    torch_network = {}
    for kk in numpy_network.keys():
        if kk in ('E1', 'E2', 'E3', 'E4'):
            torch_network[kk] = torch.tensor(numpy_network[kk], requires_grad=False)
        else:
            torch_network[kk] = torch.tensor(numpy_network[kk], requires_grad=True)
    return torch_network

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
        if kk in ('E1', 'E2', 'E3', 'E4'):
            continue
        grads[kk] = torch_network[kk].grad.clone()

    return grads

def AdamOptimizer(poems, validation_set, char_to_ind, ind_to_char, GDparams, init_net, rng, print_loss = False, print_text_before = 1000):
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
        newParams['m'] = torch.zeros_like(parameter)
        newParams['v'] = torch.zeros_like(parameter)
        AdamParams[key] = newParams

    #hprev = torch.zeros((1,m), dtype = torch.float64)
    #cprev = torch.zeros((1,m), dtype = torch.float64)
    hprev = torch.zeros(m, dtype = torch.float64)
    cprev = torch.zeros(m, dtype = torch.float64)
    print(cprev.shape)

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
                h0val = torch.zeros(m, dtype=torch.float64)
                c0val = torch.zeros(m, dtype=torch.float64)
                validation_poem = poems[validation_poem_nr]
                X_val,_ = encode_sequence(validation_poem[:len(validation_poem)-1], char_to_ind)
                Y_val, y_val = encode_sequence(validation_poem[1:len(validation_poem)], char_to_ind)
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
            hprev = torch.zeros(m, dtype = torch.float64)
            cprev = torch.zeros(m, dtype = torch.float64)
        else:
            hprev = fp_data_sequence['H'][end - start, :].detach()
            cprev = fp_data_sequence['C'][end - start, :].detach()

        if iter%100==0 and print_loss:
            print("After " + str(iter) + " iterations the smooth loss is " +str(smooth_loss))
        if iter%250==0 and print_loss:
            print("After " + str(iter) + " iterations the validation loss is " +str(validation_losses[-1]))
        if (iter+1)%print_text_before==0:
            print("Synthesised text before " + str((iter+1)) + " update steps")
            print(getTextFromY(synthesiseTextFromRNN(hprev, cprev,200, RNN, rng), ind_to_char))


    plot_smooth_loss(smooth_losses, validation_losses, validation_iterations)

    return RNN, validation_losses[-1]


def main():
    """
    text = "This is a test for mispeling, should retrn seveny percent"
    print(spelling_accuracy(text))

    test_input = "This is a test sentence"
    test_text = "This is a sentence a test"
    print("With n= 2, overlap percentage is ", ngram_overlap(test_text, test_input, 2)) # This should print 60% ??? maybe
    print("With n= 3, overlap percentage is ", ngram_overlap(test_text, test_input, 3)) # This should print 25% ??? maybe
    """

    text = load_poems()
    full_text = ' '.join(text)

    rng = initializeRNG()

    unique_chars = list(set(full_text))
    K = len(unique_chars)

    char_to_ind = {ch: i for i, ch in enumerate(unique_chars)}
    ind_to_char = {i: ch for i, ch in enumerate(unique_chars)}

    rng = initializeRNG()

    m = [25,50,100,200]
    seq_length = 25
    eta = 0.001

    validation_indices = random.sample(range(len(text)), 44)

    GD_params = {"seq_length": seq_length, "num_iterations": 10000, "eta": eta, "beta_1": 0.9, "beta_2": 0.999,
                 "epsilon": 1e-8}

    validition_losses_hl = []
    for hidden_layer_size in m:

        network = init_network_torch(K, hidden_layer_size, rng)

        trained_LSTM, last_validation_loss = AdamOptimizer(text, validation_indices, char_to_ind, ind_to_char, GD_params, network, rng, print_loss = True)
        validition_losses_hl.append(last_validation_loss)

    for i in len(m):
        print("Hidden layer size = " + str(m[i])+" had validation loss " + str(validition_losses_hl[i])+ " after " + str(GD_params["num_iterations"]) + " iterations")


if __name__ == "__main__":
    main()