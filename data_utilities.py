import os
import random  # for random choices in augmentation
import re

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from nltk.corpus import wordnet  # lexical database for synonyms
from torch.utils.data import Dataset


# -------------------
# LOAD THE DATA AND THE CORPUS
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

    lines = poems[-1].splitlines()
    poems[-1] = "\n".join(lines[:-1])

    return poems

def build_corpus_and_vocab(text):
    text = "\n\n".join(text)
    unique_chars = sorted(list(set(text)))
    K = len(unique_chars)
    char_to_ind = {ch: i for i, ch in enumerate(unique_chars)}
    ind_to_char = {i: ch for i, ch in enumerate(unique_chars)}
    return text, char_to_ind, ind_to_char, K

def build_word_vocab(poems):
    text = "\n".join(poems)
    words = text.split()
    vocab = sorted(set(words))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    K = len(word_to_idx)
    return words, word_to_idx, idx_to_word, K

# ---------------

# ---------------
# WITH WORD EMBEDDINGS
class WordDataset(Dataset):
    def __init__(self, words, word_to_idx, seq_len):
        self.seq_len = seq_len
        self.word_to_idx = word_to_idx
        data = [word_to_idx[w] for w in words]
        self.x, self.y = [], []
        for i in range(len(data) - seq_len):
            self.x.append(data[i:i+seq_len])
            self.y.append(data[i+1:i+seq_len+1])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.long)


# WITHOUT WORD EMBEDDINGS
class TextDataset(Dataset):
    def __init__(self, text, char_to_ind, seq_len):
        self.seq_len = seq_len
        self.char_to_ind = char_to_ind
        data = [char_to_ind[ch] for ch in text]
        self.x = []
        self.y = []
        for i in range(len(data) - seq_len):
            self.x.append(data[i:i+seq_len])
            self.y.append(data[i+1:i+seq_len+1])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.long)
# ------------------------


# -----------------------
# DATA AUGMENTATION UTILS
def get_synonyms(word):
    synonyms = set()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            syn = lemma.name().lower().replace('_', ' ')
            if syn != word.lower() and ' ' not in syn:
                synonyms.add(syn)
    return list(synonyms)


def synonym_replacement(words, n):
    new_words = words.copy()
    eligible_indices = [i for i, w in enumerate(words) if get_synonyms(w)]
    random.shuffle(eligible_indices)
    num_replaced = 0

    for idx in eligible_indices:
        if num_replaced >= n:
            break
        synonyms = get_synonyms(words[idx])
        if synonyms:
            new_words[idx] = random.choice(synonyms)
            num_replaced += 1
    return new_words


def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        candidates = [w for w in new_words if get_synonyms(w)]
        if not candidates:
            break
        random_word = random.choice(candidates)
        synonyms = get_synonyms(random_word)
        if not synonyms:
            continue
        random_synonym = random.choice(synonyms)
        insert_pos = random.randint(0, len(new_words))
        new_words.insert(insert_pos, random_synonym)
    return new_words

def random_swap(words, n):
    new_words = words.copy()
    length = len(new_words)
    for _ in range(n):
        idx1, idx2 = random.sample(range(length), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return new_words

def random_deletion(words, p):
    if len(words) == 1:
        return words

    new_words = []
    for w in words:
        if random.random() > p:
            new_words.append(w)
    if not new_words:
        new_words.append(random.choice(words))
    return new_words

def eda(words, sr_ratio, ri_ratio, rs_ratio, rd_prob, num_aug):
    augmented = []
    num_words = len(words)
    n_sr = max(1, int(sr_ratio * num_words))
    n_ri = max(1, int(ri_ratio * num_words))
    n_rs = max(1, int(rs_ratio * num_words))

    for _ in range(num_aug):
        a_words = words.copy()

        operations = ['sr', 'ri', 'rs', 'rd']
        random.shuffle(operations)
        for op in operations:
            if op == 'sr':
                a_words = synonym_replacement(a_words, n_sr)
            elif op == 'ri':
                a_words = random_insertion(a_words, n_ri)
            elif op == 'rs':
                a_words = random_swap(a_words, n_rs)
            elif op == 'rd':
                a_words = random_deletion(a_words, rd_prob)
        augmented.append(a_words)
    return augmented
# ------------------------

# Plotting
def plot_loss(train_losses, validation_losses, validation_iterations, training_iterations):
    plt.figure(figsize=(6, 4))
    # Plot smooth loss
    plt.plot(training_iterations, train_losses, label='Train Loss')
    plt.plot(validation_iterations, validation_losses, label='Validation Loss')
    plt.xlabel('Update Steps')
    plt.ylabel('Train Loss')
    plt.title('Train vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def save_plot_and_losses(train_loss, val_loss, val_iter, train_iter, params, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)

    param_str = f"{params['model']}_bs{params['batch_size']}_hs{params['hidden_size']}_lr{params['lr']}_layers{params['num_layers']}"

    # Save plot
    plt.figure(figsize=(6, 4))
    plt.plot(train_iter, train_loss, label='Train Loss')
    plt.plot(val_iter, val_loss, label='Validation Loss')
    plt.xlabel('Update Steps')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"loss_plot_{param_str}.png")
    plt.savefig(plot_path)
    plt.close()

    # Save raw data
    np.save(os.path.join(save_dir, f"train_loss_{param_str}.npy"), train_loss)
    np.save(os.path.join(save_dir, f"val_loss_{param_str}.npy"), val_loss)
    np.save(os.path.join(save_dir, f"val_iter_{param_str}.npy"), val_iter)
    np.save(os.path.join(save_dir, f"train_iter_{param_str}.npy"), train_iter)

def save_plot_final(train_loss, val_loss, val_iter, train_iter, name, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)

    # Save plot
    plt.figure(figsize=(6, 4))
    plt.plot(train_iter, train_loss, label='Train Loss')
    plt.plot(val_iter, val_loss, label='Validation Loss')
    plt.xlabel('Update Steps')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"loss_plot_{name}.png")
    plt.savefig(plot_path)
    plt.close()

    # Save raw data
    np.save(os.path.join(save_dir, f"train_loss_{name}.npy"), train_loss)
    np.save(os.path.join(save_dir, f"val_loss_{name}.npy"), val_loss)
    np.save(os.path.join(save_dir, f"val_iter_{name}.npy"), val_iter)
    np.save(os.path.join(save_dir, f"train_iter_{name}.npy"), train_iter)