import csv
import os
import numpy as np
import torch
import torch.nn as nn
import requests
import re
from torch.utils.data import Dataset, DataLoader
from torch.utils.hipify.hipify_python import value
import matplotlib.pyplot as plt
import nltk  # for WordNet and tokenization
from nltk.corpus import wordnet  # lexical database for synonyms
import random  # for random choices in augmentation
import itertools
from spellchecker import SpellChecker
import time
from preprocessing_for_bpe import prepare_bpe_datasets


# Download necessary NLTK data if not already present
nltk.download('wordnet')  # WordNet lexicon
nltk.download('punkt')    # Tokenizer models for splitting sentences/words


# ------------------------
# PARAMETERS
SEQ_LEN = 25
BATCH_SIZE = 32
HIDDEN_SIZE = 100
EMBEDDING_DIM = 100  # 50, 100, 200, 300
LEARNING_RATE = 0.001
EPOCHS = 20
SAMPLE_INTERVAL = 1000
SAMPLE_LENGTH = 200
PATIENCE = 3
GLOVE_PATH = fr'C:\Users\andre\DD2424-project\glove.6B\glove.6B.{EMBEDDING_DIM}d.txt'
POEMS = True
# AUGMENTATION PARAMETERS
AUGMENT = True
SR_RATIO = 0.1 # proportion of words for synonym replacement
RI_RATIO = 0.1 # proportion of words for random insertion
RS_RATIO = 0.1 # proportion of words for random swap
RD_PROB = 0.1 # probability of random deletion per word
NUM_AUGMENT = 1 # number of augmented versions per poem
# ------------------------

# ------------------------
# Data loading utilities
def read_in_data(seq_length=-1):
    book_dir = r'C:\Users\andre\Desktop\DD2424 - deep learning\Assignment4\goblet_book.txt'
    #book_dir = r'C:\Users\svenr\OneDrive - KTH\Deep Learning in Data Science\Project\DD2424-project\goblet_book.txt'
    book_fname = book_dir
    with open(book_fname, 'r', encoding='utf-8') as fid:
        book_data = fid.read()

    unique_chars = sorted(list(set(book_data)))
    K = len(unique_chars)

    char_to_ind = {ch: i for i, ch in enumerate(unique_chars)}
    ind_to_char = {i: ch for i, ch in enumerate(unique_chars)}

    return book_data, char_to_ind, ind_to_char, K

def load_glove_embeddings(glove_path, word_to_idx):
    embedding_matrix = np.random.normal(scale=0.6, size=(len(word_to_idx), EMBEDDING_DIM))
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            if word in word_to_idx:
                idx = word_to_idx[word]
                embedding_matrix[idx] = vec
    return torch.tensor(embedding_matrix, dtype=torch.float)

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

def build_corpus_and_vocab(text, poems=POEMS):
    if poems:
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
    return words, word_to_idx, idx_to_word

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

# ------------------------
# Data augmentaion utilities
def get_synonyms(word):
    """
    Retrieve a list of one-word synonyms for the given word from WordNet.
    """
    synonyms = set()  # use a set to avoid duplicates
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            syn = lemma.name().lower().replace('_', ' ')
            if syn != word.lower() and ' ' not in syn:
                synonyms.add(syn)
    return list(synonyms)  # convert back to list


def synonym_replacement(words, n):
    """
    Replace up to n random words in 'words' with synonyms.
    """
    new_words = words.copy()
    eligible_indices = [i for i, w in enumerate(words) if get_synonyms(w)]
    random.shuffle(eligible_indices)  # shuffle to pick random words
    num_replaced = 0

    for idx in eligible_indices:
        if num_replaced >= n:
            break
        synonyms = get_synonyms(words[idx])
        if synonyms:
            new_words[idx] = random.choice(synonyms)
            num_replaced += 1
    return new_words  # return modified word list


def random_insertion(words, n):
    """
    Insert up to n synonyms into random positions in the 'words' list.
    """
    new_words = words.copy()
    for _ in range(n):
        # pick a random word that has synonyms
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
    return new_words  # return modified word list

def random_swap(words, n):
    """
    Perform up to n random swaps of two words in the 'words' list.
    """
    new_words = words.copy()
    length = len(new_words)
    for _ in range(n):
        idx1, idx2 = random.sample(range(length), 2)  # pick two distinct indices
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return new_words  # return modified word list

def random_deletion(words, p):
    """
    Randomly delete each word in 'words' with probability p.
    At least one word will remain.
    """
    if len(words) == 1:
        return words  # nothing to delete if only one word

    new_words = []
    for w in words:
        if random.random() > p:
            new_words.append(w)
    if not new_words:
        # ensure at least one word remains
        new_words.append(random.choice(words))
    return new_words  # return modified word list


def eda(words, alpha_sr, alpha_ri, alpha_rs, p_rd, num_aug):
    """
    Generate 'num_aug' augmented word lists from the input 'words' using EDA operations.
    """
    augmented = []
    num_words = len(words)
    n_sr = max(1, int(alpha_sr * num_words))  # at least one replacement if alpha > 0
    n_ri = max(1, int(alpha_ri * num_words))
    n_rs = max(1, int(alpha_rs * num_words))

    for _ in range(num_aug):
        a_words = words.copy()

        # Apply one operation per augmentation (randomly choose order)
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
                a_words = random_deletion(a_words, p_rd)
        augmented.append(a_words)
    return augmented  # list of augmented word lists
# ------------------------


# ------------------------
# Model definition
class TwoLayerLSTMWord(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=2, pretrained_embeddings=None, freeze_embed_epochs=2):
        super(TwoLayerLSTMWord, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        # Weight Tying improves the performance of language models by tying (sharing) 
        # the weights of the embedding and softmax layers. This method also massively reduces the 
        # total number of parameters in the language models that it is applied to.
        self.fc.weight = self.embedding.weight
        self.freeze_embed_epochs = freeze_embed_epochs

    def forward(self, x, hidden=None):
        # x: (batch, seq_len)
        emb = self.embedding(x)  # (batch, seq_len, embedding_dim)
        out, hidden = self.lstm(emb, hidden)
        out = out.contiguous().view(-1, self.hidden_size)
        logits = self.fc(out)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)

# WITHOUT EMBEDDINGS
class TwoLayerLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=2):
        super(TwoLayerLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.lstm = nn.LSTM(input_size=vocab_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        # x: (batch, seq_len, vocab_size)
        out, hidden = self.lstm(x, hidden)
        # out: (batch, seq_len, hidden_size)
        out = out.contiguous().view(-1, self.hidden_size)
        logits = self.fc(out)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        # initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)
# ------------------------


class TwoLayerRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=2):
        super(TwoLayerRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.rnn = nn.RNN(input_size=vocab_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        out, hidden = self.rnn(x, hidden)
        out = out.contiguous().view(-1, self.hidden_size)
        logits = self.fc(out)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return h0  # Only hidden state, no cell state like in LSTM

# ------------------------
# Training and sampling

def sample_word(model, start_word, word_to_idx, idx_to_word, length, device, temperature=1.0, top_p=1.0):
    model.eval()
    input_idx = torch.tensor([[word_to_idx[start_word]]], device=device)
    hidden = model.init_hidden(batch_size=1, device=device)
    generated = [start_word]

    with torch.no_grad():
        for _ in range(length):
            logits, hidden = model(input_idx, hidden)
            logits = logits / temperature
            probs = torch.softmax(logits.squeeze(), dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=0)
            cutoff = (cumulative_probs > top_p).nonzero(as_tuple=True)[0]
            if len(cutoff) > 0:
                cutoff_i = cutoff[0] + 1
            else:
                cutoff_i = len(sorted_probs)
            truncated_probs = sorted_probs[:cutoff_i]
            truncated_indices = sorted_idx[:cutoff_i]
            truncated_probs /= truncated_probs.sum()
            choice = torch.multinomial(truncated_probs, 1).item()
            next_idx = truncated_indices[choice].item()
            next_word = idx_to_word[next_idx]
            generated.append(next_word)
            input_idx = torch.tensor([[next_idx]], device=device)
    return ' '.join(generated)

# WITHOUT EMBEDDINGS
# ------------------------
def sample(model, start_char, char_to_ind, ind_to_char, length, device, temperature = 1, top_p = 1, bpe = False, tokenizer = None):
    model.eval()
    K = model.vocab_size
    input_char = torch.zeros(1, 1, K, device=device)
    input_char[0, 0, char_to_ind[start_char]] = 1
    hidden = model.init_hidden(batch_size=1, device=device)
    generated = [start_char]
    generated_idxs = []
    if bpe:
        generated_idxs.append(char_to_ind[start_char])

    with torch.no_grad():
        for _ in range(length):
            logits, hidden = model(input_char, hidden)

            #Temperature
            logits /= temperature

            probs = torch.softmax(logits, dim=-1).squeeze()

            #Nucleus sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=0)

            #Get the first index with a cumulatve probability higher than top p
            cutoff = (cumulative_probs > top_p).nonzero(as_tuple=True)[0]
            if len(cutoff) > 0:
                cutoff_index = cutoff[0] + 1  # include at least one token
            else:
                cutoff_index = len(sorted_probs)

            truncated_probs = sorted_probs[:cutoff_index]
            truncated_indices = sorted_indices[:cutoff_index]
            truncated_probs /= truncated_probs.sum()

            sampled_index = torch.multinomial(truncated_probs, 1).item()
            char_index = truncated_indices[sampled_index].item()
            if bpe:
                generated_idxs.append(char_index)

            next_char = ind_to_char[char_index]
            generated.append(next_char)

            # prepare next input
            input_char = torch.zeros(1, 1, K, device=device)
            input_char[0, 0, char_index] = 1
    model.train()
    if not bpe:
        return ''.join(generated)
    else:
        return tokenizer.decode(generated_idxs, skip_special_tokens=True)

# ------------------------


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

def prepare_datasets(seq_len=50, poems=True):
    if poems:
        poems = load_poems()
        text, char_to_ind, ind_to_char, K = build_corpus_and_vocab(poems)
    else:
        text, char_to_ind, ind_to_char, K = read_in_data()

    split_idx = int(len(text) * 0.9)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    train_dataset = TextDataset(train_text, char_to_ind, seq_len)
    val_dataset = TextDataset(val_text, char_to_ind, seq_len)
    return train_dataset, val_dataset, char_to_ind, ind_to_char, K, train_text


# ------------------------
# Main training loop
def train_word(seq_len=SEQ_LEN, batch_size=BATCH_SIZE, hidden_size=HIDDEN_SIZE, lr=LEARNING_RATE, 
               epochs=EPOCHS, sample_interval=SAMPLE_INTERVAL, sample_length=SAMPLE_LENGTH, poems=POEMS, 
               augment=AUGMENT, sr_ratio=SR_RATIO, ri_ratio=RI_RATIO, rs_ratio=RS_RATIO, rd_prob=RD_PROB, 
               num_augment=NUM_AUGMENT):
    if poems:
        original_poems = load_poems()
        split_idx = int(len(original_poems) * 0.9)
        train_poems = original_poems[:split_idx]
        val_poems = original_poems[split_idx:]
        if augment: 
            augmented_poems = []
            for poem in train_poems:

                words = nltk.word_tokenize(poem)

                aug_word_lists = eda(words, sr_ratio, ri_ratio, rs_ratio, rd_prob, num_augment)

                for awl in aug_word_lists:
                    augmented_poems.append(' '.join(awl))

            final_poems = train_poems + augmented_poems
        else:
            final_poems = train_poems
        all_words, word_to_idx, idx_to_word = build_word_vocab(final_poems+val_poems)
        train_text = "\n".join(train_poems)
        val_text = "\n".join(val_poems)
        train_words = train_text.split()
        val_words = val_text.split()
    else:
        final_data, __, __, __ = read_in_data()
        words, word_to_idx, idx_to_word = build_word_vocab(final_data)
        split = int(len(words) * 0.9)
        train_words = words[:split]
        val_words = words[split:]

    pretrained = load_glove_embeddings(GLOVE_PATH, word_to_idx)

    # datasets & loaders
    train_dataset = WordDataset(train_words, word_to_idx, seq_len)
    val_dataset = WordDataset(val_words, word_to_idx, seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TwoLayerLSTMWord(vocab_size=len(word_to_idx), embedding_dim=EMBEDDING_DIM,
                             hidden_size=hidden_size, pretrained_embeddings=pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    training_losses = []
    training_iterations = []
    validation_losses = []
    validation_iterations = []

    best_val_loss = float('inf')
    patience_counter = 0
    step = 0

    for epoch in range(1, epochs+1):
        model.train()
        hidden = None
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits, hidden = model(x_batch, hidden)
            hidden = (hidden[0].detach(), hidden[1].detach())
            loss = criterion(logits, y_batch.view(-1))
            loss.backward()
            optimizer.step()

            if step % sample_interval == 0:
                start_word = idx_to_word[np.random.randint(0, len(word_to_idx))]
                sample_text = sample_word(model, start_word, word_to_idx, idx_to_word, sample_length, device)
                print(f"[Epoch {epoch} Step {step}] Loss: {loss.item():.4f}\nSample: {sample_text}\n")
                training_losses.append(loss.item())
                training_iterations.append(step)
            step += 1

        # validation
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits, _ = model(x_batch)
                val_loss += criterion(logits, y_batch.view(-1)).item()
        val_loss /= len(val_loader)
        print(f"==> Epoch {epoch} Val Loss: {val_loss:.4f}\n")
        validation_losses.append(val_loss)
        validation_iterations.append(step)

        # --- EARLY STOPPING ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    return training_losses, validation_losses, validation_iterations, training_iterations


# TODO: THIS ONE IS FROM RNN+GRIDSEARCH, IT HAS THE POEMS ABSTRACTED, SO I KEPT BOTH FOR NOW
# ------------------------
# Main training loop
# ------------------------
def train(train_dataset, val_dataset, char_to_ind, ind_to_char, K,
          seq_len=50, batch_size=25, hidden_size=64, lr=5e-4, epochs=20,
          sample_interval=1000, sample_length=200, num_layers=2, model_type="lstm", bpe = False, tokenizer = None):

    best_val_loss = float('inf')
    patience = 3  # Controls early stopping
    patience_counter = 0

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_type == "lstm":
        model = TwoLayerLSTM(vocab_size=K, hidden_size=hidden_size, num_layers=num_layers).to(device)
    elif model_type == "rnn":
        model = TwoLayerRNN(vocab_size=K, hidden_size=hidden_size, num_layers=num_layers).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    training_losses = []
    training_iterations = []
    validation_losses = []
    validation_iterations = []

    step = 0
    for epoch in range(1, epochs+1):
        model.train()
        hidden = None
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            batch_size_curr = x_batch.size(0)
            # one-hot encode inputs
            x_onehot = torch.zeros(batch_size_curr, seq_len, K, device=device)
            x_onehot.scatter_(2, x_batch.unsqueeze(-1), 1)

            optimizer.zero_grad()
            logits, hidden = model(x_onehot, hidden)
            # detach hidden state to prevent backprop through entire history (both for LSTM and RNN)
            if isinstance(hidden, tuple):
                hidden = (hidden[0].detach(), hidden[1].detach())
            else:
                hidden = hidden.detach()

            loss = criterion(logits, y_batch.view(-1))
            loss.backward()
            optimizer.step()

            if step % sample_interval == 0:
                start_char = ind_to_char[np.random.randint(0, K)]
                if not bpe:
                    sample_text = sample(model, start_char, char_to_ind, ind_to_char, sample_length, device)
                else:
                    sample_text = sample(model, start_char, char_to_ind, ind_to_char, sample_length, device, bpe = bpe, tokenizer = tokenizer)
                print(f"[Epoch {epoch} | Step {step}] Loss: {loss.item():.4f}\nSample: {sample_text}\n")
                training_losses.append(loss.item())
                training_iterations.append(step)
            step += 1

        # validation loss
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                batch_size_curr = x_batch.size(0)
                x_onehot = torch.zeros(batch_size_curr, seq_len, K, device=device)
                x_onehot.scatter_(2, x_batch.unsqueeze(-1), 1)

                logits, _ = model(x_onehot)
                loss_val = criterion(logits, y_batch.view(-1))
                val_loss += loss_val.item()
                val_steps += 1
        val_loss /= val_steps
        print(f"==> Epoch {epoch} validation loss: {val_loss:.4f}\n")
        validation_losses.append(val_loss)
        validation_iterations.append(step)

        # --- EARLY STOPPING ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()   # If we want to restore the model for something, not sure
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    return training_losses, validation_losses, validation_iterations, training_iterations, model, epoch


# TODO: THIS ONE IS FROM MAIN, NICER PARAMETERS, BUT DOESN'T HAVE BUNCH OF NEW STUFF (PROBABLY DOESN'T WORK FOR CUDA EITHER)
# TODO: BUT IT ALSO HAS THE AUGMENT AND STUFF LIKE THAT, SO WE HAVE TO FIGURE OUT WHAT TO DO WITH THAT
def trainMAIN(seq_len=SEQ_LEN, batch_size=BATCH_SIZE, hidden_size=HIDDEN_SIZE, lr=LEARNING_RATE,
          epochs=EPOCHS, sample_interval=SAMPLE_INTERVAL, sample_length=SAMPLE_LENGTH, poems=POEMS,
          augment=AUGMENT, sr_ratio=SR_RATIO, ri_ratio=RI_RATIO, rs_ratio=RS_RATIO, rd_prob=RD_PROB,
          num_augment=NUM_AUGMENT):
    if poems:
        original_poems = load_poems()
        split_idx = int(len(original_poems) * 0.9)
        train_poems = original_poems[:split_idx]
        val_poems = original_poems[split_idx:]
        if augment:
            augmented_poems = []
            for poem in train_poems:

                words = nltk.word_tokenize(poem)

                aug_word_lists = eda(words, sr_ratio, ri_ratio, rs_ratio, rd_prob, num_augment)

                for awl in aug_word_lists:
                    augmented_poems.append(' '.join(awl))

            final_poems = train_poems + augmented_poems
            text, char_to_ind, ind_to_char, K = build_corpus_and_vocab(final_poems + val_poems)
            train_text = "\n\n".join(final_poems)
            val_text = "\n\n".join(val_poems)
        else:
            final_poems = original_poems
            text, char_to_ind, ind_to_char, K = build_corpus_and_vocab(final_poems)

    else:
        text, char_to_ind, ind_to_char, K = read_in_data()
        split_idx = int(len(text) * 0.9)
        train_text = text[:split_idx]
        val_text = text[split_idx:]

    best_val_loss = float('inf')
    patience = PATIENCE  # Controls early stopping
    patience_counter = 0

    train_dataset = TextDataset(train_text, char_to_ind, seq_len)
    val_dataset = TextDataset(val_text, char_to_ind, seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TwoLayerLSTM(vocab_size=K, hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    training_losses = []
    training_iterations = []
    validation_losses = []
    validation_iterations = []

    step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        hidden = None
        for x_batch, y_batch in train_loader:
            batch_size_curr = x_batch.size(0)
            # one-hot encode inputs
            x_onehot = torch.zeros(batch_size_curr, seq_len, K, device=device)
            x_onehot.scatter_(2, x_batch.unsqueeze(-1), 1)

            optimizer.zero_grad()
            logits, hidden = model(x_onehot, hidden)
            # detach hidden state to prevent backprop through entire history
            hidden = (hidden[0].detach(), hidden[1].detach())

            loss = criterion(logits, y_batch.view(-1).to(device))
            loss.backward()
            optimizer.step()

            if step % sample_interval == 0:
                start_char = ind_to_char[np.random.randint(0, K)]
                sample_text = sample(model, start_char, char_to_ind, ind_to_char, sample_length, device)
                print(f"[Epoch {epoch} | Step {step}] Loss: {loss.item():.4f}\nSample: {sample_text}\n")
                training_losses.append(loss.item())
                training_iterations.append(step)
            step += 1

        # validation loss
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                batch_size_curr = x_batch.size(0)
                x_onehot = torch.zeros(batch_size_curr, seq_len, K, device=device)
                x_onehot.scatter_(2, x_batch.unsqueeze(-1), 1)

                logits, _ = model(x_onehot)
                loss_val = criterion(logits, y_batch.view(-1).to(device))
                val_loss += loss_val.item()
                val_steps += 1
        val_loss /= val_steps
        print(f"==> Epoch {epoch} validation loss: {val_loss:.4f}\n")
        validation_losses.append(val_loss)
        validation_iterations.append(step)

        # --- EARLY STOPPING ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()  # If we want to restore the model for something, not sure
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    return training_losses, validation_losses, validation_iterations, training_iterations

# ------------------------

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
    train_ngrams = set(get_ngrams(training_text, n))

    match_count = sum(1 for gram in gen_ngrams if gram in train_ngrams)
    overlap_percentage = match_count / len(gen_ngrams) * 100
    return overlap_percentage


def save_plot_and_losses(train_loss, val_loss, val_iter, train_iter, params, save_dir="results", bpe = False):
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


def main_grid_search(bpe = False):
    start_time = time.time()

    if not bpe:
        train_dataset, val_dataset, char_to_ind, ind_to_char, K, training_text = prepare_datasets(seq_len=50)
    else:
        train_dataset, val_dataset, char_to_ind, ind_to_char, K, training_text, tokenizer = prepare_bpe_datasets(seq_len=50)

    model_type = "test"

    model_types = ["lstm"]
    hidden_sizes = [100]
    lrs = [1e-3]
    batch_sizes = [32]
    num_layers_list = [2]

    if not bpe:
        metrics_file = f"results_{model_type}/metrics_summary.csv"
        os.makedirs(f"results_{model_type}", exist_ok=True)
    else:
        metrics_file = f"bpe/results_{model_type}/metrics_summary.csv"
        os.makedirs(f"bpe/results_{model_type}", exist_ok=True)


    all_combinations = list(itertools.product(model_types, hidden_sizes, lrs, batch_sizes, num_layers_list))
    total_runs = len(all_combinations)

    # Initialize CSV
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Model Type", "Hidden Size", "Learning Rate", "Batch Size", "Num Layers",
                         "Spelling Accuracy (%)", "Bigram Overlap (%)", "Trigram Overlap (%)", "Last Epoch", "Model Path"])

    counter = 1

    for model_name, hidden_size, lr, batch_size, num_layers in all_combinations:
        print(f"\nRunning: {model_name}, H={hidden_size}, LR={lr}, BS={batch_size}, NL={num_layers}")
        print(f"Run {counter} out of {total_runs}")

        if not bpe:
            train_loss, val_loss, val_iter, train_iter, model, last_epoch = train(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                char_to_ind=char_to_ind,
                ind_to_char=ind_to_char,
                K=K,
                hidden_size=hidden_size,
                lr=lr,
                batch_size=batch_size,
                num_layers=num_layers,
                model_type=model_name,
                epochs=20,
            )
        else:
            train_loss, val_loss, val_iter, train_iter, model, last_epoch = train(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                char_to_ind=char_to_ind,
                ind_to_char=ind_to_char,
                K=K,
                hidden_size=hidden_size,
                lr=lr,
                batch_size=batch_size,
                num_layers=num_layers,
                model_type=model_name,
                epochs=20,
                bpe = bpe,
                tokenizer=tokenizer
            )

        params = {
            'model': model_name,
            'hidden_size': hidden_size,
            'lr': lr,
            'batch_size': batch_size,
            'num_layers': num_layers,
            'last_epoch': last_epoch
        }
        if not bpe:
            save_plot_and_losses(train_loss, val_loss, val_iter, train_iter, params, save_dir=f"results_{model_type}")
        else:
            save_plot_and_losses(train_loss, val_loss, val_iter, train_iter, params, save_dir=f"bpe/results_{model_type}")

        # === Sample text from model ===
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #char_to_ind, ind_to_char = model.char_to_ind, model.ind_to_char  # If not saved in model, pass separately
        start_char = np.random.choice(list(char_to_ind.keys()))
        if not bpe:
            generated_text = sample(model, start_char, char_to_ind, ind_to_char, length=200, device=device)
        else:
            generated_text = sample(model, start_char, char_to_ind, ind_to_char, K, device=device, bpe=bpe, tokenizer=tokenizer)


        # === Compute metrics ===
        spell_acc = spelling_accuracy(generated_text)
        bigram_overlap = ngram_overlap(generated_text, training_text, n=2)
        trigram_overlap = ngram_overlap(generated_text, training_text, n=3)

        # === Save model ===
        param_str = f"{model_name}_bs{batch_size}_hs{hidden_size}_lr{lr}_layers{num_layers}"
        if not bpe:
            model_path = f"results_{model_type}/model_{param_str}.pt"
        else:
            model_path = f"bpe/results_{model_type}/model_{param_str}.pt"

        torch.save(model.state_dict(), model_path)

        # === Save metrics ===
        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([model_name, hidden_size, lr, batch_size, num_layers,
                             round(spell_acc, 2), round(bigram_overlap, 2), round(trigram_overlap, 2), last_epoch,
                             model_path])

    elapsed_time = time.time() - start_time  # End timer
    print(f"The whole training took {elapsed_time:.2f} seconds.")

if __name__ == '__main__':
    #train_loss, val_loss, val_iter, train_iter = train(poems=True)
    #plot_loss(train_loss, val_loss, val_iter, train_iter)
    #train(poems=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Will be using ", device)

    #train_loss, val_loss, val_iter, train_iter, model = train()
    main_grid_search(bpe = True)
