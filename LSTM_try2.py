import os
import numpy as np
import torch
import torch.nn as nn
import requests
import re
from torch.utils.data import Dataset, DataLoader
from torch.utils.hipify.hipify_python import value
import matplotlib.pyplot as plt

# ------------------------
# PARAMETERS
SEQ_LEN = 25
BATCH_SIZE = 64
HIDDEN_SIZE = 100
EMBEDDING_DIM = 100  # 50, 100, 200, 300
LEARNING_RATE = 0.0001
EPOCHS = 10
SAMPLE_INTERVAL = 1000
SAMPLE_LENGTH = 200
PATIENCE = 3
GLOVE_PATH = fr'C:\Users\andre\DD2424-project\glove.6B\glove.6B.{EMBEDDING_DIM}d.txt'
POEMS = True
# ------------------------


# ------------------------
# Data loading utilities
# ------------------------
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
# Model definition
# ------------------------
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
# Training and sampling
# ------------------------
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
def sample(model, start_char, char_to_ind, ind_to_char, length, device, temperature = 1, top_p = 1):
    model.eval()
    K = model.vocab_size
    input_char = torch.zeros(1, 1, K, device=device)
    input_char[0, 0, char_to_ind[start_char]] = 1
    hidden = model.init_hidden(batch_size=1, device=device)
    generated = [start_char]

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

            next_char = ind_to_char[char_index]
            generated.append(next_char)

            # prepare next input
            input_char = torch.zeros(1, 1, K, device=device)
            input_char[0, 0, char_index] = 1
    return ''.join(generated)


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

# ------------------------
# Main training loop
# ------------------------
def train_word(seq_len=SEQ_LEN, batch_size=BATCH_SIZE, hidden_size=HIDDEN_SIZE,
               lr=LEARNING_RATE, epochs=EPOCHS, sample_interval=SAMPLE_INTERVAL, sample_length=SAMPLE_LENGTH, poems=POEMS):
    if poems:
        poems = load_poems()
    else:
        text, __, __, __ = read_in_data()
    words, word_to_idx, idx_to_word = build_word_vocab(poems)
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

def train(seq_len=SEQ_LEN, batch_size=BATCH_SIZE, hidden_size=HIDDEN_SIZE, lr=LEARNING_RATE, epochs=EPOCHS, sample_interval=SAMPLE_INTERVAL, sample_length=SAMPLE_LENGTH, poems=POEMS):
    # prepare data
    if poems:
        poems = load_poems()
        text, char_to_ind, ind_to_char, K = build_corpus_and_vocab(poems)
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
    for epoch in range(1, epochs+1):
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
            best_model_state = model.state_dict()   # If we want to restore the model for something, not sure
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    return training_losses, validation_losses, validation_iterations, training_iterations

if __name__ == '__main__':
    train_loss, val_loss, val_iter, train_iter = train_word()
    plot_loss(train_loss, val_loss, val_iter, train_iter)

    #train_loss, val_loss, val_iter, train_iter = train()
    #plot_loss(train_loss, val_loss, val_iter, train_iter)