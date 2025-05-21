import numpy as np
import torch
import torch.nn as nn


def load_glove_embeddings(glove_path, word_to_idx, embedding_dim):
    embedding_matrix = np.random.normal(scale=0.6, size=(len(word_to_idx), embedding_dim))
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            if word in word_to_idx:
                word = word.lower()
                idx = word_to_idx[word]
                embedding_matrix[idx] = vec
    return torch.tensor(embedding_matrix, dtype=torch.float)


# ------------------------
# Model definition with embeddings
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
        if hidden_size == embedding_dim:
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

# --------------------
# Sampling
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
    model.train()
    return ' '.join(generated)