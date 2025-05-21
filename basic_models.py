import torch
import torch.nn as nn


# ------------------------
# Model definition without embeddings for LSTM
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
# Model definition without embeddings for RNN
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
# Sampling, also supports BPE
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

            #Get the first index with a cumulative probability higher than top p
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

