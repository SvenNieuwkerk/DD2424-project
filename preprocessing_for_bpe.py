import requests, re, torch
from tokenizers.implementations import ByteLevelBPETokenizer
from torch.utils.data import Dataset
import unicodedata

def load_poems():
    url = "https://www.gutenberg.org/cache/epub/12242/pg12242.txt"
    r = requests.get(url); r.encoding = "utf-8"
    raw = r.text
    start = "*** START OF THE PROJECT GUTENBERG EBOOK POEMS BY EMILY DICKINSON, THREE SERIES, COMPLETE ***"
    end   = "End of Project Gutenberg's Poems: Three Series, Complete, by Emily Dickinson"
    body = raw[ raw.find(start) + len(start) : raw.find(end) ]
    first = re.search(r'(?m)^[IVXLCDM]+\.\s*$', body)
    body = body[first.start():]
    poems = [p.strip() for p in re.split(r'(?m)^[IVXLCDM]+\.\s*$', body) if len(p.split())>10]

    # fix up a few manual slices
    poems[280] = "\n".join(poems[280].splitlines()[:4])
    poems[114] = "\n".join(poems[114].splitlines()[:9])
    poems[0]   = "\n".join(poems[0].splitlines()[6:])
    return poems

def build_char_vocab(text):
    uniq = sorted(set(text))
    char_to_ind = {ch:i for i,ch in enumerate(uniq)}
    ind_to_char = {i:ch for ch,i in char_to_ind.items()}
    return char_to_ind, ind_to_char, len(uniq)

def preprocces_text_for_byte_pair(poems, out_path="output.txt"):
    text = "\n".join(poems)
    # strip control chars, ensure NFC
    text = unicodedata.normalize("NFC", text)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

def train_tokenizer(input_path="output.txt",
                    vocab_size=10000,
                    min_freq=2,
                    prefix="bpe"):
    tok = ByteLevelBPETokenizer()
    tok.train(
      files=[input_path],
      vocab_size=vocab_size,
      min_frequency=min_freq,
      special_tokens=["<pad>","<s>","</s>","<unk>","<mask>"]
    )
    tok.save_model(".", prefix=prefix)

def load_trained_tokenizer(prefix="bpe"):
    return ByteLevelBPETokenizer(f"{prefix}-vocab.json",
                                 f"{prefix}-merges.txt")

class BPEDataset(Dataset):
    def __init__(self, ids, seq_len):
        self.seq_len = seq_len
        self.ids = ids
        self.x = [ ids[i:i+seq_len] for i in range(len(ids)-seq_len) ]
        self.y = [ ids[i+1:i+seq_len+1] for i in range(len(ids)-seq_len) ]
    def __len__(self):
        return len(self.x)
    def __getitem__(self, i):
        return (torch.tensor(self.x[i],dtype=torch.long),
                torch.tensor(self.y[i],dtype=torch.long))

def prepare_bpe_datasets(seq_len=50,
                         poems=True,
                         out_path="output.txt",
                         bpe_prefix="bpe",
                         vocab_size=3000,
                         min_frequency=2,
                         test_vocab_size = False):
    # --- 1) Load & clean text ---
    if poems:
        poems_list = load_poems()
        full_text  = "\n".join(poems_list)
    else:
        raise ValueError("Non-poems path not implemented")

    # --- 2) Write cleaned text file for BPE training ---
    preprocces_text_for_byte_pair(poems_list, out_path=out_path)

    # --- 3) Train tokenizer (writes bpe-vocab.json & bpe-merges.txt) ---
    train_tokenizer(input_path=out_path,
                    vocab_size=vocab_size,
                    min_freq=min_frequency,
                    prefix=bpe_prefix)

    # --- 4) Load the trained tokenizer and encode full corpus ---
    tokenizer = load_trained_tokenizer(prefix=bpe_prefix)
    encoding  = tokenizer.encode(full_text, add_special_tokens=False)
    ids       = encoding.ids

    # --- 5) Build vocab mappings & split indices ---
    token_to_id = tokenizer.get_vocab()
    if test_vocab_size:
        freqs = sorted(token_to_id.values(), reverse=True)
        # Percent of tokens seen ≥100 times
        percent = sum(1 for f in freqs if f >= 100) / len(freqs) * 100
        print(f"{percent:.1f}% of tokens ≥100 occurrences")
    id_to_token = {i:t for t,i in token_to_id.items()}
    K           = len(token_to_id)

    split_idx_tokens = int(0.9 * len(ids))
    split_idx_chars  = int(0.9 * len(full_text))
    train_ids = ids[:split_idx_tokens]
    val_ids   = ids[split_idx_tokens:]
    train_text = full_text[:split_idx_chars]
    val_text   = full_text[split_idx_chars:]

    # --- 6) Wrap in datasets ---
    train_ds = BPEDataset(train_ids, seq_len)
    val_ds   = BPEDataset(val_ids,   seq_len)

    # --- 7) Return same signature as before ---
    return train_ds, val_ds, token_to_id, id_to_token, K, train_text, tokenizer

prepare_bpe_datasets(test_vocab_size=True)