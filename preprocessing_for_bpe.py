import unicodedata
from tokenizers.implementations import ByteLevelBPETokenizer
import nltk
nltk.download('punkt_tab')

from data_utilities import *

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
                         out_path="output.txt",
                         bpe_prefix="bpe",
                         vocab_size=3000,
                         min_frequency=2,
                         test_vocab_size = False,
                         augment = False,
                         sr_ratio=0.1,
                         ri_ratio=0.1,
                         rs_ratio=0.1,
                         rd_prob=0.1,
                         num_augment=1):

    original_poems = load_poems()
    split_idx = int(0.9 * len(original_poems))
    train_poems = original_poems[:split_idx]
    val_poems   = original_poems[split_idx:]


    if augment:
        augmented_poems = []
        for poem in train_poems:
            words = nltk.word_tokenize(poem)

            aug_word_lists = eda(
                words,
                sr_ratio=sr_ratio,
                ri_ratio=ri_ratio,
                rs_ratio=rs_ratio,
                rd_prob=rd_prob,
                num_aug=num_augment
            )

            for awl in aug_word_lists:
                augmented_poems.append(" ".join(awl))


        final_train_poems = train_poems + augmented_poems
    else:
        final_train_poems = train_poems

    full_text_original  = "\n".join(original_poems)
    train_text = "\n".join(final_train_poems)
    val_text   = "\n".join(val_poems)


    preprocces_text_for_byte_pair(final_train_poems, out_path=out_path)

    # Train tokenizer (writes bpe-vocab.json & bpe-merges.txt) ---
    train_tokenizer(input_path=out_path,
                    vocab_size=vocab_size,
                    min_freq=min_frequency,
                    prefix=bpe_prefix)

    tokenizer = load_trained_tokenizer(prefix=bpe_prefix)
    train_encoding = tokenizer.encode(train_text, add_special_tokens=False)
    val_encoding = tokenizer.encode(val_text, add_special_tokens=False)
    train_ids = train_encoding.ids
    val_ids = val_encoding.ids

    token_to_id = tokenizer.get_vocab()
    if test_vocab_size:
        freqs = sorted(token_to_id.values(), reverse=True)
        # Percent of tokens seen ≥100 times
        percent = sum(1 for f in freqs if f >= 100) / len(freqs) * 100
        print(f"{percent:.1f}% of tokens ≥100 occurrences")
    id_to_token = {i:t for t,i in token_to_id.items()}
    K           = len(token_to_id)

    train_ds = BPEDataset(train_ids, seq_len)
    val_ds   = BPEDataset(val_ids,   seq_len)

    return train_ds, val_ds, token_to_id, id_to_token, K, train_text, tokenizer

#OLD PREPARE DATASET
def prepare_bpe_datasets_old(seq_len=50,
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

#prepare_bpe_datasets(test_vocab_size=True)