import requests
import re

from tokenizers.implementations import ByteLevelBPETokenizer

def load_poems():
    url = "https://www.gutenberg.org/cache/epub/12242/pg12242.txt"
    r = requests.get(url)
    r.encoding = 'utf-8'
    raw = r.text

    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK POEMS BY EMILY DICKINSON, THREE SERIES, COMPLETE ***"
    end_marker   = "End of Project Gutenberg's Poems: Three Series, Complete, by Emily Dickinson"

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

def build_corpus_and_vocab(poems):
    text = "\n".join(poems)
    unique_chars = sorted(list(set(text)))
    K = len(unique_chars)
    char_to_ind = {ch: i for i, ch in enumerate(unique_chars)}
    ind_to_char = {i: ch for i, ch in enumerate(unique_chars)}
    return text, char_to_ind, ind_to_char, K

def preprocces_text_for_byte_pair():
    poems = load_poems()
    text, char_to_ind, ind_to_char, K = build_corpus_and_vocab(poems)
    with open("output.txt", "w", encoding="utf-8") as file:
        file.write(text)

def get_training_corpus():
    with open("your_text.txt", encoding="utf-8") as f:
        for line in f:
            yield line.strip()

def train_tokenizer():
    # 1. Instantiate
    tokenizer = ByteLevelBPETokenizer()

    # 2. Train on your data
    tokenizer.train(
        files=[r"C:\Users\svenr\OneDrive - KTH\Deep Learning in Data Science\Project\DD2424-project\output.txt"],  # list of text files
        vocab_size=10000,  # target vocabulary size
        min_frequency=2,  # drop subwords seen fewer times
        special_tokens=[  # reserve these tokens
            "<pad>", "<s>", "</s>", "<unk>", "<mask>"
        ]
    )

    # 3. Save vocab and merges files
    tokenizer.save_model(
        "./",  # directory to write to
        prefix="bpe"  # produces bpe-vocab.json, bpe-merges.txt
    )

def test_tokenizer():
    # After training and saving:
    tokenizer = ByteLevelBPETokenizer(
        "bpe-vocab.json",
        "bpe-merges.txt"
    )
    enc = tokenizer.encode("To be or not to be.")
    print(enc.tokens)  # eg: ['To', 'Ġbe', 'Ġor', ...]
    print(enc.ids)  # e.g. [675, 12, 89, ...]

preprocces_text_for_byte_pair()
train_tokenizer()
test_tokenizer()