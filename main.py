import csv
import itertools
import time

import nltk  # for WordNet and tokenization
from spellchecker import SpellChecker
from torch.utils.data import DataLoader

from basic_models import *
from data_utilities import *
from embeddings import *
from preprocessing_for_bpe import prepare_bpe_datasets

nltk.download('wordnet')  # WordNet lexicon
nltk.download('punkt')    # Tokenizer models for splitting sentences/words


# ------------------------
# PARAMETERS
MODEL_TYPE = "lstm"
NUM_LAYERS = 2
SEQ_LEN = 50
BATCH_SIZE = 32
HIDDEN_SIZE = 100
EMBEDDING_DIM = 100  # 50, 100, 200, 300
LEARNING_RATE = 0.001
EPOCHS = 30
SAMPLE_INTERVAL = 1000
SAMPLE_LENGTH = 200
PATIENCE = 3
TEMPERATURE = 0.9
TOP_P = 0.95
USE_GLOVE = False # cannot be used at the same time as bpe
GLOVE_PATH_ANDREAS = fr'C:\Users\andre\DD2424-project\glove.6B\glove.6B.{EMBEDDING_DIM}d.txt' # download at: https://nlp.stanford.edu/projects/glove/
GLOVE_PATH = fr'C:\ALL\Univerzita\Master Year 1\2B Deep Learning in Data Science\Project\glove.6B\glove.6B.{EMBEDDING_DIM}d.txt'
USE_BPE = False # cannot be used at the same time as glove
# AUGMENTATION PARAMETERS
AUGMENT = False
SR_RATIO = 0.1 # proportion of words for synonym replacement
RI_RATIO = 0.1 # proportion of words for random insertion
RS_RATIO = 0.1 # proportion of words for random swap
RD_PROB = 0.1 # probability of random deletion per word
NUM_AUGMENT = 1 # number of augmented versions per poem
# ------------------------

# ------------------------

def prepare_datasets(seq_len=50, augment=AUGMENT, sr_ratio=SR_RATIO, ri_ratio=RI_RATIO, rs_ratio=RS_RATIO, rd_prob=RD_PROB, num_augment=NUM_AUGMENT):
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
        text, char_to_ind, ind_to_char, K = build_corpus_and_vocab(original_poems)
        train_text = "\n\n".join(train_poems)
        val_text = "\n\n".join(val_poems)

    train_dataset = TextDataset(train_text, char_to_ind, seq_len)
    val_dataset = TextDataset(val_text, char_to_ind, seq_len)
    return train_dataset, val_dataset, char_to_ind, ind_to_char, K, train_text

def prepare_datasets_word(seq_len=SEQ_LEN, augment=AUGMENT, sr_ratio=SR_RATIO, ri_ratio=RI_RATIO, rs_ratio=RS_RATIO, rd_prob=RD_PROB, num_augment=NUM_AUGMENT):
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
    all_words, word_to_idx, idx_to_word, K = build_word_vocab(final_poems+val_poems)
    train_text = "\n".join(train_poems)
    val_text = "\n".join(val_poems)
    train_words = train_text.split()
    val_words = val_text.split()
    train_dataset = WordDataset(train_words, word_to_idx, seq_len)
    val_dataset = WordDataset(val_words, word_to_idx, seq_len)
    return train_dataset, val_dataset, word_to_idx, idx_to_word, K, train_text


# ------------------------
# Main training loop
# ------------------------
def train(train_dataset, val_dataset, char_to_ind, ind_to_char, K,
          seq_len=SEQ_LEN, batch_size=BATCH_SIZE, hidden_size=HIDDEN_SIZE, lr=LEARNING_RATE, epochs=EPOCHS,
          sample_interval=SAMPLE_INTERVAL, sample_length=SAMPLE_LENGTH, num_layers=NUM_LAYERS, model_type=MODEL_TYPE,
          bpe=USE_BPE, tokenizer = None, glove=USE_GLOVE, word_to_idx=None, idx_to_word=None):

    best_val_loss = float('inf')
    patience = 3  # Controls early stopping
    patience_counter = 0

    best_model_state = None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_type == "lstm":
        if glove:
            pretrained = load_glove_embeddings(GLOVE_PATH, word_to_idx, EMBEDDING_DIM)
            model = TwoLayerLSTMWord(vocab_size=len(word_to_idx), embedding_dim=EMBEDDING_DIM,
                                     hidden_size=hidden_size, pretrained_embeddings=pretrained).to(device)
        else:
            model = TwoLayerLSTM(vocab_size=K, hidden_size=hidden_size, num_layers=num_layers).to(device)
    elif model_type == "rnn":
        if glove:
            print("Not implemented yet")
        else:
            model = TwoLayerRNN(vocab_size=K, hidden_size=hidden_size, num_layers=num_layers).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    training_losses = []
    training_iterations = []
    validation_losses = []
    validation_iterations = []

    # validation loss at the start
    model.eval()
    val_loss = 0.0
    val_steps = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            batch_size_curr = x_batch.size(0)
            if not glove:
                x_onehot = torch.zeros(batch_size_curr, seq_len, K, device=device)
                x_onehot.scatter_(2, x_batch.unsqueeze(-1), 1)

                logits, _ = model(x_onehot)
            else:
                logits, _ = model(x_batch)
            loss_val = criterion(logits, y_batch.view(-1))
            val_loss += loss_val.item()
            val_steps += 1
    val_loss /= val_steps
    print(f"\n==> Starting validation loss: {val_loss:.4f}\n")
    validation_losses.append(val_loss)
    validation_iterations.append(0)

    step = 0
    for epoch in range(1, epochs+1):
        model.train()
        if glove:
            if epoch <= model.freeze_embed_epochs:
                model.embedding.weight.requires_grad = False
            else:
                model.embedding.weight.requires_grad = True
        hidden = None
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            batch_size_curr = x_batch.size(0)
            optimizer.zero_grad()

            # one-hot encode inputs
            if not glove:
                x_onehot = torch.zeros(batch_size_curr, seq_len, K, device=device)
                x_onehot.scatter_(2, x_batch.unsqueeze(-1), 1)
                logits, hidden = model(x_onehot, hidden)
            else:
                logits, hidden = model(x_batch, hidden)

            # detach hidden state to prevent backprop through entire history (both for LSTM and RNN)
            if isinstance(hidden, tuple):
                hidden = (hidden[0].detach(), hidden[1].detach())
            else:
                hidden = hidden.detach()

            loss = criterion(logits, y_batch.view(-1))
            loss.backward()
            optimizer.step()


            if step % sample_interval == 0:
                if bpe:
                    start_char = ind_to_char[np.random.randint(0, K)]
                    sample_text = sample(model, start_char, char_to_ind, ind_to_char, sample_length, device,  temperature = TEMPERATURE, top_p = TOP_P, bpe = bpe, tokenizer = tokenizer)
                elif glove:
                    start_word = idx_to_word[np.random.randint(0, len(word_to_idx))]
                    sample_text = sample_word(model, start_word, word_to_idx, idx_to_word, sample_length, device, temperature = TEMPERATURE, top_p = TOP_P)
                else:
                    start_char = ind_to_char[np.random.randint(0, K)]
                    sample_text = sample(model, start_char, char_to_ind, ind_to_char, sample_length, device,  temperature = TEMPERATURE, top_p = TOP_P, bpe = False)
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
                if not glove:
                    x_onehot = torch.zeros(batch_size_curr, seq_len, K, device=device)
                    x_onehot.scatter_(2, x_batch.unsqueeze(-1), 1)

                    logits, _ = model(x_onehot)
                else:
                    logits, _ = model(x_batch)

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
    # Restore best model state before returning
    if best_model_state:
        model.load_state_dict(best_model_state)
    return training_losses, validation_losses, validation_iterations, training_iterations, model, epoch

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


def main_grid_search(bpe = USE_BPE, glove=USE_GLOVE, augment=AUGMENT):
    start_time = time.time()

    if bpe:
        train_dataset, val_dataset, char_to_ind, ind_to_char, K, training_text, tokenizer = prepare_bpe_datasets(seq_len=50, augment=AUGMENT)
    elif glove:
        train_dataset, val_dataset, word_to_idx, idx_to_word, K, training_text = prepare_datasets_word(seq_len=50, augment=AUGMENT)
    else:
        train_dataset, val_dataset, char_to_ind, ind_to_char, K, training_text = prepare_datasets(seq_len=50, augment=AUGMENT)

    model_path = f"A={AUGMENT}_BPE={USE_BPE}_GLOVE={USE_GLOVE}"

    model_types = ["lstm", "rnn"]
    hidden_sizes = [100]
    lrs = [1e-3]
    batch_sizes = [32]
    num_layers_list = [2]

    if bpe:
        metrics_file = f"bpe/results_{model_path}/metrics_summary.csv"
        os.makedirs(f"bpe/results_{model_path}", exist_ok=True)
    elif glove:
        metrics_file = f"glove/results_{model_path}/metrics_summary.csv"
        os.makedirs(f"glove/results_{model_path}", exist_ok=True)
    else:
        metrics_file = f"results_{model_path}/metrics_summary.csv"
        os.makedirs(f"results_{model_path}", exist_ok=True)


    all_combinations = list(itertools.product(model_types, hidden_sizes, lrs, batch_sizes, num_layers_list))
    total_runs = len(all_combinations)

    # Initialize CSV
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Model Type", "Hidden Size", "Learning Rate", "Batch Size", "Num Layers",
                         "Spelling Accuracy (%)", "Bigram Overlap (%)", "Trigram Overlap (%)", "Last Epoch", "Last_val_loss", "Best_val_loss", "Model Path"])

    counter = 1

    for model_name, hidden_size, lr, batch_size, num_layers in all_combinations:
        print(f"\nRunning: {model_name}, H={hidden_size}, LR={lr}, BS={batch_size}, NL={num_layers}, glove={glove}, bpe={bpe}, augment={augment}")
        print(f"Run {counter} out of {total_runs}")

        if bpe:
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
                epochs=EPOCHS,
                bpe = bpe,
                tokenizer=tokenizer,
                glove=False,
                word_to_idx = None,
                idx_to_word = None
            )
        elif glove:
            train_loss, val_loss, val_iter, train_iter, model, last_epoch = train(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                char_to_ind=None,
                ind_to_char=None,
                K=K,
                hidden_size=hidden_size,
                lr=lr,
                batch_size=batch_size,
                num_layers=num_layers,
                model_type=model_name,
                epochs=EPOCHS,
                bpe = bpe,
                tokenizer=None,
                glove=True,
                word_to_idx=word_to_idx,
                idx_to_word=idx_to_word
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
                epochs=EPOCHS,
                bpe=False,
                tokenizer=None,
                glove=False,
                word_to_idx=None,
                idx_to_word=None
            )

        params = {
            'model': model_name,
            'hidden_size': hidden_size,
            'lr': lr,
            'batch_size': batch_size,
            'num_layers': num_layers,
            'last_epoch': last_epoch
        }
        if bpe:
            save_plot_and_losses(train_loss, val_loss, val_iter, train_iter, params, save_dir=f"bpe/results_{model_path}/{model_name}")
        elif glove:
            save_plot_and_losses(train_loss, val_loss, val_iter, train_iter, params, save_dir=f"glove/results_{model_path}/{model_name}")
        else:
            save_plot_and_losses(train_loss, val_loss, val_iter, train_iter, params, save_dir=f"results_{model_path}/{model_name}")

        # === Sample text from model ===
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #char_to_ind, ind_to_char = model.char_to_ind, model.ind_to_char  # If not saved in model, pass separately

        num_samples = 20
        total_spell_acc = 0.0
        total_bigram_overlap = 0.0
        total_trigram_overlap = 0.0

        for i in range(1, num_samples + 1):
            if bpe:
                start_char = np.random.choice(list(char_to_ind.keys()))
                generated_text = sample(model, start_char, char_to_ind, ind_to_char, K, device=device,
                                        temperature=TEMPERATURE, top_p=TOP_P, bpe=bpe, tokenizer=tokenizer)
            elif glove:
                start_word = idx_to_word[np.random.randint(0, len(word_to_idx))]
                generated_text = sample_word(model, start_word, word_to_idx, idx_to_word, length=200, device=device,
                                             temperature=TEMPERATURE, top_p=TOP_P)
            else:
                start_char = np.random.choice(list(char_to_ind.keys()))
                generated_text = sample(model, start_char, char_to_ind, ind_to_char, length=200, device=device,
                                        temperature=TEMPERATURE, top_p=TOP_P, bpe=False)

            spell_acc = spelling_accuracy(generated_text)
            bigram_overlap = ngram_overlap(generated_text, training_text, n=2)
            trigram_overlap = ngram_overlap(generated_text, training_text, n=3)

            total_spell_acc += spell_acc
            total_bigram_overlap += bigram_overlap
            total_trigram_overlap += trigram_overlap

            print(f"Generated sample {i}/{num_samples}")

        # === Compute metrics ===
        spell_acc = total_spell_acc / num_samples
        bigram_overlap = total_bigram_overlap / num_samples
        trigram_overlap = total_trigram_overlap / num_samples

        # === Save model ===
        param_str = f"{model_name}_bs{batch_size}_hs{hidden_size}_lr{lr}_layers{num_layers}"
        if bpe:
            model_path_temp = f"bpe/results_{model_path}/{model_name}/model_{param_str}.pt"
        elif glove:
            model_path_temp = f"glove/results_{model_path}/{model_name}/model_{param_str}.pt"
        else:
            model_path_temp = f"results_{model_path}/{model_name}/model_{param_str}.pt"

        torch.save(model.state_dict(), model_path_temp)

        last_val = val_loss[-1]
        best_val = min(val_loss)

        # === Save metrics ===
        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([model_name, hidden_size, lr, batch_size, num_layers,
                             round(spell_acc, 2), round(bigram_overlap, 2), round(trigram_overlap, 2), last_epoch,
                             round(last_val, 3), round(best_val, 3), model_path])

        counter += 1

    elapsed_time = time.time() - start_time  # End timer
    print(f"The whole training took {elapsed_time:.2f} seconds.")

def grid_search_for_nucleus_and_temperature():
    _, _, char_to_ind, ind_to_char, K, training_text = prepare_datasets(seq_len=50, augment=False)
    # Load the best model from grid search for each config
    one_rnn = TwoLayerRNN(vocab_size=K, hidden_size=100, num_layers=1).to(device)
    one_lstm = TwoLayerLSTM(vocab_size=K, hidden_size=256, num_layers=1).to(device)
    two_rnn = TwoLayerRNN(vocab_size=K, hidden_size=100, num_layers=2).to(device)
    two_lstm = TwoLayerLSTM(vocab_size=K, hidden_size=64, num_layers=2).to(device)

    one_rnn.load_state_dict(torch.load("results_step2/model_rnn_bs32_hs100_lr0.001_layers1.pt"))
    one_lstm.load_state_dict(torch.load("results_step2/model_lstm_bs32_hs256_lr0.001_layers1.pt"))
    two_rnn.load_state_dict(torch.load("results_step2/model_rnn_bs32_hs100_lr0.001_layers2.pt"))
    two_lstm.load_state_dict(torch.load("results_step2/model_lstm_bs32_hs64_lr0.001_layers2.pt"))

    models = [
        ("rnn_1layer", one_rnn, 100, 1),
        ("lstm_1layer", one_lstm, 256, 1),
        ("rnn_2layer", two_rnn, 100, 2),
        ("lstm_2layer", two_lstm, 64, 2),
    ]

    generated_texts = {}

    models = [("lstm_2layer", two_lstm, 64, 2)]

    temperatures = [0.6, 0.8, 0.9, 1, 1.1, 1.2]
    nucleus_probs = [0.85, 0.9, 0.95, 0.99]

    metrics_file = "results_n_t_best_models_delete/metrics.csv"
    os.makedirs("results_n_t_best_models_delete", exist_ok=True)

    # Initialize CSV
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Hidden Size", "Num Layers", "Temperature", "Nucleus", "Spelling Accuracy (%)", "Bigram Overlap (%)", "Trigram Overlap (%)"])

    # For all combinations, sample, keep track of results
    for model_idx, (model_name, model, hidden_size, num_layers) in enumerate(models):
        print(
            f"\n Evaluating model {model_idx + 1}/{len(models)}: {model_name}, hidden_size={hidden_size}, layers={num_layers}")

        for temp_idx, temp in enumerate(temperatures):
            for nucleus_idx, nucleus in enumerate(nucleus_probs):
                print(f"Trying model: {model_name}, Temp: {temp}, Nucleus: {nucleus}")


                start_char = np.random.choice(list(char_to_ind.keys()))
                generated_text = sample(
                    model, start_char, char_to_ind, ind_to_char,
                    length=200, device=device,
                    temperature=temp, top_p=nucleus
                )

                # Store the generated text
                key = (model_name, temp, nucleus)
                generated_texts[key] = generated_text

                num_samples = 20
                total_spell_acc = 0.0
                total_bigram_overlap = 0.0
                total_trigram_overlap = 0.0

                for i in range(1, num_samples + 1):
                    start_char = np.random.choice(list(char_to_ind.keys()))
                    generated_text = sample(
                        model, start_char, char_to_ind, ind_to_char,
                        length=200, device=device,
                        temperature=temp, top_p=nucleus
                    )

                    spell_acc = spelling_accuracy(generated_text)
                    bigram_overlap = ngram_overlap(generated_text, training_text, n=2)
                    trigram_overlap = ngram_overlap(generated_text, training_text, n=3)

                    total_spell_acc += spell_acc
                    total_bigram_overlap += bigram_overlap
                    total_trigram_overlap += trigram_overlap

                    print(f"Generated sample {i}/{num_samples}")

                spell_acc = total_spell_acc / num_samples
                bigram_overlap = total_bigram_overlap / num_samples
                trigram_overlap = total_trigram_overlap / num_samples

                # Save metrics
                with open(metrics_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([model_name, hidden_size, num_layers, temp, nucleus,
                                     round(spell_acc, 2), round(bigram_overlap, 2), round(trigram_overlap, 2)])
    print("\nGenerated Samples:")
    for (model_name, temp, nucleus), text in generated_texts.items():
        print(f"\n {model_name} | Temp: {temp} | Nucleus: {nucleus}")
        print(text)


def evaluate_best_model_with_configs():
    start_time = time.time()

    configs = [
        {"bpe": False, "glove": False, "augment": False},
        {"bpe": False, "glove": False, "augment": True},
        {"bpe": True, "glove": False, "augment": False},
        {"bpe": False, "glove": True, "augment": False},
        {"bpe": True, "glove": False, "augment": True},
        {"bpe": False, "glove": True, "augment": True},
    ]

    # Best model config (replace these with your best found values)
    best_model_type = "lstm"
    best_hidden_size = 64
    best_lr = 0.001
    best_batch_size = 32
    best_num_layers = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temperature = TEMPERATURE
    top_p = TOP_P
    max_length = 200

    # Initialize CSV + text log
    metrics_file = "results_best_model_configs/metrics.csv"
    os.makedirs("results_best_model_configs", exist_ok=True)
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["BPE", "GloVe", "Augment", "Spelling Accuracy", "Bigram Overlap", "Trigram Overlap", "Last Epoch", "Last_val_loss", "Best_val_loss", "Model Path"])

    generated_texts = {}

    for i, cfg in enumerate(configs):
        bpe, glove, augment = cfg["bpe"], cfg["glove"], cfg["augment"]
        print(f"\nRunning config {i+1}/6: BPE={bpe}, GloVe={glove}, Augment={augment}")

        # Dataset prep
        if bpe:
            train_dataset, val_dataset, char_to_ind, ind_to_char, K, training_text, tokenizer = prepare_bpe_datasets(seq_len=50, augment=augment)
        elif glove:
            train_dataset, val_dataset, word_to_idx, idx_to_word, K, training_text = prepare_datasets_word(seq_len=50, augment=augment)
        else:
            train_dataset, val_dataset, char_to_ind, ind_to_char, K, training_text = prepare_datasets(seq_len=50, augment=augment)

        # Train
        train_loss, val_loss, val_iter, train_iter, model, last_epoch = train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            char_to_ind=None if glove else char_to_ind,
            ind_to_char=None if glove else ind_to_char,
            K=K,
            hidden_size=best_hidden_size,
            lr=best_lr,
            batch_size=best_batch_size,
            num_layers=best_num_layers,
            model_type=best_model_type,
            epochs=EPOCHS,
            bpe=bpe,
            tokenizer=tokenizer if bpe else None,
            glove=glove,
            word_to_idx=word_to_idx if glove else None,
            idx_to_word=idx_to_word if glove else None
        )

        # Sample
        if glove:
            start_word = idx_to_word[np.random.randint(0, len(word_to_idx))]
            generated_text = sample_word(model, start_word, word_to_idx, idx_to_word, length=max_length, device=device, temperature=temperature, top_p=top_p)
        else:
            start_char = np.random.choice(list(char_to_ind.keys()))
            generated_text = sample(model, start_char, char_to_ind, ind_to_char, length=max_length, device=device, temperature=temperature, top_p=top_p, bpe=bpe, tokenizer=tokenizer if bpe else None)

        # Save model
        model_id = f"bpe={bpe}_glove={glove}_aug={augment}"
        model_path = f"results_best_model_configs/model_{model_id}.pt"
        torch.save(model.state_dict(), model_path)

        # Store generated text
        generated_texts[model_id] = generated_text

        num_samples = 20
        total_spell_acc = 0.0
        total_bigram_overlap = 0.0
        total_trigram_overlap = 0.0

        for i in range(1, num_samples + 1):
            # Sample
            if glove:
                start_word = idx_to_word[np.random.randint(0, len(word_to_idx))]
                generated_text = sample_word(model, start_word, word_to_idx, idx_to_word, length=max_length,
                                             device=device, temperature=temperature, top_p=top_p)
            else:
                start_char = np.random.choice(list(char_to_ind.keys()))
                generated_text = sample(model, start_char, char_to_ind, ind_to_char, length=max_length, device=device,
                                        temperature=temperature, top_p=top_p, bpe=bpe,
                                        tokenizer=tokenizer if bpe else None)

            spell_acc = spelling_accuracy(generated_text)
            bigram_overlap = ngram_overlap(generated_text, training_text, n=2)
            trigram_overlap = ngram_overlap(generated_text, training_text, n=3)

            total_spell_acc += spell_acc
            total_bigram_overlap += bigram_overlap
            total_trigram_overlap += trigram_overlap

            print(f"Generated sample {i}/{num_samples}")

        spell_acc = total_spell_acc / num_samples
        bigram_overlap = total_bigram_overlap / num_samples
        trigram_overlap = total_trigram_overlap / num_samples

        # Save plot
        name = f"BPE={bpe}_GloVe={glove}_Augment={augment}"
        save_plot_final(train_loss, val_loss, val_iter, train_iter, name,
                             save_dir=f"results_best_model_configs")

        last_val = val_loss[-1]
        best_val = min(val_loss)

        # Save metrics
        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([bpe, glove, augment, round(spell_acc, 2), round(bigram_overlap, 2), round(trigram_overlap, 2), last_epoch, last_val, best_val, model_path])

    # Save all generated texts
    with open("results_best_model_configs/generated_texts_summary.txt", "w") as f:
        for model_id, text in generated_texts.items():
            f.write(f"\nModel Config: {model_id}: \n{text}\n")

    elapsed_time = time.time() - start_time  # End timer
    print(f"The whole training took {elapsed_time:.2f} seconds.")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Will be using", device)

    # main_grid_search()

    # grid_search_for_nucleus_and_temperature()

    evaluate_best_model_with_configs()
