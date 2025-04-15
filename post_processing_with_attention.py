import torch
import torch.nn as nn
import torch.optim as optim
import random
import string
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pickle
import argparse

# --- Data Generation Functions ---
def perturb_text(text, perturb_prob = 0.8, insertion_prob=0.06, deletion_prob=0.06, substitution_prob=0.085):
    result = list(text)
    char_set = string.ascii_lowercase + " "

    if random.random() > perturb_prob:
        return text
    else:
      i = 0
      while i < len(result):
          if random.random() < deletion_prob:
              result.pop(i)
          else:
              i += 1
      for i in range(len(result)):
          if random.random() < insertion_prob:
              result.insert(i, random.choice(char_set))
      for i in range(len(result)):
          if random.random() < substitution_prob:
              random_char = random.choice(char_set)
              while random_char == result[i]:
                  random_char = random.choice(char_set)
              result[i] = random_char
      return "".join(result)

def generate_dataset(correct_texts, num_samples_per_text=5):
    noisy_texts = []
    labels = []
    for text in correct_texts:
        correct_with_eos = text + "\0"
        for _ in range(num_samples_per_text):
            noisy = perturb_text(text)
            noisy_texts.append(noisy)
            labels.append(correct_with_eos)
    return noisy_texts, labels

# --- Custom Dataset ---
class TextCorrectionDataset(Dataset):
    def __init__(self, noisy_texts, correct_texts, max_len=50, vocab=None):
        self.max_len = max_len
        self.noisy_texts = noisy_texts
        self.correct_texts = correct_texts
        if vocab is None:
            self.vocab = {'<PAD>': 0, '<UNK>': 1, '\0': 2}
            for text in correct_texts + noisy_texts:
                for char in text:
                    if char not in self.vocab:
                        self.vocab[char] = len(self.vocab)
        else:
            self.vocab = vocab
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def __len__(self):
        return len(self.noisy_texts)

    def __getitem__(self, idx):
        noisy = self.noisy_texts[idx]
        correct = self.correct_texts[idx]
        noisy_tensor = self.text_to_tensor(noisy)
        correct_tensor = self.text_to_tensor(correct)
        noisy_len = min(len(noisy), self.max_len)
        return noisy_tensor, correct_tensor, noisy_len

    def text_to_tensor(self, text):
        indices = [self.vocab.get(char, self.vocab['<UNK>']) for char in text[:self.max_len]]
        if len(indices) < self.max_len:
            indices += [self.vocab['<PAD>']] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long)

# --- Autoencoder Model ---
class TextAutoencoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=128, max_len=50):
        super(TextAutoencoder, self).__init__()
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder_lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        encoded, (hidden, cell) = self.encoder_lstm(embedded)
        decoded, _ = self.decoder_lstm(encoded, (hidden, cell))
        output = self.fc(decoded)
        return output

# --- Seq2Seq Model with Attention ---
class Seq2SeqAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, max_len=50):
        super(Seq2SeqAttention, self).__init__()
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, batch_first=True)  # Context included
        self.attention = nn.Linear(hidden_dim * 2, 1)  # For Bahdanau attention
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim

    def forward(self, x, target=None, teacher_forcing_ratio=0.5):
        batch_size = x.size(0)
        seq_len = x.size(1)
        self.dropout = nn.Dropout(p=0.2)
        embedded = self.embedding(x)
        encoder_outputs, (hidden, cell) = self.encoder(embedded)
        encoder_outputs = self.dropout(encoder_outputs)

        decoder_input = torch.zeros(batch_size, 1, dtype=torch.long) # <PAD> or <SOS>
        decoder_outputs = []

        for t in range(self.max_len):
            embedded_input = self.embedding(decoder_input)  # [batch_size, 1, embedding_dim]

            # Attention across all encoder outputs
            attn_input = torch.cat((hidden[-1].unsqueeze(1).repeat(1, seq_len, 1), encoder_outputs), dim=2)
            attn_scores = self.attention(attn_input).squeeze(2)  # [batch_size, seq_len]
            attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(1)  # [batch_size, 1, seq_len]
            context = torch.bmm(attn_weights, encoder_outputs)  # [batch_size, 1, hidden_dim]

            decoder_input_with_context = torch.cat((embedded_input, context), dim=2)
            decoder_output, (hidden, cell) = self.decoder(decoder_input_with_context, (hidden, cell))
            decoder_output = self.dropout(decoder_output)
            output = self.fc(decoder_output.squeeze(1))
            decoder_outputs.append(output)

            if target is not None and random.random() < teacher_forcing_ratio:
                decoder_input = target[:, t].unsqueeze(1)
            else:
                decoder_input = torch.argmax(output, dim=1).unsqueeze(1)

        return torch.stack(decoder_outputs, dim=1)

# --- Training and Validation Function ---
def train_model(model, train_loader, val_loader, full_dataset, num_epochs=20, learning_rate=0.001, save_path="model_state.pth", vocab_save_path="vocab.pkl"):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # Training phase 
        model.train()
        train_loss = 0
        for noisy_batch, correct_batch, noisy_lengths in train_loader:
            noisy_batch = noisy_batch
            correct_batch = correct_batch

            optimizer.zero_grad()
            output = model(noisy_batch, correct_batch)
            output = output.view(-1, output.size(-1))
            correct_batch = correct_batch.view(-1)

            loss = criterion(output, correct_batch) / noisy_batch.size(1)  # Normalize by sequence length
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for noisy_batch, correct_batch, noisy_lengths in val_loader:
                noisy_batch = noisy_batch
                correct_batch = correct_batch
                output = model(noisy_batch)
                output = output.view(-1, output.size(-1))
                correct_batch = correct_batch.view(-1)
                loss = criterion(output, correct_batch) / noisy_batch.size(1)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Save the model state_dict after training
    torch.save(model.state_dict(), save_path)
    with open(vocab_save_path, "wb") as f:
        pickle.dump(full_dataset.vocab, f)
    print(f"Model saved to {save_path}, Vocabulary saved to {vocab_save_path}")

# --- Inference Function ---
def correct_text(model, noisy_texts, dataset):
    model.eval()
    if isinstance(noisy_texts, str):
        noisy_texts = [noisy_texts]
    elif not isinstance(noisy_texts, list):
        raise ValueError("Input must be a string or a list of strings")

    with torch.no_grad():
        noisy_tensors = torch.stack([dataset.text_to_tensor(text) for text in noisy_texts])
        output = model(noisy_tensors)
        predicted_indices = torch.argmax(output, dim=2)

        corrected_texts = []
        for pred_seq in predicted_indices:
            corrected = []
            for idx in pred_seq:
                char = dataset.inv_vocab.get(idx.item(), '<UNK>')
                if char == '\0':
                    break
                if idx.item() != 0:
                    corrected.append(char)
            corrected_texts.append("".join(corrected))

        return corrected_texts[0] if len(noisy_texts) == 1 else corrected_texts

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(
        description=''
    )

    # Add arguments
    parser.add_argument(
      '-i',
      '--input',
      type=str,
      help='Input file',
      required=True
    )
    parser.add_argument(
      '-o',
      '--output',
      type=str,
      help='Output file name',
      required=True
    )

    parser.add_argument(
        '-t',
        '--test',
        type=str,
        nargs='+',
        help='Optional list of strings to test',
        default=None  # Optional argument
    )


    # Parse arguments
    args = parser.parse_args()

    input_file = args.input
    output_string = args.output
    test_list = args.test

    # Read in data from file
    with open(input_file, 'r') as file:
        data = file.read().splitlines()
    # Take all text per line from data after the first space
    correct_texts = [line.split(' ', 1)[1] for line in data]

    # Generate noisy training data
    noisy_texts, labels = generate_dataset(correct_texts[0:25000], num_samples_per_text=5)

    # Split into training and validation sets (80% train, 20% val)
    train_noisy, val_noisy, train_labels, val_labels = train_test_split(
        noisy_texts, labels, test_size=0.2, random_state=42
    )

    # Create datasets with shared vocabulary
    full_dataset = TextCorrectionDataset(noisy_texts, labels, max_len=40)
    train_dataset = TextCorrectionDataset(train_noisy, train_labels, max_len=40, vocab=full_dataset.vocab)
    val_dataset = TextCorrectionDataset(val_noisy, val_labels, max_len=40, vocab=full_dataset.vocab)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Initialize model
    model = Seq2SeqAttention(vocab_size=len(full_dataset.vocab), embedding_dim=32,
                          hidden_dim=128, max_len=40)

    # Train and validate
    train_model(model, train_loader, val_loader, full_dataset, num_epochs=15,save_path=output_string+"_model.pth", vocab_save_path=output_string+"_vocab.pkl")

    # Test correction
    if test_list is not None:
      corrected_list = correct_text(model, test_list, full_dataset)
      print("\nList of Noisy Inputs:")
      for noisy, corrected in zip(test_list, corrected_list):
          print(f"Noisy: {noisy}, Corrected: {corrected}")

# --- Example Usage ---
if __name__ == "__main__":
    main()