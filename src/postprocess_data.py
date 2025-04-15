#!/usr/bin/env python3

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import string
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pickle

# Fields to process with conf thresholds
FIELDS_TO_PROCESS = {
    'Rating':0.2,
    'Where_enlisted':0.3,
    'Where_born': 0.2,
    'Occupation':0.4
}

class TextCorrectionDataset(Dataset):
    def __init__(self, noisy_texts, correct_texts, max_len=50, vocab=None):
        self.max_len = max_len
        self.noisy_texts = noisy_texts
        self.correct_texts = correct_texts
        if vocab is None:
            self.vocab = {'<PAD>': 0, '<UNK>': 1, '<EOS>': 2}
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
        
# Load the model
def load_model(model_class, vocab_size, embedding_dim, hidden_dim, max_len, load_path="model_state.pth"):
    # Recreate the model architecture
    model = model_class(vocab_size=vocab_size, embedding_dim=embedding_dim,
                        hidden_dim=hidden_dim, max_len=max_len)
    # Load the saved state_dict
    model.load_state_dict(torch.load(load_path))
    # Set to evaluation mode
    model.eval()
    return model
    
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
        probs = torch.softmax(output, dim=2)  # [batch_size, max_len, vocab_size]
        predicted_indices = torch.argmax(output, dim=2)

        corrected_texts = []
        for pred_seq_i, pred_seq in enumerate(predicted_indices):
            corrected = []
            confidences = []
            eos_idx = None
            for i, idx in enumerate(pred_seq):
                char = dataset.inv_vocab.get(idx.item(), '<UNK>')
                prob = probs[pred_seq_i, i, idx].item()
                corrected.append(char)
                confidences.append(prob)
                if char == '\0':
                    eos_idx = i
                    break

            full_string = "".join(corrected)
            if eos_idx != -1:
                corrected_text = full_string[:eos_idx]
                confidence = sum(confidences[:eos_idx]) / eos_idx  # Average up to <EOS>
            else:
                corrected_text = full_string
                confidence = sum(confidences) / len(confidences)  # Average all if no <EOS>

            #print(f"Corrected Text: {corrected_text}")
            #print(f"Confidence: {confidence}")
            corrected_texts.append((corrected_text, confidence))

        return corrected_texts
        
def load_model_from_files(input_dir_in):
    input_dir = "/home/li1mah/bluejackets/pp_models/"+input_dir_in
    with open(input_dir+"_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    full_dataset = TextCorrectionDataset([], [], max_len=50, vocab=vocab)  # Empty lists if only inferring

    # Load the model
    loaded_model = load_model(
        model_class=Seq2SeqAttention,
        vocab_size=len(full_dataset.vocab),
        embedding_dim=32,  # 32 for TextAutoencoder model
        hidden_dim=128,    # 128 for TextAutoencoder model
        max_len=40, # 40
        load_path=input_dir+"_model.pth"
    )

    return loaded_model, full_dataset

def load_models(models):
    loaded_models = {}
    for model_name, model_path in models.items():
        loaded_models[model_name] = load_model_from_files(model_path)
    return loaded_models 
