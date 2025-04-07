# Pytorch-lgebra-l-neal-UD-
Universidad distrital Francisco José de Caldas 
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# 1. Tokenizador y vocabulario
tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for label, text in data_iter:
        yield tokenizer(text)

train_iter = IMDB(split='train')
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# 2. Procesamiento de datos
def process(text):
    return torch.tensor(vocab(tokenizer(text)), dtype=torch.long)

label_map = {"neg": 0, "pos": 1}

def collate_batch(batch):
    texts, labels = [], []
    for label, text in batch:
        texts.append(process(text))
        labels.append(torch.tensor(label_map[label], dtype=torch.long))
    texts = pad_sequence(texts, batch_first=True)
    labels = torch.tensor(labels)
    return texts, labels

train_iter, test_iter = IMDB(split='train'), IMDB(split='test')
train_loader = DataLoader(list(train_iter), batch_size=16, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(list(test_iter), batch_size=16, shuffle=False, collate_fn=collate_batch)

# 3. Modelo LSTM
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden[-1])

model = SentimentLSTM(len(vocab), embed_dim=64, hidden_dim=128, output_dim=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 4. Entrenamiento
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):
    model.train()
    total_loss = 0
    for text, label in train_loader:
        text, label = text.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(text)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# 5. Evaluación
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for text, label in test_loader:
        text, label = text.to(device), label.to(device)
        output = model(text)
        _, predicted = torch.max(output, 1)
        correct += (predicted == label).sum().item()
        total += label.size(0)

print(f"Precisión en test: {100 * correct / total:.2f}%")
