import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Тут примеры текстов , так как мне лень датасет искать
texts = [
    "Яблоки растут на деревьях.",
    "Яблоки бывают зелеными и красными.",
    "Яблоки полезны для здоровья.",
    "Яблоки можно есть сырыми или вареными.",
    "Яблоки используются в выпечке."
]

texts = [[ord(char) for char in text] for text in texts]
max_value = max(max(text) for text in texts)

# тут гиперы
input_dim = max_value + 1
hidden_dim = 64
num_heads = 4
num_levels = 2
num_lstm_layers = 1
seq_length = 50
batch_size = 2
num_epochs = 10
learning_rate = 0.001

class AppleTextDataset(Dataset):
    def __init__(self, texts, seq_length):
        self.texts = texts
        self.seq_length = seq_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if len(text) < self.seq_length:
            text += [0] * (self.seq_length - len(text))  # тут пэддинг
        return torch.tensor(text[:self.seq_length], dtype=torch.long)

dataset = AppleTextDataset(texts, seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# тут сама моделька
class MultiLevelAttention(nn.Module):
    def __init__(self, input_dim, num_heads, num_levels):
        super(MultiLevelAttention, self).__init__()
        self.num_levels = num_levels
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(input_dim, num_heads) for _ in range(num_levels)
        ])

    def forward(self, x):
        for attention_layer in self.attention_layers:
            x, _ = attention_layer(x, x, x)
        return x

class DynamicNormalization(nn.Module):
    def __init__(self, input_dim):
        super(DynamicNormalization, self).__init__()
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        return self.norm(x)

class MLATCE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_levels, num_lstm_layers):
        super(MLATCE, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_lstm_layers, batch_first=True)
        self.multi_level_attention = MultiLevelAttention(hidden_dim, num_heads, num_levels)
        self.dynamic_norm = DynamicNormalization(hidden_dim)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        attention_out = self.multi_level_attention(lstm_out)
        normalized_out = self.dynamic_norm(attention_out)
        output = self.fc(normalized_out)
        return output

model = MLATCE(input_dim, hidden_dim, num_heads, num_levels, num_lstm_layers)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output.view(-1, input_dim), batch.view(-1))
        loss.backward()
        optimizer.step()
    print(f'Эпох {epoch+1}/{num_epochs}, Лох: {loss.item()}')
torch.save(model.state_dict(), 'огурец.pth')

total_params = sum(p.numel() for p in model.parameters())
print(f'Общ. Параметров: {total_params}')
