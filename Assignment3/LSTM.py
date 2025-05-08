import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

SEQ_LEN = 24
BATCH_SIZE = 64
EPOCHS = 20
LR = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def load_and_scale(file, scaler=None, encoder=None, fit=False):
    df = pd.read_csv(file)
    df = df[['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']].dropna()
    numeric_features = df[['pollution', 'dew', 'temp', 'press', 'wnd_spd', 'snow', 'rain']].values
    wind_dir = df[['wnd_dir']].values  #保持二维结构

    if fit:
        scaler = MinMaxScaler()
        numeric_scaled = scaler.fit_transform(numeric_features)

        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        wind_encoded = encoder.fit_transform(wind_dir)

        data = np.concatenate([numeric_scaled, wind_encoded], axis=1)
        return data, scaler, encoder
    else:
        numeric_scaled = scaler.transform(numeric_features)
        wind_encoded = encoder.transform(wind_dir)

        data = np.concatenate([numeric_scaled, wind_encoded], axis=1)
        return data

class PollutionDataset(Dataset):
    def __init__(self, data, seq_len):
        self.X = []
        self.y = []
        for i in range(len(data) - seq_len):
            self.X.append(data[i:i+seq_len])
            self.y.append(data[i+seq_len][0])  #pollution
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_data, scaler, encoder = load_and_scale('LSTM-Multivariate_pollution.csv', fit=True)
test_data = load_and_scale('pollution_test_data1.csv', scaler, encoder=encoder)

train_dataset = PollutionDataset(train_data, SEQ_LEN)
test_dataset = PollutionDataset(test_data, SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  #最后一个时间步的输出
        return self.fc(out)

model = LSTMModel(input_size=11, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(train_loader):.4f}")

model.eval()
predictions = []
actuals = []

predictions = []
actuals = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        output = model(X_batch)
        predictions.append(output)
        actuals.append(y_batch)

predictions = torch.cat(predictions, dim=0)
actuals = torch.cat(actuals, dim=0)
criterion = torch.nn.MSELoss()
mse = criterion(predictions, actuals)
print(f'Mean Squared Error: {mse.item():.4f}')
predictions = predictions.cpu().numpy()
actuals = actuals.cpu().numpy()

#反归一化
pollution_idx = 0
pred_inv = scaler.inverse_transform(np.hstack([predictions, np.zeros((len(predictions), 6))]))[:, pollution_idx]
true_inv = scaler.inverse_transform(np.hstack([actuals, np.zeros((len(actuals), 6))]))[:, pollution_idx]

plt.figure(figsize=(12,6))
plt.plot(true_inv, label='True PM2.5')
plt.plot(pred_inv, label='Predicted PM2.5')
plt.legend()
plt.title('PM2.5 Prediction (External Test Set)')
plt.xlabel('Samples')
plt.ylabel('PM2.5')
plt.show()
