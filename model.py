import torch
import torch.nn as nn # nn stands for neural network
import torch.nn.functional as F

class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(LSTMPredictor, self).__init__()

        # 로그 데이터 LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.device = device

    def forward(self, data): # forward: 출력값 계산
        # 로그 데이터 처리

        data = data.to(torch.float32)

        data_out, _ = self.lstm(data)
        data_out = data_out[:, -1, :]

        self.bn = nn.BatchNorm1d(self.hidden_size, device=self.device)
        data_out = self.bn(data_out)

        output = F.relu(self.fc(data_out)) # activation function: ReLU
        # output = self.fc(F.relu(data_out))  # Combine ReLU and fc

        # 병합 및 출력
        # combined = torch.cat((log_features, packet_features), dim=1)
        # output = self.fc(features)

        return output
