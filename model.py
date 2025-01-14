import torch
import torch.nn as nn # nn stands for neural network
import torch.nn.functional as F

class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMPredictor, self).__init__()

        # 로그 데이터 LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 32)

        # 병합 및 출력
        self.fc = nn.Linear(64, output_size)

    def forward(self, data): # forward: 출력값 계산
        # 로그 데이터 처리
        data_out, _ = self.log_lstm(data)
        data_out = data_out[:, -1, :]
        features = F.relu(self.log_fc(data_out)) # activation function: ReLU

        # 병합 및 출력
        # combined = torch.cat((log_features, packet_features), dim=1)

        output = self.fc(features)
        return output
