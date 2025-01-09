import torch.nn as nn
import torch.nn.functional as F

class LSTMPredictor(nn.Module):
    def __init__(self, log_input_size, packet_input_size, hidden_size, output_size):
        super(LSTMPredictor, self).__init__()
        # 로그 데이터 LSTM
        self.log_lstm = nn.LSTM(log_input_size, hidden_size, batch_first=True)
        self.log_fc = nn.Linear(hidden_size, 32)

        # 패킷 데이터 LSTM
        self.packet_lstm = nn.LSTM(packet_input_size, hidden_size, batch_first=True)
        self.packet_fc = nn.Linear(hidden_size, 32)

        # 병합 및 출력
        self.fc = nn.Linear(64, output_size)

    def forward(self, logs, packets):
        # 로그 데이터 처리
        log_out, _ = self.log_lstm(logs)
        log_out = log_out[:, -1, :]
        log_features = F.relu(self.log_fc(log_out))

        # 패킷 데이터 처리
        packet_out, _ = self.packet_lstm(packets)
        packet_out = packet_out[:, -1, :]
        packet_features = F.relu(self.packet_fc(packet_out))

        # 병합 및 출력
        combined = torch.cat((log_features, packet_features), dim=1)
        output = self.fc(combined)
        return output
