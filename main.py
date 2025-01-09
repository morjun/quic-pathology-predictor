import torch # pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
from torch.utils.data import DataLoader
from dataset import PathologyDataset
from model import LSTMPredictor
from preprocess import load_and_preprocess_data

# 데이터 로드 및 전처리
X_train_logs, X_test_logs, X_train_packets, X_test_packets, y_train, y_test = load_and_preprocess_data(
    'log_data.csv', 'packet_data.csv'
)

# Dataset 및 DataLoader 생성
train_dataset = PathologyDataset(X_train_logs, X_train_packets, y_train)
test_dataset = PathologyDataset(X_test_logs, X_test_packets, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 모델 초기화
model = LSTMPredictor(log_input_size=100, packet_input_size=3, hidden_size=64, output_size=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# 학습 루프
def train_model(model, train_loader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for logs, packets, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(logs, packets)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}")

# 평가 루프
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for logs, packets, labels in test_loader:
            outputs = model(logs, packets)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.2f}")

# 학습 및 평가 실행
train_model(model, train_loader, optimizer, criterion, epochs=10)
evaluate_model(model, test_loader)
