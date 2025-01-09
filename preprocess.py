import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(log_file, packet_file):
    # 데이터 로드
    log_data = pd.read_csv(log_file)
    packet_data = pd.read_csv(packet_file)

    # 로그 데이터 임베딩
    vectorizer = TfidfVectorizer(max_features=100)
    log_embeddings = vectorizer.fit_transform(log_data['log_message']).toarray()

    # 패킷 데이터 정규화 및 시계열 정리
    packet_features = ['rtt', 'packet_size', 'flags']
    packet_sequences = packet_data[packet_features].values.reshape(-1, 50, len(packet_features))

    # 라벨 추출
    labels = log_data['pathology'].values

    # 학습/검증 데이터 분리
    X_train_logs, X_test_logs, X_train_packets, X_test_packets, y_train, y_test = train_test_split(
        log_embeddings, packet_sequences, labels, test_size=0.2, random_state=42
    )

    # PyTorch 텐서 변환
    return (
        torch.tensor(X_train_logs, dtype=torch.float32),
        torch.tensor(X_test_logs, dtype=torch.float32),
        torch.tensor(X_train_packets, dtype=torch.float32),
        torch.tensor(X_test_packets, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(y_test, dtype=torch.long),
    )
