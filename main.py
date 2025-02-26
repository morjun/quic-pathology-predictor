import os
import argparse
import re
import pandas as pd
from pathlib import Path
import torch # pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
from torch.utils.data import DataLoader
from dataset import PathologyDataset
from model import LSTMPredictor
from preprocess import prepare_dataset
from datetime import datetime
import ast

# 학습 루프
def train_model(model, train_loader, optimizer, criterion, device, epochs=50):
    model.train() # 학습 모드로 전환
    for epoch in range(epochs):
        total_loss = 0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            data = data.to(torch.float32)
            labels = labels.to(torch.long)

            outputs = model(data)  # Callable한 객체: 내부적으로 forward() 호출
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}")

# 평가 루프
def evaluate_model(model, test_loader, device, stats_frame, test_indices):
    model.eval() # evaluation 모드로 전환
    correct = 0
    total = 0
    eval_df = pd.DataFrame()

    predictions = []  # List to store predictions
    true_labels = []  # List to store true labels

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)

            comparison = torch.stack([predicted, labels], dim=1).cpu().numpy()
            # Append predicted and actual labels

            predictions.extend(predicted.cpu().numpy())  # Collect predictions
            true_labels.extend(labels.cpu().numpy())     # Collect true labels

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Add predictions and true labels to the eval dataframe
    eval_df['predicted'] = predictions

    eval_df['label'] = true_labels

    eval_df['index'] = test_indices

    final_df = eval_df.merge(stats_frame, left_on='index', right_on='index', how = 'left')
    print(final_df)
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.2f}")

def count_files_by_extension_with_pathlib(folder_path, extension):
    """
    폴더를 순회하며 특정 확장자의 파일 개수를 센다 (pathlib 사용).

    Parameters:
        folder_path (str): 순회할 폴더 경로.
        extension (str): 찾고자 하는 파일 확장자 (예: ".txt").

    Returns:
        int: 확장자에 해당하는 파일 개수.
    """
    folder = Path(folder_path)
    return sum(1 for file in folder.rglob(f"*{extension}"))

def get_labels_from_stats(stats_path, time_datetime):
    """
    stats.csv 파일에서 레이블 정보를 추출한다.

    Parameters:
        stats_path (str): stats.csv 파일 경로.

    Returns:
        None
    """
    stats_frame = pd.read_csv(stats_path)
    stats_frame.columns = stats_frame.columns.str.replace('"', '')

    stats_frame['timestamp'] = pd.to_datetime(stats_frame['time'], format='%Y-%m-%d %H:%M:%S')
    filtered_frame = stats_frame[stats_frame['timestamp'] == time_datetime]
    # print(filtered_frame)

    # stats.csv 파일에서 레이블 정보 추출
    labels = filtered_frame['pathology'].values

    return labels, filtered_frame

def main():
    # 데이터 로드 및 전처리 (training, test 데이터셋 생성)
    parser = argparse.ArgumentParser(description="Show spin bit")
    parser.add_argument("--stats", "-s", help="stats.csv path", type=str, required=True)
    parser.add_argument("--epochs", "-e", help="Number of epochs", type=int, default=200)
    parser.add_argument("paths", metavar="paths", type=str, nargs="+")
    args = parser.parse_args()

    sessions = []
    stats_frame = None

    for full_path in args.paths:
        pcap_count_per_path = count_files_by_extension_with_pathlib(full_path, ".pcap")

        full_path = os.path.relpath(full_path)

        splitted_path = os.path.split(full_path) # ('...', 'l0b0d0_t0')

        arg_path_parts = splitted_path[1].split("_") # ['l0b0d0', 't0']
        parametric_path = arg_path_parts[0] # l0b0d0

        time = 0
        if len(arg_path_parts) > 1:
            time = arg_path_parts[1] # t0
            time = time[1:] # 0
        time_datetime = datetime.fromtimestamp(int(time))

        labels, stats_frame = get_labels_from_stats(args.stats, time_datetime)

        for i in range(pcap_count_per_path):
            if i > 0:
                spin_filename = f"{parametric_path}_{i+1}_spin.csv"
                cwnd_filename = f"{parametric_path}_{i+1}_cwnd.csv"
                lost_filename = f"{parametric_path}_{i+1}_lost.csv"
                throughput_filename = f"{parametric_path}_{i+1}.csv"
            else:
                spin_filename = f"{parametric_path}_spin.csv"
                cwnd_filename = f"{parametric_path}_cwnd.csv"
                lost_filename = f"{parametric_path}_lost.csv"
                throughput_filename = f"{parametric_path}.csv"
            
            spinFrame = pd.read_csv(os.path.join(full_path, spin_filename))
            cwndFrame = pd.read_csv(os.path.join(full_path, cwnd_filename))
            lostFrame = pd.read_csv(os.path.join(full_path, lost_filename))
            throughputFrame = pd.read_csv(os.path.join(full_path, throughput_filename))

            sessions.append({'throughputFrame': throughputFrame, 'spinFrame': spinFrame, 'lostFrame': lostFrame, 'cwndFrame': cwndFrame, 'label': ast.literal_eval(labels[i])})

    X_train, X_test, y_train, y_test, train_indices, test_indices = prepare_dataset(sessions, timesteps=50)

    # for session in sessions:
    #     print(session['label'])
    #     print(session['throughputFrame'].head())
    
    print(f"X_train shape: {X_train.shape}")
    # print(f"X_train : {X_train}")
    print(f"X_test shape: {X_test.shape}")
    # print(f"X_test : {X_test}")
    print(f"y_train shape: {y_train.shape}")
    # print(f"y_train: {y_train}")
    print(f"y_test shape: {y_test.shape}")
    # print(f"y_test: {y_test}")

    overlap = set(map(tuple, X_train)) & set(map(tuple, X_test))
    print(f"Number of overlapping samples: {len(overlap)}")

    # Dataset 및 DataLoader 생성
    train_dataset = PathologyDataset(X_train, y_train)
    test_dataset = PathologyDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 모델 초기화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMPredictor(input_size=7, hidden_size=64, output_size=2, device=device).to(device)
    #input: throughput, spinfrequency, rack, fack, probe, cwnd 
    #output: label with Softmax

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # 학습 및 평가 실행
    train_model(model, train_loader, optimizer, criterion, device, epochs=args.epochs)
    evaluate_model(model, test_loader, device, stats_frame, test_indices)

if __name__ == "__main__":
    main()