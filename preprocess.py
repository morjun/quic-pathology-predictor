import pandas as pd
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

QUIC_TRACE_PACKET_LOSS_RACK = 0
QUIC_TRACE_PACKET_LOSS_FACK = 1
QUIC_TRACE_PACKET_LOSS_PROBE = 2

def preprocess_to_fixed_timesteps(throughputFrame, spinFrame, lostFrame, cwndFrame, timesteps):
    """
    msquic 로그와 패킷 데이터를 고정된 타임스텝 시계열 데이터로 변환.
    
    Parameters:
        spinFrame (DataFrame): spin bit 데이터 (time, spin).
        lostFrame (DataFrame): 패킷 손실 데이터 (time, loss).
        cwndFrame (DataFrame): CWnd 데이터 (time, cwnd).
        wMaxFrame (DataFrame): WindowMax 데이터 (time, wMax).
        timesteps (int): 고정 타임스텝 개수.
    
    Returns:
        np.ndarray: 고정된 타임스텝으로 변환된 시계열 데이터.
    """
    
    cwndFrame['time'] = cwndFrame['time'].astype(np.float32)
    spinFrame['time'] = spinFrame['time'].astype(np.float32)
    lostFrame['time'] = lostFrame['time'].astype(np.float32)
    throughputFrame['Interval start'] = throughputFrame['Interval start'].astype(np.float32)
    # print(throughputFrame.head())

    rack_count = lostFrame[lostFrame['loss'] == QUIC_TRACE_PACKET_LOSS_RACK].shape[0]
    fack_count = lostFrame[lostFrame['loss'] == QUIC_TRACE_PACKET_LOSS_FACK].shape[0]
    probe_count = lostFrame[lostFrame['loss'] == QUIC_TRACE_PACKET_LOSS_PROBE].shape[0]

    # print(f"Processing data with loss count {rack_count}, {fack_count}, {probe_count}")

    # 시간 범위 결정
    total_duration = max(
        cwndFrame['time'].max(),
        spinFrame['time'].max(),
        lostFrame['time'].max(),
        throughputFrame['Interval start'].max()
    )

    step_duration = total_duration / timesteps
    
    # 타임스텝별 데이터 프레임 생성
    time_bins = np.linspace(0, total_duration, timesteps + 1)
    # print(time_bins)
    fixed_timesteps = pd.DataFrame({'time': time_bins[:-1]})

    # Throughput 데이터 합치기
    throughputFrame['Interval Bin'] = pd.cut(throughputFrame['Interval start'], bins=time_bins, include_lowest=True)
    throughput_agg = throughputFrame.groupby('Interval Bin', observed = False).sum().reset_index()
    fixed_timesteps['throughput'] = (throughput_agg['All Packets'] / step_duration).astype(np.float32) #bps

    # 타임스텝별 Spin Frequency 계산
    spinFrame["spin_change"] = spinFrame["spin"].diff().fillna(0).abs()
    spinFrame["spin_change"] = spinFrame["spin_change"].astype(np.float32) # 총 spin 횟수

    spins_per_group = spinFrame.groupby(pd.cut(spinFrame["time"], bins=time_bins), observed= False)["spin_change"].sum()

    spin_freq_agg = spins_per_group / step_duration  # Spin frequency 계산
    fixed_timesteps["spin_frequency"] = (spin_freq_agg.values).astype(np.float32)

    # Loss 데이터: 0, 1, 2 개수 계산
    loss_counts = pd.crosstab(
        pd.cut(lostFrame["time"], bins=time_bins, include_lowest=True),
        lostFrame["loss"],
        dropna=False,
    ).astype(np.int32) # 길이 49

    # print(f"loss counts: {loss_counts}")

    # 빈 타임스텝을 0으로 채움
    loss_types = {
        QUIC_TRACE_PACKET_LOSS_RACK: "loss_rack_count",
        QUIC_TRACE_PACKET_LOSS_FACK: "loss_fack_count",
        QUIC_TRACE_PACKET_LOSS_PROBE: "loss_probe_count"
    }
    for loss_type, column_name in loss_types.items():
        fixed_timesteps[column_name] = loss_counts.get(loss_type, pd.Series(0, index=loss_counts.index)).values

    # CWnd 데이터 합치기
    cwndFrame['time Bin'] = pd.cut(cwndFrame['time'], bins=time_bins)
    cwnd_agg = cwndFrame.groupby('time Bin', observed=False).mean().reset_index()
    cwnd_agg['cwnd'] = cwnd_agg['cwnd'].astype(np.float32).ffill()
    fixed_timesteps['cwnd'] = cwnd_agg['cwnd'].astype(np.float32)

    # 결측값 처리 (NaN -> 0)
    fixed_timesteps.fillna(0, inplace=True)

    # print(fixed_timesteps.dtypes)
    # print(fixed_timesteps.head())

    return fixed_timesteps.to_numpy()

def prepare_dataset(sessions, timesteps=50):
    """
    여러 통신 세션 데이터를 LSTM 학습용 데이터로 변환.
    
    Parameters:
        sessions (list of dict): 각 세션의 데이터 딕셔너리 목록. 
            각 딕셔너리는 'throughputFrame' ,'spinFrame', 'lostFrame', 'cwndFrame', 'label' 키를 포함.
        timesteps (int): 고정된 타임스텝 개수.

    Returns:
        X (np.ndarray): LSTM 입력 데이터 (num_sessions, timesteps, features).
        y (np.ndarray): LSTM 출력 레이블 (num_sessions, ).
    """
    X = []
    y = []
    indices = []

    for i, session in enumerate(sessions):
        # print(session)
        # 각 세션의 데이터를 고정된 타임스텝으로 변환
        fixed_data = preprocess_to_fixed_timesteps(
            session['throughputFrame'],
            session['spinFrame'],
            session['lostFrame'],
            session['cwndFrame'],
            timesteps=timesteps
        )

        # 데이터가 지정된 타임스텝보다 짧은 경우 패딩
        if fixed_data.shape[0] < timesteps:
            padding = np.zeros((timesteps - fixed_data.shape[0], fixed_data.shape[1]))
            fixed_data = np.vstack([fixed_data, padding])

        # 데이터가 긴 경우 자르기
        elif fixed_data.shape[0] > timesteps:
            fixed_data = fixed_data[:timesteps, :]

        X.append(fixed_data)
        y.append(session['label'])
        indices.append(i)
        # print(fixed_data)
    
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y, indices, test_size=0.2, random_state=42
    )

    X_train = np.stack(X_train)
    X_test = np.stack(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    # PyTorch 텐서 변환
    return (
        torch.tensor(X_train),
        torch.tensor(X_test),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(y_test, dtype=torch.long),
        train_indices,
        test_indices
    )