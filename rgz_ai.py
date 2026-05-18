import numpy as np
import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import average_precision_score, f1_score
import xgboost as xgb

all_dfs = []
for scenario in ['Indoor', 'Mobility', 'Outdoor', 'Pedestrian']:
    scenario_path = os.path.join('Channel Logs', scenario)
    if os.path.exists(scenario_path):
        for file in glob.glob(os.path.join(scenario_path, '*.csv')):
            df = pd.read_csv(file)

            for col in df.columns:
                if col not in ['Timestamp', 'Date', 'Time']:
                    df[col] = df[col].astype(str).str.strip('"').str.strip()
                    df[col] = df[col].replace('-', np.nan)
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            file_eid = os.path.basename(file).replace('.csv', '').strip()
            if 'Eid' in df.columns and len(df) > 0:
                first_val = str(df['Eid'].iloc[0]).strip('"').strip()
                eid = first_val if first_val and first_val != 'nan' else file_eid
            else:
                eid = file_eid

            df['Eid'] = eid
            df['scenario'] = scenario
            all_dfs.append(df)

if not all_dfs:
    raise FileNotFoundError("Не найдено ни одного файла в папке 'Channel Logs'!")

channel_data = pd.concat(all_dfs, ignore_index=True)

channel_data['Timestamp'] = channel_data['Timestamp'].astype(str).str.replace('\n', '').str.replace('\r', '').str.strip(
    '"').str.strip()
channel_data['datetime'] = pd.to_datetime(channel_data['Timestamp'], format='%Y.%m.%d_%H.%M.%S', errors='coerce')

events_data = pd.read_csv(os.path.join('YouTuve QoE Events', 'events.csv'))
events_data['Eid'] = events_data['Eid'].astype(str).str.strip('"').str.strip()
events_data['is_buffer'] = (events_data['Status'] == 3).astype(int)
events_data['datetime'] = pd.to_datetime(events_data['Date'].astype(str) + ' ' + events_data['Time'].astype(str),
                                         dayfirst=True, errors='coerce')

qoe_path = os.path.join('YouTuve QoE Events', 'qoe.csv')
if os.path.exists(qoe_path):
    qoe_data = pd.read_csv(qoe_path)
    qoe_data['Eid'] = qoe_data['Eid'].astype(str).str.strip('"').str.strip()
    qoe_data['datetime'] = pd.to_datetime(qoe_data['Date'].astype(str) + ' ' + qoe_data['Time'].astype(str),
                                          dayfirst=True, errors='coerce')
    for col in ['Video Bytes Downloaded', 'Loaded Percentage']:
        if col in qoe_data.columns:
            qoe_data[col] = pd.to_numeric(qoe_data[col].astype(str).str.strip('"').str.strip(), errors='coerce')
else:
    qoe_data = pd.DataFrame()

common_eids = set(channel_data['Eid'].unique()) & set(events_data['Eid'].unique())

if len(common_eids) == 0:
    raise ValueError("Нет общих сессий Eid между логами радиоканала и логами событий!")

history_seconds = 25
future_seconds = 10

features_list = []
targets_list = []

for eid in common_eids:
    channel_eid = channel_data[channel_data['Eid'] == eid].sort_values('datetime').dropna(subset=['datetime'])
    channel_eid = channel_eid.ffill().bfill()

    events_eid = events_data[(events_data['Eid'] == eid) & (events_data['is_buffer'] == 1)].dropna(subset=['datetime'])

    if not qoe_data.empty:
        qoe_eid = qoe_data[qoe_data['Eid'] == eid].sort_values('datetime').dropna(subset=['datetime'])
        qoe_eid = qoe_eid.ffill().bfill()
    else:
        qoe_eid = pd.DataFrame()

    if len(channel_eid) < 5:
        continue

    for idx in range(len(channel_eid)):
        current_time = channel_eid.iloc[idx]['datetime']
        start_history = current_time - pd.Timedelta(seconds=history_seconds)

        history_chan = channel_eid[
            (channel_eid['datetime'] >= start_history) & (channel_eid['datetime'] <= current_time)]
        if len(history_chan) < 2:
            continue

        if not qoe_eid.empty:
            history_qoe = qoe_eid[(qoe_eid['datetime'] >= start_history) & (qoe_eid['datetime'] <= current_time)]
        else:
            history_qoe = pd.DataFrame()

        end_future = current_time + pd.Timedelta(seconds=future_seconds)
        future_events = events_eid[(events_eid['datetime'] > current_time) & (events_eid['datetime'] <= end_future)]
        future_buffering = 1 if len(future_events) > 0 else 0

        row = {}

        chan_cols = ['RSRP', 'RSRQ', 'CQI', 'SNR', 'DL_bitrate', 'UL_bitrate']
        available_chan = [c for c in chan_cols if c in history_chan.columns]
        for col in available_chan:
            col_data = history_chan[col].values
            col_data = col_data[~np.isnan(col_data)]
            if len(col_data) > 0:
                row[f'{col}_mean'] = np.mean(col_data)
                row[f'{col}_std'] = np.std(col_data) if len(col_data) > 1 else 0
                row[f'{col}_min'] = np.min(col_data)
                row[f'{col}_max'] = np.max(col_data)
                row[f'{col}_trend'] = np.polyfit(range(len(col_data)), col_data, 1)[0] if len(col_data) > 1 else 0

        if not history_qoe.empty:
            qoe_cols = ['Video Bytes Downloaded', 'Loaded Percentage']
            available_qoe = [c for c in qoe_cols if c in history_qoe.columns]
            for col in available_qoe:
                col_data = history_qoe[col].values
                col_data = col_data[~np.isnan(col_data)]
                if len(col_data) > 0:
                    row[f'{col}_mean'] = np.mean(col_data)
                    row[f'{col}_std'] = np.std(col_data) if len(col_data) > 1 else 0
                    row[f'{col}_min'] = np.min(col_data)
                    row[f'{col}_max'] = np.max(col_data)
                    row[f'{col}_trend'] = np.polyfit(range(len(col_data)), col_data, 1)[0] if len(col_data) > 1 else 0

        if row:
            features_list.append(row)
            targets_list.append(future_buffering)

X_raw = pd.DataFrame(features_list)
y = np.array(targets_list)

print(f"Создано {len(X_raw)} примеров (0={np.sum(y == 0)}, 1={np.sum(y == 1)})")
if X_raw.empty:
    raise ValueError("DataFrame признаков пуст")

X_raw = X_raw.fillna(0)

constant_cols = [col for col in X_raw.columns if X_raw[col].std() == 0]
X_raw = X_raw.drop(columns=constant_cols)

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_raw)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

ratio = np.sum(y_train == 0) / max(1, np.sum(y_train == 1))

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=ratio,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

y_test_proba = model.predict_proba(X_test)[:, 1]

thresholds = np.linspace(0.1, 0.9, 50)
best_threshold = 0.5
best_f1 = 0

for t in thresholds:
    y_pred_t = (y_test_proba >= t).astype(int)
    f1_t = f1_score(y_test, y_pred_t)
    if f1_t > best_f1:
        best_f1 = f1_t
        best_threshold = t

y_pred = (y_test_proba >= best_threshold).astype(int)

pr_auc = average_precision_score(y_test, y_test_proba)
f1 = f1_score(y_test, y_pred)

print(f"PR-AUC: {pr_auc:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Оптимальный порог: {best_threshold:.3f}")
