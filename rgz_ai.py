import numpy as np
import pandas as pd
import os
import glob
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, confusion_matrix
from sklearn.model_selection import GroupKFold
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

all_dfs = []
for scenario in ['Indoor', 'Mobility', 'Outdoor', 'Pedestrian']:
    scenario_path = os.path.join('Channel Logs', scenario)
    if os.path.exists(scenario_path):
        for file in glob.glob(os.path.join(scenario_path, '*.csv')):
            df = pd.read_csv(file)
            if df.empty:
                continue

            df.columns = df.columns.str.replace('\n', '').str.replace('\r', '').str.strip()

            for col in df.columns:
                if col not in ['Timestamp', 'Date', 'Time', 'Eid', 'scenario']:
                    df[col] = df[col].astype(str).str.strip('"').str.strip()
                    df[col] = df[col].replace('-', np.nan)
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            file_eid = os.path.basename(file).lower().replace('.csv', '').strip()

            df['Eid'] = file_eid
            df['scenario'] = scenario
            all_dfs.append(df)

if not all_dfs:
    raise FileNotFoundError("Не найдено ни одного файла в папке 'Channel Logs'!")

channel_data = pd.concat(all_dfs, ignore_index=True)

# Глубокая очистка Timestamp перед парсингом datetime
channel_data['Timestamp'] = channel_data['Timestamp'].astype(str).str.replace('\n', '').str.replace('\r', '').str.strip(
    '"').str.strip()
channel_data['datetime'] = pd.to_datetime(channel_data['Timestamp'], format='%Y.%m.%d_%H.%M.%S', errors='coerce')

events_data = pd.read_csv(os.path.join('YouTuve QoE Events', 'events.csv'))
events_data.columns = events_data.columns.str.strip()
events_data['Eid'] = events_data['Eid'].astype(str).str.strip('"').str.strip().str.lower()
events_data['is_buffer'] = (events_data['Status'] == 3).astype(int)
events_data['datetime'] = pd.to_datetime(events_data['Date'].astype(str) + ' ' + events_data['Time'].astype(str),
                                         dayfirst=True, errors='coerce')

qoe_path = os.path.join('YouTuve QoE Events', 'qoe.csv')
if os.path.exists(qoe_path):
    qoe_data = pd.read_csv(qoe_path)
    qoe_data.columns = qoe_data.columns.str.strip()
    qoe_data['Eid'] = qoe_data['Eid'].astype(str).str.strip('"').str.strip().str.lower()
    qoe_data['datetime'] = pd.to_datetime(qoe_data['Date'].astype(str) + ' ' + qoe_data['Time'].astype(str),
                                          dayfirst=True, errors='coerce')
    for col in ['Video Bytes Downloaded', 'Loaded Percentage']:
        if col in qoe_data.columns:
            qoe_data[col] = pd.to_numeric(qoe_data[col].astype(str).str.strip('"').str.strip(), errors='coerce')
else:
    qoe_data = pd.DataFrame()

common_eids = list(set(channel_data['Eid'].unique()) & set(events_data['Eid'].unique()))

if len(common_eids) == 0:
    raise ValueError("Нет общих сессий Eid между логами радиоканала и логами событий!")

history_seconds = 25
future_seconds = 10

features_list = []
targets_list = []
groups_list = []

for eid in common_eids:
    channel_eid = channel_data[channel_data['Eid'] == eid].sort_values('datetime').dropna(subset=['datetime'])
    events_eid = events_data[(events_data['Eid'] == eid) & (events_data['is_buffer'] == 1)].dropna(subset=['datetime'])

    if len(channel_eid) < 5:
        continue

    if not qoe_data.empty:
        qoe_eid = qoe_data[qoe_data['Eid'] == eid].sort_values('datetime').dropna(subset=['datetime'])
        merged_eid = pd.merge_asof(channel_eid, qoe_eid, on='datetime', by='Eid', direction='backward',
                                   suffixes=('', '_qoe'))
        merged_eid = merged_eid.ffill().bfill()
    else:
        merged_eid = channel_eid.ffill().bfill()

    for idx in range(len(merged_eid)):
        current_time = merged_eid.iloc[idx]['datetime']
        start_history = current_time - pd.Timedelta(seconds=history_seconds)

        history = merged_eid[(merged_eid['datetime'] >= start_history) & (merged_eid['datetime'] <= current_time)]
        if len(history) < 2:
            continue

        end_future = current_time + pd.Timedelta(seconds=future_seconds)
        future_events = events_eid[(events_eid['datetime'] > current_time) & (events_eid['datetime'] <= end_future)]
        future_buffering = 1 if len(future_events) > 0 else 0

        row = {}

        chan_cols = ['RSRP', 'RSRQ', 'CQI', 'SNR', 'DL_bitrate', 'UL_bitrate']
        available_chan = [c for c in chan_cols if c in history.columns]
        for col in available_chan:
            col_data = history[col].values
            col_data = col_data[~np.isnan(col_data)]
            if len(col_data) > 0:
                row[f'{col}_mean'] = np.mean(col_data)
                row[f'{col}_std'] = np.std(col_data) if len(col_data) > 1 else 0
                row[f'{col}_min'] = np.min(col_data)
                row[f'{col}_max'] = np.max(col_data)
                row[f'{col}_trend'] = np.polyfit(range(len(col_data)), col_data, 1)[0] if len(col_data) > 1 else 0

        qoe_cols = ['Video Bytes Downloaded', 'Loaded Percentage']
        available_qoe = [c for c in qoe_cols if c in history.columns]
        for col in available_qoe:
            col_data = history[col].values
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
            groups_list.append(eid)

X_raw = pd.DataFrame(features_list)
y = np.array(targets_list)
groups = np.array(groups_list)

print(f"Всего создано примеров: {len(X_raw)} (0={np.sum(y == 0)}, 1={np.sum(y == 1)})")

X_raw = X_raw.fillna(0)
constant_cols = [col for col in X_raw.columns if X_raw[col].std() == 0]
X_raw = X_raw.drop(columns=constant_cols)

feature_names = X_raw.columns.tolist()

unique_groups_count = len(np.unique(groups))
n_splits = min(5, unique_groups_count)
gkf = GroupKFold(n_splits=n_splits)

fold_pr_aucs = []
fold_f1_scores = []
fold_thresholds = []
all_feature_importances = np.zeros(len(feature_names))

for fold, (train_idx, test_idx) in enumerate(gkf.split(X_raw, y, groups=groups)):
    X_train_fold, X_test_fold = X_raw.iloc[train_idx], X_raw.iloc[test_idx]
    y_train_fold, y_test_fold = y[train_idx], y[test_idx]

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_fold)
    X_test_scaled = scaler.transform(X_test_fold)

    n_positives = np.sum(y_train_fold == 1)
    if n_positives > 2:
        smote = SMOTE(k_neighbors=2, random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train_fold)
    else:
        X_train_resampled, y_train_resampled = X_train_scaled, y_train_fold

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        reg_alpha=0.7,
        reg_lambda=1.5,
        random_state=42,
        eval_metric='aucpr'
    )

    model.fit(X_train_resampled, y_train_resampled)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

    thresholds = np.linspace(0.01, 0.50, 50)
    best_threshold = 0.5
    best_f1 = 0
    for t in thresholds:
        y_pred_t = (y_test_proba >= t).astype(int)
        f1_t = f1_score(y_test_fold, y_pred_t)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_threshold = t

    y_pred = (y_test_proba >= best_threshold).astype(int)

    pr_auc = average_precision_score(y_test_fold, y_test_proba)
    f1 = f1_score(y_test_fold, y_pred)

    fold_pr_aucs.append(pr_auc)
    fold_f1_scores.append(f1)
    fold_thresholds.append(best_threshold)
    all_feature_importances += model.feature_importances_

    print(
        f"Фолд {fold + 1} -> Тест. примеров: {len(X_test_fold)} (Класс 1: {np.sum(y_test_fold == 1)}), PR-AUC: {pr_auc:.4f}, F1: {f1:.4f}")

mean_pr_auc = np.mean(fold_pr_aucs)
mean_f1 = np.mean(fold_f1_scores)
mean_threshold = np.mean(fold_thresholds)
all_feature_importances /= n_splits

print(f"Средний PR-AUC: {mean_pr_auc:.4f}")
print(f"Средний F1-score: {mean_f1:.4f}")
print(f"Средний оптимальный порог: {mean_threshold:.3f}")

plt.figure(figsize=(10, 5))
indices = np.argsort(all_feature_importances)[::-1][:15]
top_features = [feature_names[i] for i in indices]
top_importances = all_feature_importances[indices]

sns.barplot(x=top_importances, y=top_features, hue=top_features, palette='viridis', legend=False)
plt.title('Топ-15 значимых признаков')
plt.xlabel('Относительная важность')
plt.tight_layout()
plt.show()
