import numpy as np
import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns


def detect_outliers_iqr(data, multiplier=1.5):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return (data < lower_bound) | (data > upper_bound)


def detect_outliers_zscore(data, threshold=3):
    z_scores = np.abs((data - np.mean(data)) / np.std(data))
    return z_scores > threshold


def cap_outliers(data, lower_percentile=1, upper_percentile=99):
    lower = np.percentile(data, lower_percentile)
    upper = np.percentile(data, upper_percentile)
    return np.clip(data, lower, upper)


all_dfs = []
for scenario in ['Indoor', 'Mobility', 'Outdoor', 'Pedestrian']:
    scenario_path = os.path.join('Channel Logs', scenario)
    if os.path.exists(scenario_path):
        for file in glob.glob(os.path.join(scenario_path, '*.csv')):
            df = pd.read_csv(file)
            for col in df.columns:
                df[col] = df[col].astype(str).str.strip('"').str.strip()
                df[col] = df[col].replace('-', np.nan)
                df[col] = pd.to_numeric(df[col], errors='coerce')
            eid = os.path.basename(file).replace('.csv', '')
            df['Eid'] = eid
            all_dfs.append(df)

channel_data = pd.concat(all_dfs, ignore_index=True)
events_data = pd.read_csv(os.path.join('YouTuve QoE Events', 'events.csv'))

print(f"Channel logs: {len(channel_data)} записей, {channel_data['Eid'].nunique()} экспериментов")
print(f"Events: {len(events_data)} записей, {events_data['Eid'].nunique()} экспериментов")

events_data['is_buffer'] = (events_data['Status'] == 3).astype(int)

common_eids = set(channel_data['Eid'].unique()) & set(events_data['Eid'].unique())
print(f"Общих Eid: {len(common_eids)}")

history_window = 10
future_window = 10

features_list = []
targets_list = []

outlier_stats = {}

for eid in common_eids:
    channel_eid = channel_data[channel_data['Eid'] == eid].sort_values('Timestamp')
    channel_eid = channel_eid.ffill().bfill()

    for col in ['RSRP', 'RSRQ', 'CQI', 'SNR', 'DL_bitrate', 'UL_bitrate']:
        if col in channel_eid.columns:
            col_data = channel_eid[col].values
            col_data = col_data[~np.isnan(col_data)]

            outliers_iqr = detect_outliers_iqr(col_data)
            outliers_zscore = detect_outliers_zscore(col_data)
            outliers_combined = outliers_iqr | outliers_zscore

            outlier_stats[f'{eid}_{col}'] = {
                'total': len(col_data),
                'outliers_iqr': np.sum(outliers_iqr),
                'outliers_zscore': np.sum(outliers_zscore),
                'outliers_combined': np.sum(outliers_combined),
                'outlier_percent': np.sum(outliers_combined) / len(col_data) * 100
            }

            col_data_capped = cap_outliers(col_data)
            channel_eid[col] = col_data_capped

    events_eid = events_data[events_data['Eid'] == eid].sort_values('TimeStall')
    events_eid['time'] = pd.to_numeric(events_eid['TimeStall'], errors='coerce')

    if len(channel_eid) < history_window + future_window:
        continue

    for t in range(history_window, len(channel_eid) - future_window):
        history = channel_eid.iloc[t - history_window:t]

        current_time = t
        future_buffering = 0

        for _, event in events_eid.iterrows():
            if history_window <= event['time'] <= t + future_window:
                if event['is_buffer'] == 1:
                    future_buffering = 1
                    break

        row = {}

        for col in ['RSRP', 'RSRQ', 'CQI', 'SNR', 'DL_bitrate', 'UL_bitrate']:
            if col in history.columns:
                col_data = history[col].values
                col_data = col_data[~np.isnan(col_data)]
                if len(col_data) > 0:
                    row[f'{col}_mean'] = np.mean(col_data)
                    row[f'{col}_std'] = np.std(col_data)
                    row[f'{col}_min'] = np.min(col_data)
                    row[f'{col}_max'] = np.max(col_data)
                    row[f'{col}_trend'] = np.polyfit(range(len(col_data)), col_data, 1)[0] if len(col_data) > 1 else 0
                    row[f'{col}_outlier_ratio'] = np.sum(detect_outliers_iqr(col_data)) / len(col_data)

        if row:
            features_list.append(row)
            targets_list.append(future_buffering)

print(f"\nСтатистика выбросов:")
outlier_df = pd.DataFrame(outlier_stats).T
print(f"Средний процент выбросов: {outlier_df['outlier_percent'].mean():.2f}%")
print(f"Макс процент выбросов: {outlier_df['outlier_percent'].max():.2f}%")

X_raw = pd.DataFrame(features_list)
y = np.array(targets_list)

print(f"\nСоздано {len(X_raw)} посекундных примеров")
print(f"Распределение: 0={np.sum(y == 0)}, 1={np.sum(y == 1)}")
print(f"Доля буферизации: {np.mean(y) * 100:.2f}%")

for col in X_raw.columns:
    if 'outlier_ratio' in col:
        outliers_col = X_raw[col].values
        outliers_mask = outliers_col > 0.3
        if np.sum(outliers_mask) > 0:
            print(f"Удалено {np.sum(outliers_mask)} записей с {col} > 0.3")
            X_raw = X_raw[~outliers_mask]
            y = y[~outliers_mask]

X_raw = X_raw.fillna(0)

constant_cols = [col for col in X_raw.columns if X_raw[col].std() == 0]
X_raw = X_raw.drop(columns=constant_cols)
print(f"Признаков после обработки: {X_raw.shape[1]}")

for col in X_raw.columns:
    if 'std' in col or 'trend' in col:
        X_raw[col] = cap_outliers(X_raw[col].values, lower_percentile=5, upper_percentile=95)

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_raw)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTrain: {X_train.shape}, классы: 0={np.sum(y_train == 0)}, 1={np.sum(y_train == 1)}")
print(f"Test: {X_test.shape}, классы: 0={np.sum(y_test == 0)}, 1={np.sum(y_test == 1)}")

ratio = np.sum(y_train == 0) / max(1, np.sum(y_train == 1))

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=ratio,
    random_state=42,
    eval_metric='logloss',
    reg_alpha=0.1,
    reg_lambda=1.0
)

model.fit(X_train, y_train)

y_val_proba = model.predict_proba(X_train)[:, 1]

thresholds = np.linspace(0.1, 0.9, 50)
best_threshold = 0.5
best_f1 = 0

for t in thresholds:
    y_pred_t = (y_val_proba >= t).astype(int)
    f1_t = f1_score(y_train, y_pred_t)
    if f1_t > best_f1:
        best_f1 = f1_t
        best_threshold = t

print(f"Лучший порог: {best_threshold:.3f}, F1 на train: {best_f1:.4f}")

print("\nОценка на тестовой выборке")

y_test_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_test_proba >= best_threshold).astype(int)

pr_auc = average_precision_score(y_test, y_test_proba)
f1 = f1_score(y_test, y_pred)

print(f"PR-AUC: {pr_auc:.4f}")
print(f"F1-score: {f1:.4f}")

print("\nВажность признаков")

importance = pd.DataFrame({
    'feature': X_raw.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance.head(10))

plt.figure(figsize=(10, 6))
plt.barh(importance['feature'].head(10), importance['importance'].head(10))
plt.xlabel('Важность')
plt.title('Топ-10 важных признаков (с обработкой выбросов)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("\nМатрица ошибок")

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Нет буфера', 'Буфер'],
            yticklabels=['Нет буфера', 'Буфер'])
plt.xlabel('Предсказано')
plt.ylabel('Истина')
plt.title(f'Матрица ошибок (порог={best_threshold:.3f})')
plt.tight_layout()
plt.show()

print("\nPR-кривая")

precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, 'b-', linewidth=2)
plt.fill_between(recall, precision, alpha=0.3)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'PR-кривая (AUC = {pr_auc:.4f})')
plt.grid(True, alpha=0.3)
plt.show()

print(f"\nИтог")
print(f"PR-AUC: {pr_auc:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Оптимальный порог: {best_threshold:.3f}")
