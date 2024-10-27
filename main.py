import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import threading
import time
import itertools
import warnings

# Подавление мешающих предупреждений
warnings.filterwarnings("ignore")

# Глобальный флаг для работы "загрузки"
done = False

# Функция для отображения индикатора загрузки
def spinning_cursor(task_message):
    def spinner():
        global done
        for cursor in itertools.cycle(['|', '/', '-', '\\']):
            if done:
                break
            print(f'\r{task_message} {cursor}', end='')
            time.sleep(0.1)
        print(f'\r{task_message} Готово!')

    t = threading.Thread(target=spinner)
    t.start()
    return t

# Функция для обучения модели
def train_model(X, y, task_message="Обучаю модель для предсказания досрочного выхода на пенсию..."):
    global done
    done = False
    t = spinning_cursor(task_message)

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучение модели
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    done = True
    t.join()
    print(f'\r{task_message} Готово!')
    
    return model

# Чтение данных из csv файла
def load_data(file_path, encoding='cp1251', sep=";", task_message="Загрузка данных..."):
    global done
    done = False
    t = spinning_cursor(task_message)
    data = pd.read_csv(file_path, encoding=encoding, sep=sep)
    done = True
    t.join()
    return data

# Преобразование текстовых данных "нет" и "да" в числовые для работы модели
def preprocess_data(df):
    return df.replace({'нет': 0, 'да': 1})

# Подготовка данных клиента
def process_client_data(client_df):
    client_df['accnt_bgn_date'] = pd.to_datetime(client_df['accnt_bgn_date'])
    client_df['prsnt_age'] = client_df['prsnt_age'].fillna(client_df['brth_yr'].apply(lambda x: 2024 - x))
    client_df['erly_pnsn_flg'] = client_df['erly_pnsn_flg'].astype(int)
    return client_df

# Объединение данных клиента и транзакций
def merge_data(client_df, movement_df, task_message="Объединение данных..."):
    global done
    done = False
    t = spinning_cursor(task_message)
    full_df = pd.merge(client_df, movement_df, on='accnt_id', how='left')
    full_df = full_df.groupby('accnt_id').first().reset_index()
    done = True
    t.join()
    return full_df

# Преобразование категориальных данных в числовые
def encode_categorical(full_df, categorical_columns):
    le = LabelEncoder()
    for col in categorical_columns:
        full_df[col] = le.fit_transform(full_df[col].astype(str))
    return full_df

# Преобразование временных меток в Unix-время для работы модели
def convert_dates_to_unix(full_df):
    full_df['accnt_bgn_date'] = pd.to_datetime(full_df['accnt_bgn_date']).astype('int64') // 10**9
    full_df['oprtn_date'] = pd.to_datetime(full_df['oprtn_date']).astype('int64') // 10**9
    return full_df

# Загрузка модели и предсказание выхода на пенсию
def load_model_and_predict(X, model_path='early_retirement_model_100.pkl', task_message="Предсказание выхода на пенсию..."):
    global done
    done = False
    t = spinning_cursor(task_message)
    model = joblib.load(model_path)
    y_pred = model.predict(X)
    done = True
    t.join()
    return y_pred

# Сохранение результата в файл 'submission.csv'
def save_predictions(accnt_ids, predictions, filename='submission.csv', task_message="Сохранение результатов..."):
    global done
    done = False
    t = spinning_cursor(task_message)
    result_df = pd.DataFrame({'accnt_id': accnt_ids, 'erly_pnsn_flg': predictions})
    result_df.to_csv(filename, index=False, encoding='utf-8')
    done = True
    t.join()
    print("Файл с предсказаниями сохранён как 'submission.csv'")

# Функция для расчета метрик качества модели
def calculate_metrics(y_true, y_pred, y_proba=None):
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    logloss = log_loss(y_true, y_proba) if y_proba is not None else None
    auc_roc = roc_auc_score(y_true, y_proba) if y_proba is not None else None
    
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    if logloss is not None:
        print(f"Log Loss: {logloss:.4f}")
    if auc_roc is not None:
        print(f"AUC-ROC: {auc_roc:.4f}")
    return f1, accuracy, precision, recall, logloss, auc_roc

# Функция для построения и вывода графиков по метрикам
def plot_selected_metrics(y_true, y_pred, y_proba):
    f1 = f1_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_proba) if y_proba is not None else None
    logloss = log_loss(y_true, y_proba) if y_proba is not None else None

    metrics = {
        "F1 Score": f1,
        "AUC-ROC": auc_roc,
        "Log Loss": logloss
    }
    
    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette="viridis", width=0.2)
    
    for p in bar_plot.patches:
        bar_plot.annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='bottom', fontsize=12)
    plt.title("Результаты по метрикам F1, AUC-ROC и Log Loss", pad = 20)
    plt.ylabel("Значение")
    plt.xlabel("Метрика")
    plt.ylim(0, 1 if logloss is None else max(1, logloss + 0.1))
    plt.show()

# Функция для построения и вывода графика "Доля преждевременного выхода на пенсию по группам (каждые 3 года)"
def plot_early_retirement_statistics(client_df):
    client_df['accnt_bgn_year'] = pd.to_datetime(client_df['accnt_bgn_date'], format='%d.%m.%Y').dt.year

    client_df['year_group'] = (client_df['accnt_bgn_year'] // 3) * 3

    yearly_stats = client_df.groupby('year_group')['erly_pnsn_flg'].mean().reset_index()
    yearly_stats['count'] = client_df.groupby('year_group')['erly_pnsn_flg'].count().values
    yearly_stats['percentage'] = (yearly_stats['count'] / yearly_stats['count'].sum()) * 100

    plt.figure(figsize=(8, 10))
    wedges, _ = plt.pie(yearly_stats['count'], labels=None, autopct=None, startangle=140,
                                        radius=0.8, pctdistance=0.75)

    legend_labels = [f"{year} - {count} ({percentage:.1f}%)" 
                     for year, count, percentage in zip(yearly_stats['year_group'].astype(str), 
                                                         yearly_stats['count'], 
                                                         yearly_stats['percentage'])]
    plt.legend(wedges, legend_labels, title="Годы (Количество и Процент)", loc="upper left", 
               bbox_to_anchor=(0.9, 1), fontsize=8)

    plt.title("Доля преждевременного выхода на пенсию по группам (каждые 3 года)")
    plt.axis('equal')

    plt.tight_layout()
    plt.show()

# Основная функция выполнения всех шагов
def main():
    client_df = load_data('train\\cntrbtrs_clnts_ops_trn.csv', task_message="Считываю данные клиента...")
    movement_df = load_data('train\\trnsctns_ops_trn.csv', task_message="Считываю данные о транзакциях...")
    
    client_df = preprocess_data(client_df)
    movement_df = preprocess_data(movement_df)
    
    client_df = process_client_data(client_df)
    full_df = merge_data(client_df, movement_df)
    
    # Список с названиями столбцов с текстовыми значениями из объединенных данных клиента и транзакций, которые необходимо преобразовать в числовой формат для работы модели
    categorical_columns = [
        'gndr', 'accnt_status', 'prvs_npf', 'brth_plc', 'addrss_type', 'rgn', 
        'dstrct', 'city', 'sttlmnt', 'pstl_code', 'okato', 'mvmnt_type', 'sum_type', 'cmmnt'
    ]
    
    full_df = encode_categorical(full_df, categorical_columns)
    full_df = convert_dates_to_unix(full_df)
    
    X = full_df[[
        'gndr', 'prsnt_age', 'accnt_bgn_date', 'cprtn_prd_d', 
        'accnt_status', 'pnsn_age', 'rgn', 'dstrct', 'lk', 'assgn_npo', 'assgn_ops', 'mvmnt_type'
    ]]
    y = full_df['erly_pnsn_flg']
    
    if os.path.isfile('early_retirement_model.pkl'):
        y_proba = load_model_and_predict(X)
        y_pred = (y_proba >= 0.5).astype(int)
        save_predictions(full_df['accnt_id'], y_pred)

        # Расчет и вывод метрик
        calculate_metrics(y, y_pred, y_proba)
        plot_selected_metrics(y, y_pred, y_proba)
        plot_early_retirement_statistics(client_df)
    else:
        # Обучение модели
        model = train_model(X, y)

        # Сохранение модели
        joblib.dump(model, 'early_retirement_model.pkl')
        
if __name__ == "__main__":
    main()