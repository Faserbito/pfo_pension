import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
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

# Чтние данных из csv файла
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
def load_model_and_predict(X, model_path='early_retirement_model.pkl', task_message="Предсказание выхода на пенсию..."):
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
def calculate_metrics(y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    return f1, accuracy, precision, recall

# Основная функция выполнения всех шагов
def main():
    client_df = load_data('cntrbtrs_clnts_ops_trn.csv', task_message="Считываю данные клиента...")
    movement_df = load_data('trnsctns_ops_trn.csv', task_message="Считываю данные о движении средств...")
    
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
        'slctn_nmbr', 'gndr', 'prsnt_age', 'accnt_bgn_date', 'cprtn_prd_d', 
        'accnt_status', 'pnsn_age', 'prvs_npf', 'brth_plc', 'addrss_type', 'rgn', 
        'dstrct', 'city', 'sttlmnt', 'pstl_code', 'okato', 'phn', 'email', 
        'lk', 'assgn_npo', 'assgn_ops', 'mvmnt_type', 'sum_type', 'sum'
    ]]
    y = full_df['erly_pnsn_flg']
    
    y_pred = load_model_and_predict(X)
    save_predictions(full_df['accnt_id'], y_pred)

    # Расчет и вывод метрик
    calculate_metrics(y, y_pred)
    
if __name__ == "__main__":
    main()



# # Расчет f1 метрики
# def evaluate_model(y_true, y_pred, task_message="Оценка модели..."):
#     global done
#     done = False
#     t = spinning_cursor(task_message)
#     f1 = f1_score(y_true, y_pred, average='weighted')
#     done = True
#     t.join()
#     print(f'F1-score для предсказания досрочного выхода на пенсию: {f1}')
#     return f1

# evaluate_model(y, y_pred)