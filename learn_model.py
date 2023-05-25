import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib
import time

# Загрузка данных
data = pd.read_csv('other_rows.csv')
# # берем последние 10 строк из датафрейма
# last_10_rows = data.tail(10)

# # берем все строки, кроме последних 10
# other_rows = data.head(len(data) - 10)
# # сохраняем данные с последними 10 строками в файл
# last_10_rows.to_csv('last_10_rows.csv', index=False)

# # сохраняем остальные данные в другой файл
# other_rows.to_csv('other_rows.csv', index=False)
# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(data.drop('Outcome', axis=1), data['Outcome'], test_size=0.2, random_state=42)

# Создание и обучение модели случайного леса
model_RF = RandomForestClassifier(random_state=42)
model_RF.fit(X_train, y_train)

# Оценка качества модели случайного леса на тестовом наборе
accuracy_RF = model_RF.score(X_test, y_test)
print("RandomForestClassifier Accuracy:", accuracy_RF)

# Классификационный отчет для модели случайного леса
start = time.time()
y_pred_RF = model_RF.predict(X_test)
result_RF_time =  time.time() - start
report_RF = classification_report(y_test, y_pred_RF)
print(report_RF)
joblib.dump(model_RF, 'model_RF.joblib')

# Создание и обучение модели логистической регрессии
model_LR = LogisticRegression(random_state=42)
model_LR.fit(X_train, y_train)

# Оценка качества модели логистической регрессии на тестовом наборе
accuracy_LR = model_LR.score(X_test, y_test)
print("LogisticRegression Accuracy:", accuracy_LR)

# Классификационный отчет для модели логистической регрессии
start = time.time()
y_pred_LR = model_LR.predict(X_test)
result_LR_time = time.time() - start 
report_LR = classification_report(y_test, y_pred_LR)
print(report_LR)
joblib.dump(model_LR, 'model_LR.joblib')

# Сохранение результатов в файл
with open('result.txt', 'w') as f:
    f.write(f"RandomForestClassifier Accuracy: {accuracy_RF}\n")
    f.write(f"RandomForestClassifier Time Speed: {result_RF_time}\n")
    f.write(f"RandomForestClassifier Classification Report:\n{report_RF}\n\n")
    f.write(f"LogisticRegression Accuracy: {accuracy_LR}\n")
    f.write(f"LogisticRegression Time Speed: {result_LR_time}\n")
    f.write(f"LogisticRegression Classification Report:\n{report_LR}")
