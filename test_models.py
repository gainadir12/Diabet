import pandas as pd
from sklearn.metrics import classification_report
import joblib

# Загрузка данных
data = pd.read_csv('diabetes.csv')

# Разделение данных на обучающий и тестовый наборы
X_test = data.drop('Outcome', axis=1)
y_test = data['Outcome']

# Загрузка моделей
model_RF = joblib.load('model_RF.joblib')
model_LR = joblib.load('model_LR.joblib')

# Оценка качества модели случайного леса на тестовом наборе
accuracy_RF = model_RF.score(X_test, y_test)
print("RandomForestClassifier Accuracy:", accuracy_RF)

# Классификационный отчет для модели случайного леса
y_pred_RF = model_RF.predict(X_test)
report_RF = classification_report(y_test, y_pred_RF)
print(report_RF)

# Оценка качества модели логистической регрессии на тестовом наборе
accuracy_LR = model_LR.score(X_test, y_test)
print("LogisticRegression Accuracy:", accuracy_LR)

# Классификационный отчет для модели логистической регрессии
y_pred_LR = model_LR.predict(X_test)
report_LR = classification_report(y_test, y_pred_LR)
print(report_LR)

# Сохранение результатов в файл
with open('result.txt', 'w') as f:
    f.write(f"RandomForestClassifier Accuracy: {accuracy_RF}\n")
    f.write(f"RandomForestClassifier Classification Report:\n{report_RF}\n\n")
    f.write(f"LogisticRegression Accuracy: {accuracy_LR}\n")
    f.write(f"LogisticRegression Classification Report:\n{report_LR}")
