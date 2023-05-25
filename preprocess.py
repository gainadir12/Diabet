import pandas as pd
# data = pd.read_csv('diabetes.csv')
# print(data.head())
# Загрузка данных
data = pd.read_csv('diabetes.csv')
# берем последние 10 строк из датафрейма
last_10_rows = data.tail(77)

# берем все строки, кроме последних 10
other_rows = data.head(len(data) - 77)
# сохраняем данные с последними 10 строками в файл
last_10_rows.to_csv('last_10_rows.csv', index=False)

# сохраняем остальные данные в другой файл
other_rows.to_csv('other_rows.csv', index=False)
