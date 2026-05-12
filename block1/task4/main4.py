# K9: Практическое задание 4 - Временные ряды (авиапассажиры)
# K9: Студент: Липенков Александр ПИШ-212
# K9: Среда: PyCharm Professional, Python 3.12

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# ------------------- 1. Загрузка данных -------------------
file_path = "passengers.csv"   # файл в той же папке
df = pd.read_csv(file_path)

print("Первые 5 строк данных:")
print(df.head())
print("\nИнформация о данных:")
print(df.info())

# Преобразуем столбец Month в datetime и установим как индекс
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
print(f"\nПериод данных: с {df.index.min()} по {df.index.max()}")

# ------------------- 2. Визуализация временного ряда -------------------
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['#Passengers'], color='blue', linewidth=1.5)
plt.title('Временной ряд: количество авиапассажиров (1949–1960)')
plt.xlabel('Дата')
plt.ylabel('Количество пассажиров (тыс.)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('time_series.png', dpi=150)
plt.show()

# ------------------- 3. Декомпозиция временного ряда -------------------
# Аддитивная декомпозиция (предполагаем линейный тренд и сезонность)
decomposition = seasonal_decompose(df['#Passengers'], model='additive', period=12)
fig = decomposition.plot()
fig.set_size_inches(12, 8)
plt.suptitle('Декомпозиция временного ряда (аддитивная)', fontsize=14)
plt.tight_layout()
plt.savefig('decomposition.png', dpi=150)
plt.show()

# ------------------- 4. Построение линейной модели тренда -------------------
# Создаём числовой признак времени (месяцы от начала)
df['time_index'] = np.arange(1, len(df) + 1)   # январь 1949 = 1
X = df[['time_index']]
y = df['#Passengers']

# Линейная регрессия (тренд)
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Коэффициент детерминации R^2
r2 = r2_score(y, y_pred)
print("\n" + "="*50)
print("ЛИНЕЙНАЯ МОДЕЛЬ ТРЕНДА")
print("="*50)
print(f"Уравнение тренда: Пассажиры = {model.intercept_:.2f} + {model.coef_[0]:.4f} * t")
print(f"где t – номер месяца (t=1 для января 1949)")
print(f"Коэффициент детерминации R² = {r2:.6f}")

# Оценка качества
if r2 >= 0.8:
    quality = "очень хорошее"
elif r2 >= 0.6:
    quality = "хорошее"
else:
    quality = "среднее/низкое"
print(f"\nВывод: Качество линейной модели тренда – {quality} (R² = {r2:.4f})")
print("В модели не учтена сезонность, поэтому R² ограничен.")

# Визуализация тренда
plt.figure(figsize=(12, 5))
plt.plot(df.index, y, label='Фактические данные', color='blue', alpha=0.7)
plt.plot(df.index, y_pred, label='Линейный тренд', color='red', linestyle='--', linewidth=2)
plt.title('Линейная модель тренда временного ряда')
plt.xlabel('Дата')
plt.ylabel('Количество пассажиров')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('linear_trend.png', dpi=150)
plt.show()

# ------------------- 5. Прогноз на январь и март 1961 -------------------
# t для января 1961: после декабря 1960 (t=144) следующий t=145
# Для марта 1961: t = 145 + 2 = 147
t_jan_1961 = 145
t_mar_1961 = 147

pred_jan = model.intercept_ + model.coef_[0] * t_jan_1961
pred_mar = model.intercept_ + model.coef_[0] * t_mar_1961

print("\n" + "="*50)
print("ПРОГНОЗ ПО ЛИНЕЙНОЙ МОДЕЛИ ТРЕНДА")
print("="*50)
print(f"Прогноз на январь 1961 (t={t_jan_1961}): {pred_jan:.2f} тыс. пассажиров")
print(f"Прогноз на март 1961   (t={t_mar_1961}): {pred_mar:.2f} тыс. пассажиров")
print("\nПримечание: прогноз учитывает только общий тренд и не учитывает сезонные колебания.")
print("Реальные значения будут выше летом и ниже весной/осенью.")

# ------------------- Дополнительно: прогноз с учётом сезонности (не обязательно, но для полноты) -------------------
# Можно добавить сезонные коэффициенты из декомпозиции
seasonal = decomposition.seasonal[:12]  # средние сезонные эффекты
# Для января (месяц 1) и марта (месяц 3) в цикле 12
# Индексы: январь - 0, март - 2
seasonal_jan = seasonal.iloc[0]   # январь
seasonal_mar = seasonal.iloc[2]   # март

pred_jan_seas = pred_jan + seasonal_jan
pred_mar_seas = pred_mar + seasonal_mar

print("\n" + "="*50)
print("ПРОГНОЗ С УЧЁТОМ СЕЗОННОСТИ (аддитивная модель)")
print("="*50)
print(f"Сезонная поправка для января: {seasonal_jan:.2f}")
print(f"Сезонная поправка для марта:  {seasonal_mar:.2f}")
print(f"Прогноз на январь 1961 с учётом сезонности: {pred_jan_seas:.2f} тыс.")
print(f"Прогноз на март 1961 с учётом сезонности:   {pred_mar_seas:.2f} тыс.")

# ------------------- Ссылка на GitHub репозиторий -------------------
print("\n" + "="*50)
print("ССЫЛКА НА РЕПОЗИТОРИЙ")
print("="*50)
print("GitHub: https://github.com/Kango911/dosusu2sem")