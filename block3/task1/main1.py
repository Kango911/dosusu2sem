# K9: Практическое задание "Большие данные" - Анализ данных такси NYC
# K9: Студент: Липенков Александр ПИШ-212
# K9: Данные: Yellow Taxi Trip Records за сентябрь 2016 года
# K9: Источник: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
# K9: Файл: yellow_tripdata_2016-09.parquet
# K9: Среда выполнения: PyCharm Professional, Python 3.12

import dask.dataframe as dd
import warnings
import numpy as np
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

# ------------------- Задача 1: Загрузка и анализ (10 мин) -------------------
print("=" * 60)
print("ЗАДАЧА 1: ЗАГРУЗКА И АНАЛИЗ ДАННЫХ")
print("=" * 60)

# 1.1 Загружаем данные (Parquet файл)
ddf = dd.read_parquet('yellow_tripdata_2016-09.parquet')

# 1.2 Количество записей и список колонок
print(f"\nКоличество записей в датасете: {ddf.shape[0].compute():,}")
print("Список колонок:")
for col in ddf.columns:
    print(f"  - {col}")

# 1.3 Среднее расстояние и средняя стоимость поездки
mean_distance = ddf['trip_distance'].mean().compute()
mean_amount = ddf['total_amount'].mean().compute()
print(f"\nСреднее расстояние поездки: {mean_distance:.2f} миль")
print(f"Средняя стоимость поездки: ${mean_amount:.2f}")

# ------------------- Задача 2: Анализ по часам (10 мин) -------------------
print("\n" + "=" * 60)
print("ЗАДАЧА 2: АНАЛИЗ ПО ЧАСАМ")
print("=" * 60)

# 2.1 Добавляем колонку pickup_hour (час начала поездки)
ddf['pickup_hour'] = ddf['tpep_pickup_datetime'].dt.hour

# 2.2 Час с максимальным количеством поездок
hour_counts = ddf.groupby('pickup_hour').size().compute()
peak_hour = hour_counts.idxmax()
peak_hour_count = hour_counts.max()
print(f"\nЧас с максимальным количеством поездок: {peak_hour}:00")
print(f"Количество поездок: {peak_hour_count:,}")

# 2.3 Средняя стоимость поездки по часам
hourly_avg_amount = ddf.groupby('pickup_hour')['total_amount'].mean().compute()
most_expensive_hour = hourly_avg_amount.idxmax()
most_expensive_value = hourly_avg_amount.max()
print(f"\nСамый дорогой час (средняя стоимость): {most_expensive_hour}:00 (${most_expensive_value:.2f})")

# Дополнительная таблица
print("\nСредняя стоимость по часам (0–7):")
for hour in range(8):
    val = hourly_avg_amount.get(hour, 0)
    print(f"  {hour:02d}:00 - ${val:.2f}")

# ------------------- Задача 3: Модель предсказания (15 мин) -------------------
print("\n" + "=" * 60)
print("ЗАДАЧА 3: МОДЕЛЬ ПРЕДСКАЗАНИЯ СТОИМОСТИ ПОЕЗДКИ")
print("=" * 60)

# 3.1 Очистка данных: удаляем записи с trip_distance <= 0 или total_amount <= 0
print("\nОчистка данных...")
original_count = ddf.shape[0].compute()
ddf_clean = ddf[(ddf['trip_distance'] > 0) & (ddf['total_amount'] > 0)]
clean_count = ddf_clean.shape[0].compute()
print(f"До очистки: {original_count:,} записей")
print(f"После очистки: {clean_count:,} записей")
print(f"Удалено записей: {original_count - clean_count:,}")

# 3.2 Подготовка признаков (trip_distance, pickup_hour)
# Вместо работы с dask_ml (проблемы с чанками) – загружаем необходимые данные в память
# 10 млн строк × 2 признака = около 160 МБ – допустимо для большинства ПК.
print("\nЗагрузка данных в память для обучения модели...")
X_df = ddf_clean[['trip_distance', 'pickup_hour']].compute()
y_df = ddf_clean['total_amount'].compute()

# Проверяем размеры
print(f"Размер X: {X_df.shape}, размер y: {y_df.shape}")

# 3.3 Обучение модели линейной регрессии (sklearn)
print("\nОбучение модели линейной регрессии (sklearn)...")
model = LinearRegression()
model.fit(X_df, y_df)

# Коэффициенты
coef_distance = model.coef_[0]
coef_hour = model.coef_[1]
intercept = model.intercept_

# R²
r2 = model.score(X_df, y_df)

print(f"\nУравнение регрессии:")
print(f"total_amount = {intercept:.4f} + {coef_distance:.4f} * trip_distance + {coef_hour:.4f} * pickup_hour")
print(f"Коэффициент детерминации R² = {r2:.4f}")

# 3.4 Анализ влияния факторов
print("\nВлияние факторов на стоимость поездки:")
print(f"  • Расстояние (trip_distance): коэффициент = {coef_distance:.4f}")
print(f"    → увеличение расстояния на 1 милю увеличивает стоимость на ${coef_distance:.2f}")
print(f"  • Час начала (pickup_hour): коэффициент = {coef_hour:.4f}")
print(f"    → изменение часа начала на 1 единицу изменяет стоимость на ${coef_hour:.2f}")

if abs(coef_distance) > abs(coef_hour):
    stronger = "расстояние поездки (trip_distance)"
    stronger_val = abs(coef_distance)
    weaker = "час начала (pickup_hour)"
    weaker_val = abs(coef_hour)
else:
    stronger = "час начала (pickup_hour)"
    stronger_val = abs(coef_hour)
    weaker = "расстояние поездки (trip_distance)"
    weaker_val = abs(coef_distance)

print(f"\nВЫВОД: Сильнее на стоимость влияет {stronger}")
print(f"  (коэффициент {stronger_val:.4f} против {weaker_val:.4f} у {weaker})")

if coef_hour > 0:
    print("\n  Положительный коэффициент часа означает, что поздние часы (вечер/ночь)")
    print("  в среднем дают более дорогие поездки (возможно из-за ночных тарифов).")
else:
    print("\n  Отрицательный коэффициент часа означает, что дневные поездки дороже ночных.")

# ------------------- Итоговые выводы -------------------
print("\n" + "=" * 60)
print("ИТОГОВЫЕ ВЫВОДЫ ПО ЗАДАНИЮ")
print("=" * 60)
print(f"1. Среднее расстояние поездки: {mean_distance:.2f} миль, средняя стоимость: ${mean_amount:.2f}.")
print(f"2. Пик активности такси приходится на {peak_hour}:00 ({peak_hour_count:,} поездок).")
print(f"3. Самая высокая средняя стоимость поездки наблюдается в {most_expensive_hour}:00 (${most_expensive_value:.2f}).")
print(f"4. Модель линейной регрессии объясняет {r2 * 100:.1f}% вариации стоимости (R² = {r2:.4f}).")
print("5. На стоимость сильнее влияет расстояние поездки (коэффициент выше).")
print("6. Модель может быть улучшена добавлением дополнительных признаков (число пассажиров, день недели, район и т.д.).")

# ------------------- Ссылка на GitHub репозиторий -------------------
print("\n" + "=" * 60)
print("ССЫЛКА НА РЕПОЗИТОРИЙ")
print("=" * 60)
print("GitHub: https://github.com/Kango911/dosusu2sem")