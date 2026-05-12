# K9: Практическое задание 1 - Парная линейная регрессия
# K9: Студент: Липенков Александр ПИШ-212
# K9: Среда: PyCharm Professional, Python 3.12

import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ------------------- Часть 1. Загрузка данных -------------------
file_path = "Статистика по рейсам (1).xlsx"   # К9: убедитесь, что файл лежит в папке task1
df = pd.read_excel(file_path, sheet_name="Данные")

print("Первые 5 строк данных:")
print(df.head())
print("\nРазмер данных:", df.shape)

# Выбираем переменные
X = df["Вес транспорта, т"]
Y = df["Расходтоплива, л/100 км"]

# ------------------- Часть 1. Расчёт по формулам (pandas) -------------------
n = len(X)
mean_x = X.mean()
mean_y = Y.mean()

# Ковариация и дисперсия по формулам
cov_xy = ((X - mean_x) * (Y - mean_y)).sum() / n
var_x = ((X - mean_x) ** 2).sum() / n

# Коэффициенты регрессии
b1 = cov_xy / var_x
b0 = mean_y - b1 * mean_x

print("\n" + "="*50)
print("ЧАСТЬ 1. Регрессия через формулы Pandas")
print("="*50)
print(f"Коэффициент b0 (intercept): {b0:.6f}")
print(f"Коэффициент b1 (slope):     {b1:.6f}")
print(f"\nУравнение регрессии:")
print(f"Расход топлива = {b0:.6f} + {b1:.6f} * Вес транспорта")

# Коэффициент детерминации R^2 = квадрат корреляции Пирсона
corr = X.corr(Y)
r_squared = corr ** 2
print(f"\nКоэффициент корреляции Пирсона: {corr:.6f}")
print(f"Коэффициент детерминации R^2:   {r_squared:.6f}")

# Вывод о качестве уравнения
print("\nВывод по Части 1:")
if r_squared > 0.7:
    print("  R^2 > 0.7 → уравнение хорошо описывает зависимость.")
elif r_squared > 0.5:
    print("  R^2 между 0.5 и 0.7 → умеренная объясняющая способность.")
else:
    print("  R^2 <= 0.5 → модель плохо описывает данные.")

# ------------------- Часть 2. Регрессия через statsmodels -------------------
print("\n" + "="*50)
print("ЧАСТЬ 2. Регрессия через statsmodels")
print("="*50)

X_sm = sm.add_constant(X)   # добавляем константу
model = sm.OLS(Y, X_sm).fit()

print(model.summary())

b0_sm = model.params['const']
b1_sm = model.params['Вес транспорта, т']
r2_sm = model.rsquared

print(f"\nКоэффициенты из statsmodels:")
print(f"b0 (intercept) = {b0_sm:.6f}")
print(f"b1 (slope)     = {b1_sm:.6f}")
print(f"R^2 (statsmodels) = {r2_sm:.6f}")

# ------------------- Сравнение результатов -------------------
print("\n" + "="*50)
print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
print("="*50)
print(f"b0:  Pandas = {b0:.8f} | statsmodels = {b0_sm:.8f}  (разница: {abs(b0-b0_sm):.2e})")
print(f"b1:  Pandas = {b1:.8f} | statsmodels = {b1_sm:.8f}  (разница: {abs(b1-b1_sm):.2e})")
print(f"R^2: Pandas = {r_squared:.8f} | statsmodels = {r2_sm:.8f}  (разница: {abs(r_squared-r2_sm):.2e})")

print("\nВЫВОДЫ ПО ЗАДАНИЮ:")
print("• Ручной расчёт (pandas) и statsmodels дают одинаковые коэффициенты.")
print("• Коэффициент детерминации R^2 = {:.4f}".format(r_squared))
print("• Вес транспорта объясняет {:.1f}% вариации расхода топлива.".format(r_squared*100))