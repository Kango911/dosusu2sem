# K9: Практическое задание 1 (блок 2) - Парная линейная регрессия
# K9: Студент: Липенков Александр ПИШ-212
# K9: Датасет: Driving Behavior (train_motion_data.csv)
# K9: Цель: построить парную регрессию между двумя сенсорными осями (AccX и AccY)

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ------------------- 1. Загрузка данных -------------------
# Пробуем найти файл в текущей папке или в подпапке block2
possible_paths = [
    "train_motion_data.csv",
    "block2/train_motion_data.csv",
    "../train_motion_data.csv",
    "../block2/train_motion_data.csv"
]

file_path = None
for path in possible_paths:
    if os.path.exists(path):
        file_path = path
        break

if file_path is None:
    print("Ошибка: файл train_motion_data.csv не найден!")
    print("Проверьте пути:", possible_paths)
    sys.exit(1)

print(f"Загрузка данных из: {file_path}")
df = pd.read_csv(file_path)
print(f"Размер данных: {df.shape[0]} строк, {df.shape[1]} столбцов")
print("Первые 3 строки:")
print(df.head(3))

# ------------------- 2. Выбор переменных -------------------
# Для парной регрессии возьмём X = AccX, Y = AccY (можно выбрать любую другую пару)
X = df['AccX']
Y = df['AccY']

print(f"\nИспользуемые переменные: X = AccX, Y = AccY")
print(f"Диапазон X: [{X.min():.3f}, {X.max():.3f}]")
print(f"Диапазон Y: [{Y.min():.3f}, {Y.max():.3f}]")

# ------------------- 3. Расчёт по формулам (pandas) -------------------
print("\n" + "="*60)
print("ЧАСТЬ 1. Регрессия через формулы Pandas")
print("="*60)

n = len(X)
mean_x = X.mean()
mean_y = Y.mean()

cov_xy = ((X - mean_x) * (Y - mean_y)).sum() / n
var_x = ((X - mean_x) ** 2).sum() / n

b1 = cov_xy / var_x          # коэффициент наклона
b0 = mean_y - b1 * mean_x    # свободный член

print(f"Коэффициент b0 (intercept): {b0:.6f}")
print(f"Коэффициент b1 (slope):     {b1:.6f}")
print(f"\nУравнение регрессии:")
print(f"AccY = {b0:.6f} + {b1:.6f} * AccX")

# Коэффициент детерминации R^2 = квадрат корреляции Пирсона
corr = X.corr(Y)
r_squared_part1 = corr ** 2
print(f"\nКоэффициент корреляции Пирсона: {corr:.6f}")
print(f"Коэффициент детерминации R^2:   {r_squared_part1:.6f}")

# Качество модели
print("\nВывод по Части 1:")
if abs(r_squared_part1) > 0.7:
    print("  R^2 > 0.7 → сильная линейная связь между AccX и AccY.")
elif abs(r_squared_part1) > 0.5:
    print("  R^2 между 0.5 и 0.7 → умеренная связь.")
else:
    print("  R^2 <= 0.5 → связь слабая, линейная модель плохо описывает данные.")

# ------------------- 4. Диаграмма рассеяния с линией регрессии -------------------
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, alpha=0.5, s=1, color='blue', label='Исходные данные')

# Линия регрессии
x_range = np.linspace(X.min(), X.max(), 100)
y_pred_line = b0 + b1 * x_range
plt.plot(x_range, y_pred_line, color='red', linewidth=2, label='Линия регрессии')

plt.xlabel('AccX (м/с²)')
plt.ylabel('AccY (м/с²)')
plt.title(f'Парная регрессия: AccY ~ AccX\nR² = {r_squared_part1:.4f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('parnaya_regressiya_block2.png', dpi=150)
plt.show()

# ------------------- 5. Регрессия через statsmodels -------------------
print("\n" + "="*60)
print("ЧАСТЬ 2. Регрессия через statsmodels")
print("="*60)

X_sm = sm.add_constant(X)   # добавляем константу
model = sm.OLS(Y, X_sm).fit()

# Полная сводка
print(model.summary())

b0_sm = model.params['const']
b1_sm = model.params['AccX']
r2_sm = model.rsquared

print(f"\nКоэффициенты из statsmodels:")
print(f"b0 (intercept) = {b0_sm:.6f}")
print(f"b1 (slope)     = {b1_sm:.6f}")
print(f"R^2 (statsmodels) = {r2_sm:.6f}")

# ------------------- 6. Сравнение результатов -------------------
print("\n" + "="*60)
print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
print("="*60)
print(f"b0:  Pandas = {b0:.8f} | statsmodels = {b0_sm:.8f}  (разница: {abs(b0-b0_sm):.2e})")
print(f"b1:  Pandas = {b1:.8f} | statsmodels = {b1_sm:.8f}  (разница: {abs(b1-b1_sm):.2e})")
print(f"R^2: Pandas = {r_squared_part1:.8f} | statsmodels = {r2_sm:.8f}  (разница: {abs(r_squared_part1-r2_sm):.2e})")

# ------------------- 7. Итоговые выводы -------------------
print("\n" + "="*60)
print("ВЫВОДЫ ПО ЗАДАНИЮ (блок 2)")
print("="*60)
print("1. Построена парная линейная регрессия между AccX (независимая) и AccY (зависимая).")
print("2. Ручной расчёт (pandas) и statsmodels дают идентичные коэффициенты.")
print(f"3. Коэффициент детерминации R² = {r2_sm:.4f} означает, что")
print(f"   {r2_sm*100:.1f}% вариации AccY объясняется изменением AccX.")
print("4. Связь между осями ускорения в данном датасете ", end="")
if r2_sm > 0.5:
    print("умеренная/сильная.")
else:
    print("слабая.")
print("5. Такая регрессия может быть полезна для анализа манёвров: "
      "например, при поворотах AccX и AccY коррелируют.")
print("6. График рассеяния с линией регрессии сохранён как 'parnaya_regressiya_block2.png'.")

# ------------------- Ссылка на GitHub репозиторий -------------------
print("\n" + "="*60)
print("ССЫЛКА НА РЕПОЗИТОРИЙ")
print("="*60)
print("GitHub: https://github.com/Kango911/dosusu2sem")