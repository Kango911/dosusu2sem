# K9: Практическое задание 3 - Полиномиальная регрессия
# K9: Студент: Липенков Александр ПИШ-212
# K9: Среда: PyCharm Professional, Python 3.12

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')

# ------------------- 1. Загрузка данных -------------------
file_path = "Полином_ЛР.xlsx"  # убедитесь, что файл в той же папке
df = pd.read_excel(file_path, sheet_name="Лист1")

X = df['X'].values.reshape(-1, 1)
Y = df['Y'].values

print("Данные загружены. Размер:", df.shape)
print("Первые 5 строк:")
print(df.head())

# ------------------- 2. Построение моделей -------------------
degrees = [1, 2, 3]
models = {}
r2_scores = {}

plt.figure(figsize=(12, 8))
plt.scatter(X, Y, color='blue', alpha=0.6, label='Исходные данные')

for deg in degrees:
    # Создаём полиномиальные признаки
    poly = PolynomialFeatures(degree=deg)
    X_poly = poly.fit_transform(X)

    # Обучаем модель
    model = LinearRegression()
    model.fit(X_poly, Y)
    y_pred = model.predict(X_poly)

    # Вычисляем R^2
    r2 = r2_score(Y, y_pred)
    r2_scores[deg] = r2
    models[deg] = (poly, model)

    print(f"\nПолином степени {deg}:")
    print(f"  Коэффициенты: {model.intercept_:.4f}, {model.coef_[1:]}")
    print(f"  R^2 = {r2:.6f}")

    # Сортируем X для гладкой линии
    X_sorted = np.sort(X, axis=0)
    X_poly_sorted = poly.transform(X_sorted)
    y_curve = model.predict(X_poly_sorted)
    plt.plot(X_sorted, y_curve, linewidth=2, label=f'Степень {deg} (R²={r2:.4f})')

# ------------------- 3. Выбор лучшей модели -------------------
best_deg = max(r2_scores, key=r2_scores.get)
print("\n" + "=" * 50)
print(f"ЛУЧШАЯ МОДЕЛЬ: полином степени {best_deg}")
print(f"Максимальный R² = {r2_scores[best_deg]:.6f}")
print("=" * 50)

# ------------------- 4. График для лучшей модели -------------------
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Полиномиальная регрессия: сравнение степеней\nЛучшая – степень {best_deg} (R²={r2_scores[best_deg]:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('polynomial_regression.png', dpi=150)
plt.show()

# ------------------- 5. Отдельный график только для лучшей модели -------------------
best_poly, best_model = models[best_deg]
y_pred_best = best_model.predict(best_poly.transform(X))

plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='blue', alpha=0.6, label='Исходные данные')
X_sorted = np.sort(X, axis=0)
X_poly_sorted = best_poly.transform(X_sorted)
y_curve_best = best_model.predict(X_poly_sorted)
plt.plot(X_sorted, y_curve_best, color='red', linewidth=2, label=f'Полином {best_deg} степени')
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Лучшая модель: полином {best_deg} степени\nR² = {r2_scores[best_deg]:.4f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('best_polynomial.png', dpi=150)
plt.show()

# ------------------- 6. Анализ и построение полинома 4-й степени -------------------
print("\n" + "=" * 50)
print("ВОПРОС 7. Как построить полином четвёртой степени?")
print("=" * 50)
print("Для построения полинома 4-й степени достаточно изменить значение degree=4")
print("в функции PolynomialFeatures. Ниже приведён код для степени 4:")

# Демонстрация для степени 4
deg4 = 4
poly4 = PolynomialFeatures(degree=deg4)
X_poly4 = poly4.fit_transform(X)
model4 = LinearRegression()
model4.fit(X_poly4, Y)
y_pred4 = model4.predict(X_poly4)
r2_4 = r2_score(Y, y_pred4)

print(f"\nПолином степени 4:")
print(f"  R² = {r2_4:.6f}")
print(f"  Коэффициенты: intercept = {model4.intercept_:.4f}")
print(f"  Коэффициенты признаков: {model4.coef_[1:]}")

# Визуализация полинома 4-й степени (опционально)
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='blue', alpha=0.6, label='Данные')
X_sorted = np.sort(X, axis=0)
X_poly4_sorted = poly4.transform(X_sorted)
y_curve4 = model4.predict(X_poly4_sorted)
plt.plot(X_sorted, y_curve4, color='green', linewidth=2, label=f'Полином степени 4 (R²={r2_4:.4f})')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Полиномиальная регрессия 4-й степени')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('degree4_polynomial.png', dpi=150)
plt.show()

# ------------------- 7. Выводы -------------------
print("\n" + "=" * 50)
print("ВЫВОДЫ")
print("=" * 50)
print(f"• Линейная модель (степень 1) дала R² = {r2_scores[1]:.4f}")
print(f"• Полином 2-й степени: R² = {r2_scores[2]:.4f}")
print(f"• Полином 3-й степени: R² = {r2_scores[3]:.4f}")
print(f"• Полином 4-й степени: R² = {r2_4:.4f}")
print(f"\nЛучшая модель среди 1-3 степеней – степень {best_deg} (R² = {r2_scores[best_deg]:.4f})")
if r2_4 > r2_scores[best_deg]:
    print("Полином 4-й степени оказался ещё лучше (R² выше).")
else:
    print("Полином 4-й степени не превзошёл лучшую из первых трёх моделей.")
print("\nРекомендация: использовать полином степени, дающий максимальный R², но остерегаться переобучения.")

# ------------------- Ссылка на GitHub репозиторий -------------------
print("\n" + "=" * 50)
print("ССЫЛКА НА РЕПОЗИТОРИЙ")
print("=" * 50)
print("GitHub: https://github.com/Kango911/dosusu2sem")