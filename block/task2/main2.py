# K9: Практическое задание 2 - Множественная линейная регрессия
# K9: Студент: Липенков Александр ПИШ-212
# K9: Среда: PyCharm Professional, Python 3.12

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
import sys
import warnings

warnings.filterwarnings('ignore')

# Настройка отображения графиков (для корректного вывода)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def main():
    # ------------------- 1. Загрузка данных -------------------
    file_path = "Статистика по рейсам (1).xlsx"

    # Если файл не в текущей папке, можно задать полный путь (раскомментируйте при необходимости)
    # file_path = r"C:\Users\lipen\PycharmProjects\dosusu2sem\task2\Статистика по рейсам (1).xlsx"

    if not os.path.exists(file_path):
        print(f"Ошибка: файл '{file_path}' не найден!")
        print(f"Текущая рабочая папка: {os.getcwd()}")
        print("Пожалуйста, поместите файл 'Статистика по рейсам (1).xlsx' в папку с программой.")
        sys.exit(1)

    try:
        df = pd.read_excel(file_path, sheet_name="Данные")
        print("Файл успешно загружен!")
        print(f"Размер данных: {df.shape[0]} строк, {df.shape[1]} столбцов\n")
        print("Первые 3 строки:")
        print(df.head(3))
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        sys.exit(1)

    # Проверка наличия необходимых столбцов
    required_cols = ["Вес транспорта, т", "Скорость, км/ч", "Расходтоплива, л/100 км"]
    for col in required_cols:
        if col not in df.columns:
            print(f"Ошибка: в файле отсутствует столбец '{col}'")
            sys.exit(1)

    Y = df["Расходтоплива, л/100 км"]  # зависимая переменная
    X_weight = df["Вес транспорта, т"]  # фактор 1
    X_speed = df["Скорость, км/ч"]  # фактор 2

    # ------------------- 2. Парная регрессия (только вес) -------------------
    print("\n" + "=" * 70)
    print("ЧАСТЬ 1. ПАРНАЯ ЛИНЕЙНАЯ РЕГРЕССИЯ: расход топлива ~ вес")
    print("=" * 70)

    X1 = sm.add_constant(X_weight)  # добавляем свободный член
    model_simple = sm.OLS(Y, X1).fit()

    b0_simple = model_simple.params['const']
    b1_simple = model_simple.params['Вес транспорта, т']
    r2_simple = model_simple.rsquared

    print(f"\nУравнение регрессии:")
    print(f"Расход топлива = {b0_simple:.6f} + {b1_simple:.6f} * Вес")
    print(f"\nКоэффициент детерминации R² = {r2_simple:.6f}")

    # Оценка качества
    if r2_simple >= 0.7:
        qual_simple = "хорошее"
    elif r2_simple >= 0.5:
        qual_simple = "умеренное"
    else:
        qual_simple = "слабое"
    print(f"Качество парной модели: {qual_simple} (R² = {r2_simple:.4f})")

    # ------------------- 3. Диаграмма рассеяния + линия регрессии -------------------
    plt.figure(figsize=(10, 6))
    plt.scatter(X_weight, Y, alpha=0.6, color='blue', label='Исходные данные')

    # Построение линии регрессии
    x_range = np.linspace(X_weight.min(), X_weight.max(), 100)
    y_pred_line = b0_simple + b1_simple * x_range
    plt.plot(x_range, y_pred_line, color='red', linewidth=2, label='Линия регрессии')

    plt.xlabel('Вес транспорта, т')
    plt.ylabel('Расход топлива, л/100 км')
    plt.title(f'Парная регрессия: расход топлива ~ вес\nR² = {r2_simple:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('parnaya_regressiya.png', dpi=150)
    plt.show()

    # ------------------- 4. Множественная регрессия (вес + скорость) -------------------
    print("\n" + "=" * 70)
    print("ЧАСТЬ 2. МНОЖЕСТВЕННАЯ ЛИНЕЙНАЯ РЕГРЕССИЯ: расход топлива ~ вес + скорость")
    print("=" * 70)

    X_mult = sm.add_constant(df[["Вес транспорта, т", "Скорость, км/ч"]])
    model_mult = sm.OLS(Y, X_mult).fit()

    b0_mult = model_mult.params['const']
    b1_mult = model_mult.params['Вес транспорта, т']
    b2_mult = model_mult.params['Скорость, км/ч']
    r2_mult = model_mult.rsquared

    print(f"\nУравнение множественной регрессии:")
    print(f"Расход топлива = {b0_mult:.6f} + {b1_mult:.6f} * Вес + {b2_mult:.6f} * Скорость")
    print(f"\nКоэффициент детерминации R² = {r2_mult:.6f}")

    if r2_mult >= 0.7:
        qual_mult = "хорошее"
    elif r2_mult >= 0.5:
        qual_mult = "умеренное"
    else:
        qual_mult = "слабое"
    print(f"Качество множественной модели: {qual_mult} (R² = {r2_mult:.4f})")

    # ------------------- 5. Сравнение моделей -------------------
    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 70)
    print(f"R² (парная, только вес)          = {r2_simple:.6f}")
    print(f"R² (множественная, вес+скорость) = {r2_mult:.6f}")
    print(f"Изменение R²: {r2_mult - r2_simple:+.6f} ({(r2_mult - r2_simple) / r2_simple * 100:+.2f}%)")

    if r2_mult > r2_simple:
        best = "множественная регрессия"
        print("\nДобавление переменной 'Скорость' улучшило качество модели.")
    else:
        best = "парная регрессия"
        print("\nДобавление переменной 'Скорость' не улучшило модель.")
    print(f"Лучшая модель: {best}")

    # ------------------- 6. Полная статистика множественной модели -------------------
    print("\n" + "=" * 70)
    print("ПОДРОБНАЯ СТАТИСТИКА МНОЖЕСТВЕННОЙ МОДЕЛИ (statsmodels)")
    print("=" * 70)
    print(model_mult.summary())

    # ------------------- 7. Итоговые выводы -------------------
    print("\n" + "=" * 70)
    print("ВЫВОДЫ ПО ЗАДАНИЮ")
    print("=" * 70)
    print("1. Построены две модели линейной регрессии (парная и множественная).")
    print(f"2. Парная модель объясняет {r2_simple * 100:.1f}% вариации расхода топлива,")
    print("   что говорит о заметном влиянии веса, но оставляет значительную необъяснённую часть.")
    print(f"3. Множественная модель объясняет {r2_mult * 100:.1f}% вариации,")
    print("   то есть учёт скорости дополнительно повышает качество.")
    print("4. Коэффициент при скорости положительный/отрицательный (нужно посмотреть по факту):")
    print(
        f"   b2 = {b2_mult:.6f} → с ростом скорости расход {'увеличивается' if b2_mult > 0 else 'уменьшается'} (при фиксированном весе).")
    print("5. Таким образом, для прогнозирования расхода топлива целесообразно использовать")
    print("   множественную модель, так как она учитывает два ключевых фактора.")


if __name__ == "__main__":
    main()