# K9: Домашнее задание "Математическая оптимизация"
# K9: Студент: Липенков Александр ПИШ-212
# K9: Задача 1: Транспортная задача
# K9: Реализация в Pyomo с солвером GLPK (или CBC)
# K9: Среда: PyCharm Professional, Python 3.12

import pyomo.environ as pyo

# ------------------- Данные задачи -------------------
# Запасы на складах (тонн)
supply = {"Склад 1": 200, "Склад 2": 150, "Склад 3": 300}

# Спрос магазинов (тонн)
demand = {"Магазин 1": 100, "Магазин 2": 200, "Магазин 3": 120, "Магазин 4": 230}

# Матрица стоимости перевозки (руб/тонна)
cost = {
    ("Склад 1", "Магазин 1"): 300,
    ("Склад 1", "Магазин 2"): 500,
    ("Склад 1", "Магазин 3"): 400,
    ("Склад 1", "Магазин 4"): 600,
    ("Склад 2", "Магазин 1"): 400,
    ("Склад 2", "Магазин 2"): 350,
    ("Склад 2", "Магазин 3"): 450,
    ("Склад 2", "Магазин 4"): 500,
    ("Склад 3", "Магазин 1"): 500,
    ("Склад 3", "Магазин 2"): 400,
    ("Склад 3", "Магазин 3"): 350,
    ("Склад 3", "Магазин 4"): 450,
}

# Списки узлов
supply_nodes = list(supply.keys())
demand_nodes = list(demand.keys())

# Проверка баланса
total_supply = sum(supply.values())
total_demand = sum(demand.values())
print("=" * 60)
print("ЗАДАЧА 1: ТРАНСПОРТНАЯ ЗАДАЧА")
print("=" * 60)
print(f"Суммарный запас: {total_supply} тонн")
print(f"Суммарный спрос: {total_demand} тонн")
if total_supply == total_demand:
    print("Баланс соблюден – задача закрытая.")
else:
    print("Внимание: дисбаланс! Необходимо добавить фиктивный склад/магазин.")
    # В данном случае баланс идеальный, можно продолжать

# ------------------- Построение модели -------------------
model = pyo.ConcreteModel()

# Индексы
model.I = pyo.Set(initialize=supply_nodes)
model.J = pyo.Set(initialize=demand_nodes)

# Переменные: количество перевозок от i к j (неотрицательные)
model.x = pyo.Var(model.I, model.J, domain=pyo.NonNegativeReals)

# Целевая функция: минимизация суммарных затрат
def obj_rule(m):
    return sum(cost[i, j] * m.x[i, j] for i in m.I for j in m.J)
model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

# Ограничения: вывоз со склада i не может превышать запас
def supply_rule(m, i):
    return sum(m.x[i, j] for j in m.J) <= supply[i]
model.supply_constraint = pyo.Constraint(model.I, rule=supply_rule)

# Ограничения: спрос магазина j должен быть удовлетворен точно
def demand_rule(m, j):
    return sum(m.x[i, j] for i in m.I) == demand[j]
model.demand_constraint = pyo.Constraint(model.J, rule=demand_rule)

# ------------------- Решение -------------------
# Выбираем солвер: glpk (кроссплатформенный) или cbc
solver = pyo.SolverFactory('glpk')
# Если glpk не установлен, попробуйте cbc:
# solver = pyo.SolverFactory('cbc')

result = solver.solve(model, tee=True)  # tee=True для вывода лога солвера

# Проверка статуса решения
if result.solver.termination_condition == pyo.TerminationCondition.optimal:
    print("\nРешение найдено (оптимальное).")
else:
    print(f"\nСтатус решения: {result.solver.termination_condition}")

# ------------------- Вывод результатов -------------------
print("\nОПТИМАЛЬНЫЙ ПЛАН ПЕРЕВОЗОК (тонн):")
for i in model.I:
    for j in model.J:
        val = pyo.value(model.x[i, j])
        if val > 1e-6:  # выводим только ненулевые перевозки
            print(f"  {i} -> {j}: {val:.2f} т, стоимость {cost[i, j]} руб/т -> {val * cost[i, j]:.2f} руб")

total_cost = pyo.value(model.obj)
print(f"\nСУММАРНАЯ СТОИМОСТЬ ПЕРЕВОЗОК: {total_cost:.2f} руб.")

# Проверка выполнения ограничений (отгрузка со складов)
print("\nПроверка отгрузки со складов (должна равняться запасу):")
for i in model.I:
    shipped = sum(pyo.value(model.x[i, j]) for j in model.J)
    print(f"  {i}: отгружено {shipped:.2f} т, запас {supply[i]} т")

print("\nПроверка поставок в магазины (должна равняться спросу):")
for j in model.J:
    received = sum(pyo.value(model.x[i, j]) for i in model.I)
    print(f"  {j}: получено {received:.2f} т, спрос {demand[j]} т")

# ------------------- Ссылка на репозиторий -------------------
print("\n" + "=" * 60)
print("ССЫЛКА НА РЕПОЗИТОРИЙ")
print("=" * 60)
print("GitHub: https://github.com/Kango911/dosusu2sem")