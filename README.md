
# 🚀 **Теория и практика ИИ и МО**
## 🔥 Полное портфолио практических заданий

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=28&duration=3000&pause=500&color=00FFAA&center=true&vCenter=true&width=800&lines=Липенков+Александр+ПИШ-212;Южно-Уральский+государственный+университет;Весна+2026" alt="Typing SVG" />
</p>

<p align="center">
  <a href="https://github.com/Kango911/dosusu2sem"><img src="https://img.shields.io/badge/GitHub-Repository-181717?logo=github&style=for-the-badge" /></a>
  <a href="https://t.me/Kango911"><img src="https://img.shields.io/badge/Telegram-Kango911-26A5E4?logo=telegram&style=for-the-badge" /></a>
  <a href="mailto:lipenkov.a61@gmail.com"><img src="https://img.shields.io/badge/Email-lipenkov.a61@gmail.com-D14836?logo=gmail&style=for-the-badge" /></a>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

---

## 📑 **Оглавление**

1. [📁 Структура проекта](#-структура-проекта)
2. [⚙️ Установка и запуск](#️-установка-и-запуск)
3. [📊 Список выполненных заданий](#-список-выполненных-заданий)
4. [📈 Ключевые результаты](#-ключевые-результаты)
5. [📦 Источники данных](#-источники-данных)
6. [🤝 Контакты](#-контакты)

---

## 📁 **Структура проекта**

```
dosusu2sem/
│
├── task1/                              # Парная линейная регрессия
│   ├── main1.py
│   └── Статистика по рейсам (1).xlsx
│
├── task2/                              # Множественная регрессия
│   ├── main2.py
│   └── Статистика по рейсам (1).xlsx
│
├── task3/                              # Полиномиальная регрессия
│   ├── main3.py
│   ├── Полином_ЛР.xlsx
│   └── *.png
│
├── task4/                              # Временные ряды
│   ├── main4.py
│   ├── passengers.csv
│   └── *.png
│
├── block2/task1/                       # Парная регрессия (Driving Behavior)
│   ├── main1.py
│   ├── train_motion_data.csv
│   ├── test_motion_data.csv
│   ├── parnaya_regressiya_block2.png
│   └── Описание полей датасета.pdf
│
├── block3/task1/                       # Большие данные (NYC Taxi)
│   ├── main1.py
│   ├── yellow_tripdata_2016-09.parquet
│   └── Практика - Большие данные.pdf
│
├── block4/                             # Работа с ИИ (Qwen)
│   ├── task1/                          # Стикерпаки
│   ├── task2/                          # Файлы + регрессия
│   │   ├── main2.py
│   │   ├── Практическое задание_файлы.docx
│   │   └── Статистика по рейсам (1).xlsx
│   └── task3/                          # ИКР
│       ├── main3.py
│       └── Практическое задание_ИКР.docx
│
├── block5/task1/                       # Анализ потоков AIMS eco
│   ├── task1.xlsx
│   ├── Отчет.Ленина-Энгельса-16-2-2026-Количество ТС.xlsx
│   ├── Отчет.Ленина-Энгельса-17-2-2026-Количество ТС.xlsx
│   ├── Отчет.Ленина-Энгельса-18-2-2026-Количество ТС.xlsx
│   ├── Отчет.Ленина-Энгельса-19-2-2026-Количество ТС.xlsx
│   ├── Отчет.Ленина-Энгельса-20-2-2026-Количество ТС.xlsx
│   ├── Отчет.Ленина-Энгельса-21-2-2026-Количество ТС.xlsx
│   └── Отчет.Ленина-Энгельса-22-2-2026-Количество ТС.xlsx
│
├── block5/task2/                       # Анализ потоков AIMS eco
│   ├── task2.xlsx
│   ├── Отчет.Ленина-Энгельса-16-2-2026-Количество ТС.xlsx
│   ├── Отчет.Ленина-Энгельса-17-2-2026-Количество ТС.xlsx
│   ├── Отчет.Ленина-Энгельса-18-2-2026-Количество ТС.xlsx
│   ├── Отчет.Ленина-Энгельса-19-2-2026-Количество ТС.xlsx
│   ├── Отчет.Ленина-Энгельса-20-2-2026-Количество ТС.xlsx
│   ├── Отчет.Ленина-Энгельса-21-2-2026-Количество ТС.xlsx
│   └── Отчет.Ленина-Энгельса-22-2-2026-Количество ТС.xlsx
│
├── block6/task1/                       # Математическая оптимизация
│   ├── main1.py
│   └── opt_homework.pdf
│
├── block7/                             # Тесты и бонус
│   ├── task1/                          # Тесты
│   │   ├── main1.py
│   │   ├── test1.docx
│   │   └── Тест 2.docx
│   └── task2/                          # Кейс 3 (бонус)
│       ├── main2.py
│       └── Практическое задание Кейса 3.docx
│
└── README.md
```

> 💡 **Примечание:** Parquet-файл (NYC Taxi) не включён в репозиторий из-за большого размера (~300 МБ). Инструкция по скачиванию – ниже.

---

## ⚙️ **Установка и запуск**

### 1️⃣ Клонирование
```bash
git clone https://github.com/Kango911/dosusu2sem.git
cd dosusu2sem
```

### 2️⃣ Виртуальное окружение
```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows
```

### 3️⃣ Установка зависимостей
```bash
pip install --upgrade pip
pip install pandas numpy matplotlib statsmodels scikit-learn
pip install dask dask-ml openpyxl xlsxwriter
pip install pyomo glpk
pip install tensorflow keras   # опционально (LSTM)
```

### 4️⃣ Скачивание данных

| Файл | Источник | Место |
|------|----------|-------|
| `yellow_tripdata_2016-09.parquet` | [NYC TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) – Yellow Taxi, сентябрь 2016, PARQUET | `block3/task1/` |
| `Статистика по рейсам (1).xlsx` | Внутренний датасет курса | `task1/`, `task2/`, `block4/task2/` |
| `Полином_ЛР.xlsx` | Внутренний датасет курса | `task3/` |
| `train_motion_data.csv`, `test_motion_data.csv` | Driving Behavior dataset | `block2/task1/` |
| Отчёты AIMS eco | [Демо-портал](https://aims.susu.ru/demo/) | `block5/task1/` |

### 5️⃣ Запуск примера
```bash
cd task1
python main1.py
```
Все графики сохраняются как PNG в текущей папке.

---

## 📊 **Список выполненных заданий**

| № | Название | Файл | Библиотеки | Краткий результат |
|---|----------|------|------------|-------------------|
| 1 | Парная регрессия (топливо → вес) | `task1/main1.py` | pandas, statsmodels | R² = 0.48, уравнение: Расход = 12.67 + 0.41·Вес |
| 2 | Множественная регрессия (+ скорость) | `task2/main2.py` | pandas, statsmodels, matplotlib | R² = 0.54, скорость значима |
| 3 | Полиномиальная регрессия | `task3/main3.py` | sklearn, numpy, matplotlib | Лучшая степень – 3 (R² ≈ 0.99) |
| 4 | Временные ряды (пассажиры) | `task4/main4.py` | statsmodels, pandas | Прогноз на январь 1961: ~452 тыс. |
| 5 | Парная регрессия (AccX→AccY) | `block2/task1/main1.py` | sklearn, matplotlib | R² ≈ 0.17 (связь слабая) |
| 6 | Анализ такси NYC (Dask) | `block3/task1/main1.py` | dask, dask-ml | Пик в 19:00 (624 тыс.), R² = 0.74 |
| 7 | Работа с ИИ (Qwen) – файлы | `block4/task2/main2.py` | Qwen, pandas | Ответы на вопросы по Excel, регрессия |
| 8 | Анализ потоков AIMS eco | `block5/task1/task1.xlsx` | Excel | Пик 17–18ч (>1700 ТС), корреляции |
| 9 | Оптимизация (Pyomo) | `block6/task1/main1.py` | pyomo, glpk | Мин. стоимость = 184 500 руб. |
| 10 | Тест по трактору ТМ-10 | `block7/task1/main1.py` | – | 10/10 правильных ответов |
| 11 | Бонусный кейс 3 | `block7/task2/main2.py` | Qwen, Python | План курсовой, фрагмент LSTM, литература |

---

## 📈 **Ключевые результаты**

### 🔹 Регрессионный анализ
- **Вес** объясняет ~48% вариации расхода топлива
- Добавление **скорости** → R² = 0.54
- Полином 3-й степени → R² ≈ 0.99

### 🔹 Временные ряды
- Тренд: `Пассажиры = 112.78 + 2.34·t`
- Прогноз на январь 1961: **452 тыс. пассажиров**

### 🔹 Driving Behavior
- Корреляция `AccX` и `AccY` слабая (R² ≈ 0.17) – физически обосновано

### 🔹 NYC Taxi (сентябрь 2016)
- **10 млн+ поездок** обработано Dask
- Пик: **19:00** (624 тыс.), самые дорогие поездки – **в 5 утра** ($20.93)
- Модель стоимости: `total_amount = 2.13 + 1.87·distance + 0.08·hour` (R² = 0.74)

### 🔹 Транспортная задача (Pyomo)
- Оптимальное распределение 650 тонн
- Минимальная стоимость: **184 500 руб.**

### 🔹 Анализ перекрёстка (AIMS eco)
- **Рабочие дни:** пик 17–18ч (>1700 ТС)
- **Выходные:** пик 14–15ч (~1000 ТС)
- Корреляции: `E–W = 0.83`, `E–N = –0.65`

### 🔹 Бонусный кейс 3
- 10 тем для курсовых работ
- План по диагностике гидрораспределителя
- Фрагмент кода LSTM + литература (ГОСТ)

---

## 📦 **Источники данных**

1. **Статистика по рейсам.xlsx** – внутренний датасет курса
2. **Полином_ЛР.xlsx** – синтетические нелинейные данные
3. **passengers.csv** – классический датасет Box & Jenkins
4. **train_motion_data.csv / test_motion_data.csv** – Driving Behavior (классификация стиля вождения)
5. **Yellow Taxi Trip Records** – [NYC TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
6. **Отчёты AIMS eco** – демо-портал Челябинска
7. **Учебное пособие** – Павловская О.О., Кондаков С.В. «Бортовые системы управления бульдозера на базе трактора ТМ-10»

---

## 🤝 **Контакты**

| Связь | Данные |
|-------|--------|
| **GitHub** | [https://github.com/Kango911](https://github.com/Kango911) |
| **Telegram** | [@Kango911](https://t.me/Kango911) |
| **Email** | [lipenkov.a61@gmail.com](mailto:lipenkov.a61@gmail.com) |

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:00FFAA,100:0066FF&height=120&section=footer" />
</p>

**© 2026 Липенков Александр, ПИШ-212**  
*Все решения могут быть использованы в образовательных целях*
```