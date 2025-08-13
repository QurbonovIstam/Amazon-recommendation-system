# 🛒 Гибридная рекомендательная система для отзывов на товары Amazon  

## 🔗 Google Colab Notebook
[Open in Colab](https://colab.research.google.com/drive/1uny-9RoI_jlYNkgP2TemxGQ0rcAtEdSQ?usp=drive_link)

**Hybrid Recommendation System for Amazon Product Reviews**

---

## 🇷🇺 О проекте

### 🎯 Цель
Разработать гибридную рекомендательную систему для предсказания следующего товара, который может заинтересовать пользователя, на основе его истории оценок и отзывов.

### 📂 Датасет
- Источник: [Kaggle — Amazon Product Reviews](https://www.kaggle.com/datasets)  
- Количество записей: **568,454**  
- Количество признаков: **10**  
- Основные поля: `Id`, `ProductId`, `UserId`, `ProfileName`, `HelpfulnessNumerator`, `HelpfulnessDenominator`, `Score`, `Time`, `Summary`, `Text`  

### 🔍 Предварительная обработка
- Удалены записи с пропущенными значениями.  
- Отобраны только пользователи с ≥5 отзывами.  
- Данные разделены на `train` и `test` по принципу **Leave-One-Out** — последняя покупка в тест, остальные в обучение.

---

## 🧠 Построенные модели

### 1️⃣ **Базовая модель** — Коллаборативная фильтрация (ALS)
- Метод: Alternating Least Squares (из библиотеки `implicit`)  
- Гиперпараметры: `factors=100`, `regularization=0.01`, `iterations=25`  
- Результат: **Hit@10 = 0.4542**  

### 2️⃣ **Гибридная модель (Summary)** — ALS + TF-IDF на кратких описаниях
- Логика: Переранжирование топ-20 кандидатов по косинусному сходству с усреднённым вектором TF-IDF на `Summary`.  
- Результат: **Hit@10 = 0.2408** (ухудшение, краткие описания слишком шумные).  

### 3️⃣ **Гибридная модель (Full Text)** — ALS + TF-IDF на полных текстах отзывов
- Логика: Аналогично предыдущей, но с использованием `Text` (полных отзывов).  
- Результат: **Hit@10 = 0.4667** (улучшение по сравнению с чистым ALS).  

---

## 📊 Итоговые результаты

| Модель                | Описание                                     | Hit@10  |
|-----------------------|----------------------------------------------|---------|
| **Original ALS**      | Коллаборативная фильтрация                   | 0.4542  |
| **Hybrid (Summary)**  | ALS + TF-IDF на кратких описаниях             | 0.2408  |
| **Hybrid (Full Text)**| ALS + TF-IDF на полных текстах                | 0.4667  |

---

## 📈 Выводы
- Чистая коллаборативная фильтрация — сильная базовая модель.
- Полные тексты отзывов несут полезный сигнал и способны улучшить качество рекомендаций.
- Краткие описания (`Summary`) слишком зашумлены.

---

## 🚀 Возможные дальнейшие шаги
- Использовать **BERT / RoBERTa** для генерации семантических эмбеддингов.
- Экспериментировать с размером топа кандидатов (топ-50, топ-100).
- Настроить веса при комбинировании оценок ALS и контентной модели.

---

## 🔗 Google Colab Notebook
[Открыть проект в Colab](https://colab.research.google.com/drive/1uny-9RoI_jlYNkgP2TemxGQ0rcAtEdSQ?usp=drive_link)

---

## 🇬🇧 English Version

### 🎯 Goal
To build a hybrid recommendation system for predicting the next product a user might purchase based on their review history and product ratings.

### 📂 Dataset
- Source: [Kaggle — Amazon Product Reviews](https://www.kaggle.com/datasets)  
- Records: **568,454**  
- Features: **10**  
- Key columns: `Id`, `ProductId`, `UserId`, `ProfileName`, `HelpfulnessNumerator`, `HelpfulnessDenominator`, `Score`, `Time`, `Summary`, `Text`

### 🔍 Preprocessing
- Removed missing values.  
- Filtered users with ≥5 reviews.  
- Train/test split using **Leave-One-Out** — last purchase in test set.

---

## 🧠 Models

### 1️⃣ **Baseline** — Collaborative Filtering (ALS)
- Method: Alternating Least Squares (`implicit` library)  
- Hyperparameters: `factors=100`, `regularization=0.01`, `iterations=25`  
- Result: **Hit@10 = 0.4542**

### 2️⃣ **Hybrid (Summary)** — ALS + TF-IDF on product summaries
- Reranked top-20 ALS candidates using cosine similarity with average TF-IDF vector from summaries.  
- Result: **Hit@10 = 0.2408** (worse due to noisy summaries).

### 3️⃣ **Hybrid (Full Text)** — ALS + TF-IDF on full review texts
- Same logic, but using TF-IDF vectors from `Text`.  
- Result: **Hit@10 = 0.4667** (improved over ALS baseline).

---

## 📊 Final Results

| Model                 | Description                                  | Hit@10  |
|-----------------------|----------------------------------------------|---------|
| **Original ALS**      | Collaborative Filtering                      | 0.4542  |
| **Hybrid (Summary)**  | ALS + TF-IDF on summaries                     | 0.2408  |
| **Hybrid (Full Text)**| ALS + TF-IDF on full review texts             | 0.4667  |

---

## 📈 Conclusions
- Baseline ALS is strong, but hybrid models can outperform it.
- Full review texts provide valuable signal for reranking.
- Summaries are too noisy to be useful.

---

## 🚀 Future Work
- Use **BERT / RoBERTa** for semantic embeddings.
- Experiment with candidate list size (top-50, top-100).
- Adjust blending weights between ALS and content-based scores.

---


