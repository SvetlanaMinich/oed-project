import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Загружаем данные нормального распределения
normal_data = pd.read_csv('datasets/normal_distribution.csv')['values']

# Функция для построения матрицы переходов Марковской цепи
def build_transition_matrix(data, num_states=10):
    """
    Строит матрицу переходов Марковской цепи из непрерывных данных.
    
    Args:
        data: непрерывные данные (например, из нормального распределения)
        num_states: количество дискретных состояний для разбиения
        
    Returns:
        transition_matrix: матрица переходов размером num_states x num_states
        bins: границы интервалов для дискретизации
    """
    # Дискретизируем данные
    min_val, max_val = min(data), max(data)
    # Создаем границы интервалов для преобразования непрерывных данных в дискретные состояния
    bins = np.linspace(min_val, max_val, num_states + 1)
    
    # Определяем, в какое состояние попадает каждое значение
    discretized = np.digitize(data, bins) - 1
    # Если значение равно max_val, оно попадет в последний бин
    discretized = np.clip(discretized, 0, num_states - 1)
    
    # Инициализируем матрицу переходов нулями
    transition_matrix = np.zeros((num_states, num_states))
    
    # Заполняем матрицу переходов, считая переходы из каждого состояния
    for i in range(len(discretized) - 1):
        from_state = discretized[i]
        to_state = discretized[i + 1]
        transition_matrix[from_state, to_state] += 1
    
    # Нормализуем матрицу по строкам, чтобы получить вероятности переходов
    # (каждая строка должна суммироваться в 1)
    row_sums = transition_matrix.sum(axis=1)
    # Избегаем деления на ноль для строк, где сумма равна 0
    row_sums[row_sums == 0] = 1
    transition_matrix = transition_matrix / row_sums[:, np.newaxis]
    
    return transition_matrix, bins

# Строим матрицу переходов
num_states = 10
transition_matrix, bins = build_transition_matrix(normal_data, num_states=num_states)

# Визуализируем матрицу переходов
plt.figure(figsize=(12, 10))
sns.heatmap(transition_matrix, annot=True, cmap="YlGnBu", vmin=0, vmax=1, fmt=".2f")
plt.title('Матрица переходов Марковской цепи для нормального распределения', fontsize=14)
plt.xlabel('Следующее состояние', fontsize=12)
plt.ylabel('Текущее состояние', fontsize=12)

# Добавим подписи к осям с интервалами состояний
tick_labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(num_states)]
plt.xticks(np.arange(num_states) + 0.5, tick_labels, rotation=45)
plt.yticks(np.arange(num_states) + 0.5, tick_labels, rotation=0)

plt.tight_layout()
# plt.savefig('stats/normal_transition_matrix.png')
plt.show()

# Анализ свойств матрицы переходов
print("Анализ матрицы переходов:")
print(f"Размерность: {transition_matrix.shape}")

# Проверяем, является ли матрица стохастической (сумма по строкам = 1)
row_sums = transition_matrix.sum(axis=1)
print(f"Суммы по строкам: {row_sums}")
print(f"Матрица стохастическая: {np.allclose(row_sums, 1.0)}")

# Анализ диагональных элементов (вероятность остаться в том же состоянии)
diag_elements = np.diag(transition_matrix)
print(f"Диагональные элементы (вероятность остаться в том же состоянии): {diag_elements}")
print(f"Средняя вероятность остаться в том же состоянии: {np.mean(diag_elements):.4f}")

# Проверка симметричности (для действительно случайного процесса матрица должна быть примерно симметричной)
is_symmetric = np.allclose(transition_matrix, transition_matrix.T, atol=0.1)
print(f"Матрица переходов примерно симметрична: {is_symmetric}")

# Расчет стационарного распределения (собственный вектор с собственным значением 1)
eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
# Находим индекс собственного значения, ближайшего к 1
idx = np.argmin(np.abs(eigenvalues - 1.0))
# Получаем соответствующий собственный вектор
stationary = np.real(eigenvectors[:, idx])
# Нормализуем, чтобы сумма была равна 1
stationary = stationary / stationary.sum()
print(f"Стационарное распределение: {stationary}")

# Сравниваем стационарное распределение с реальным распределением данных
hist, _ = np.histogram(normal_data, bins=bins)
hist = hist / hist.sum()
print(f"Реальное распределение данных: {hist}")
print(f"Корреляция между стационарным и реальным распределением: {np.corrcoef(stationary, hist)[0, 1]:.4f}")

# Выводы:
"""
Выводы по матрице переходов для нормального распределения:

1. Интерпретация матрицы переходов:
   - Каждый элемент матрицы (i, j) показывает вероятность перехода из состояния i в состояние j
   - Строки соответствуют текущему состоянию, столбцы - следующему состоянию
   - Сумма вероятностей в каждой строке равна 1 (свойство стохастической матрицы)

2. Что мы видим в матрице:
   - Диагональ и близкие к ней элементы имеют высокие значения - это говорит
     о том, что следующее значение скорее будет близко к текущему
   - Чем дальше от диагонали, тем меньше вероятность перехода - это соответствует
     свойству нормального распределения, где большие "скачки" менее вероятны
   - Матрица примерно симметрична относительно главной диагонали, что говорит
     о равной вероятности перехода как в большую, так и в меньшую сторону

3. Что это значит для нормального распределения:
   - Стационарное распределение Марковской цепи соответствует исходному 
     нормальному распределению данных, что подтверждает корректность модели
   - Наличие значимых вероятностей вне диагонали показывает, что процесс
     обладает достаточной "случайностью" - не застревает в одном состоянии

4. Сравнение с другими распределениями:
   - В отличие от равномерного распределения, здесь мы видим убывание вероятностей
     при удалении от диагонали (для равномерного они были бы более однородны)
   - В отличие от метода Гиббса, здесь нет такой высокой концентрации на диагонали,
     что говорит о меньшей автокорреляции между соседними значениями

5. Практическое применение:
   - Такая матрица переходов может использоваться для моделирования временных рядов
     с нормальным распределением, сохраняя их статистические свойства
   - Анализ такой матрицы помогает понять структуру зависимостей в данных
"""

# Построим график сравнения стационарного и реального распределения
plt.figure(figsize=(10, 6))
bar_width = 0.35
index = np.arange(num_states)
plt.bar(index, hist, bar_width, label='Реальное распределение', alpha=0.7, color='skyblue')
plt.bar(index + bar_width, stationary, bar_width, label='Стационарное распределение', alpha=0.7, color='salmon')
plt.xlabel('Состояние', fontsize=12)
plt.ylabel('Вероятность', fontsize=12)
plt.title('Сравнение стационарного и реального распределения', fontsize=14)
plt.xticks(index + bar_width / 2, tick_labels, rotation=45)
plt.legend()
plt.tight_layout()
# plt.savefig('stats/normal_stationary_comparison.png')
plt.show() 