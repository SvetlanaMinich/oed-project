import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Энтропия - это мера неопределенности или случайности в данных.

# Загружаем данные из CSV
uniform_data = pd.read_csv('datasets/uniform_distribution.csv')['values']

bins = 100
hist, bin_edges = np.histogram(uniform_data, bins=bins)
# Нормализуем, чтобы получить вероятности
probabilities = hist / len(uniform_data)
# Убираем нулевые вероятности (log2(0) не определен)
probabilities = probabilities[probabilities > 0]

# Считаем энтропию через классическое определение энтропии Шеннона:
# Используем log₂, поэтому единица измерения - биты
# 1 бит энтропии = неопределенность одного честного броска монеты
uniform_entropy = -np.sum(probabilities * np.log2(probabilities))

print(f'Энтропия равномерного распределения: {uniform_entropy}')

# Визуализация энтропии
plt.figure(figsize=(10, 6))
plt.hist(uniform_data, bins=bins, color='orange', edgecolor='black')
plt.title(f'Гистограмма равномерного распределения\nЭнтропия: {uniform_entropy:.4f} бит')
plt.xlabel('Значение')
plt.ylabel('Частота')
plt.grid(alpha=0.3)
plt.savefig('entropy/uniform_distribution_entropy.png')
plt.show()

# Теоретически, для идеального равномерного распределения на отрезке [a, b],
# Энтропия вычисляется как log2(b - a)
# Для промежутка от 1 до 100 теоретическая энтропия = log2(99) ≈ 6.63 бит
# Разница между теоретической и вычисленной энтропией связана с дискретизацией
# и конечным размером выборки 