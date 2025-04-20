import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Энтропия - это мера неопределенности или случайности в данных.


normal_data = pd.read_csv('datasets/normal_distribution.csv')['values']

bins = 100
hist, bin_edges = np.histogram(normal_data, bins=bins)
# Нормализуем, чтобы получить вероятности
probabilities = hist / len(normal_data)
# Убираем нулевые вероятности (log2(0) не определен)
probabilities = probabilities[probabilities > 0]

# Считаем энтропию через классическое определение энтропии Шеннона:
# Используем log₂, поэтому единица измерения - биты
# 1 бит энтропии = неопределенность одного честного броска монеты
normal_entropy = -np.sum(probabilities * np.log2(probabilities))

print(f'Энтропия нормального распределения: {normal_entropy}')

# Визуализация энтропии
plt.figure(figsize=(10, 6))
plt.hist(normal_data, bins=bins, color='lightgreen', edgecolor='black')
plt.title(f'Гистограмма нормального распределения\nЭнтропия: {normal_entropy:.4f} бит')
plt.xlabel('Значение')
plt.ylabel('Частота')
plt.grid(alpha=0.3)
plt.savefig('entropy/normal_distribution_entropy.png')
plt.show()