import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Энтропия - это мера неопределенности или случайности в данных.

# Загружаем данные из CSV
exponential_data = pd.read_csv('datasets/exponential_distribution.csv')['values']

bins = 100
hist, bin_edges = np.histogram(exponential_data, bins=bins)
# Нормализуем, чтобы получить вероятности
probabilities = hist / len(exponential_data)
# Убираем нулевые вероятности (log2(0) не определен)
probabilities = probabilities[probabilities > 0]

# Считаем энтропию через классическое определение энтропии Шеннона
exponential_entropy = -np.sum(probabilities * np.log2(probabilities))

print(f'Энтропия экспоненциального распределения: {exponential_entropy}')

# Визуализация энтропии
plt.figure(figsize=(10, 6))
plt.hist(exponential_data, bins=bins, color='salmon', edgecolor='black')
plt.title(f'Гистограмма экспоненциального распределения\nЭнтропия: {exponential_entropy:.4f} бит')
plt.xlabel('Значение')
plt.ylabel('Частота')
plt.grid(alpha=0.3)
plt.savefig('entropy/exponential_distribution_entropy.png')
plt.show()

# Интересный факт: для экспоненциального распределения с параметром λ
# теоретическая энтропия равна 1 - ln(λ) + γ бит, 
# где γ ≈ 0.57721 - постоянная Эйлера-Маскерони
# Для λ = 1 энтропия должна быть примерно 1.577 бит 