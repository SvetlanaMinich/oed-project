import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Загружаем данные экспоненциального распределения
exponential_data = pd.read_csv('datasets/exponential_distribution.csv')['values']

# Преобразуем в pandas Series для удобства
exponential_series = pd.Series(exponential_data)

# Создаем фигуру с двумя графиками
plt.figure(figsize=(15, 10))

# ACF — автокорреляционная функция до лага 50
plt.subplot(2, 1, 1)
plot_acf(exponential_series, lags=50, alpha=0.05)
plt.title('Автокорреляционная функция (ACF) для экспоненциального распределения', fontsize=14)
plt.xlabel('Лаг', fontsize=12)
plt.ylabel('Автокорреляция', fontsize=12)
plt.grid(alpha=0.3)

# PACF — частичная автокорреляционная функция
plt.subplot(2, 1, 2)
plot_pacf(exponential_series, lags=50, method='ywm', alpha=0.05)  # 'ywm' — стабильный метод
plt.title('Частичная автокорреляционная функция (PACF) для экспоненциального распределения', fontsize=14)
plt.xlabel('Лаг', fontsize=12)
plt.ylabel('Частичная автокорреляция', fontsize=12)
plt.grid(alpha=0.3)

plt.show()

# Вычислим несколько значений автокорреляции для проверки
print("Значения автокорреляции для экспоненциального распределения:")
for lag in range(1, 6):
    acf_value = exponential_series.autocorr(lag=lag)
    print(f"Лаг {lag}: {acf_value:.6f}")

# Для хорошего генератора случайных чисел с экспоненциальным распределением 
# все значения автокорреляции (кроме лага 0) должны быть близки к нулю.
# Экспоненциальное распределение часто используется для моделирования 
# времени между событиями, и отсутствие автокорреляции означает, 
# что события происходят независимо друг от друга. 