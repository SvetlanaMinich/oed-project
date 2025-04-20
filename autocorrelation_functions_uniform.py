import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Загружаем данные равномерного распределения
uniform_data = pd.read_csv('datasets/uniform_distribution.csv')['values']

# Преобразуем в pandas Series для удобства
uniform_series = pd.Series(uniform_data)

# Создаем фигуру с двумя графиками
plt.figure(figsize=(15, 10))

# ACF — автокорреляционная функция до лага 50
plt.subplot(2, 1, 1)
plot_acf(uniform_series, lags=50, alpha=0.05)
plt.title('Автокорреляционная функция (ACF) для равномерного распределения', fontsize=14)
plt.xlabel('Лаг', fontsize=12)
plt.ylabel('Автокорреляция', fontsize=12)
plt.grid(alpha=0.3)

# PACF — частичная автокорреляционная функция
plt.subplot(2, 1, 2)
plot_pacf(uniform_series, lags=50, method='ywm', alpha=0.05)  # 'ywm' — стабильный метод
plt.title('Частичная автокорреляционная функция (PACF) для равномерного распределения', fontsize=14)
plt.xlabel('Лаг', fontsize=12)
plt.ylabel('Частичная автокорреляция', fontsize=12)
plt.grid(alpha=0.3)

plt.show()

# Вычислим несколько значений автокорреляции для проверки
print("Значения автокорреляции для равномерного распределения:")
for lag in range(1, 6):
    acf_value = uniform_series.autocorr(lag=lag)
    print(f"Лаг {lag}: {acf_value:.6f}")

# Для хорошего генератора случайных чисел с равномерным распределением
# все значения автокорреляции (кроме лага 0) должны быть близки к нулю,
# как и в случае с нормальным распределением 