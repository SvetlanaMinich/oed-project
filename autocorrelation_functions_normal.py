import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Загружаем данные нормального распределения
normal_data = pd.read_csv('datasets/normal_distribution.csv')['values']

# Преобразуем в pandas Series для удобства
normal_series = pd.Series(normal_data)

# Создаем фигуру с двумя графиками
plt.figure(figsize=(15, 10))

# ACF — автокорреляционная функция до лага 50
plt.subplot(2, 1, 1)
plot_acf(normal_series, lags=50, alpha=0.05)
plt.title('Автокорреляционная функция (ACF) для нормального распределения', fontsize=14)
plt.xlabel('Лаг', fontsize=12)
plt.ylabel('Автокорреляция', fontsize=12)
plt.grid(alpha=0.3)

# PACF — частичная автокорреляционная функция
plt.subplot(2, 1, 2)
plot_pacf(normal_series, lags=50, method='ywm', alpha=0.05)  # 'ywm' — стабильный метод
plt.title('Частичная автокорреляционная функция (PACF) для нормального распределения', fontsize=14)
plt.xlabel('Лаг', fontsize=12)
plt.ylabel('Частичная автокорреляция', fontsize=12)
plt.grid(alpha=0.3)

# plt.tight_layout()
plt.show()

# Дополнительно: вычислим несколько значений автокорреляции вручную для проверки
print("Значения автокорреляции:")
for lag in range(1, 6):
    acf_value = normal_series.autocorr(lag=lag)
    print(f"Лаг {lag}: {acf_value:.6f}")

# Пояснение различий между ACF и PACF:
"""
ACF (автокорреляционная функция):
- Показывает корреляцию между временным рядом и его лагами
- Учитывает как прямые, так и косвенные зависимости
- Для истинно случайных данных все значения ACF должны быть близки к нулю
- Голубые границы показывают доверительный интервал (обычно 95%)

PACF (частичная автокорреляционная функция):
- Показывает корреляцию между временным рядом и его лагами, 
  НО с исключением влияния промежуточных лагов
- Измеряет только прямые зависимости
- Позволяет выявить непосредственное влияние конкретного лага

Для истинно случайных данных (как в идеальном нормальном распределении) 
обе функции должны показывать значения близкие к нулю для всех лагов,
кроме нулевого (который всегда равен 1).
""" 