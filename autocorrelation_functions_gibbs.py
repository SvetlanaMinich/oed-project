import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Загружаем данные метода Гиббса
gibbs_data = pd.read_csv('datasets/gibbs_sampler.csv')

# Преобразуем в pandas Series для удобства
gibbs_x_series = pd.Series(gibbs_data['x'])
gibbs_y_series = pd.Series(gibbs_data['y'])

# Создаем фигуру с четырьмя графиками (ACF и PACF для X и Y)
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# ACF для X
plot_acf(gibbs_x_series, lags=50, alpha=0.05, ax=axes[0, 0])
axes[0, 0].set_title('ACF для X (метод Гиббса)', fontsize=14)
axes[0, 0].set_xlabel('Лаг', fontsize=12)
axes[0, 0].set_ylabel('Автокорреляция', fontsize=12)
axes[0, 0].grid(alpha=0.3)

# PACF для X
plot_pacf(gibbs_x_series, lags=50, method='ywm', alpha=0.05, ax=axes[0, 1])
axes[0, 1].set_title('PACF для X (метод Гиббса)', fontsize=14)
axes[0, 1].set_xlabel('Лаг', fontsize=12)
axes[0, 1].set_ylabel('Частичная автокорреляция', fontsize=12)
axes[0, 1].grid(alpha=0.3)

# ACF для Y
plot_acf(gibbs_y_series, lags=50, alpha=0.05, ax=axes[1, 0])
axes[1, 0].set_title('ACF для Y (метод Гиббса)', fontsize=14)
axes[1, 0].set_xlabel('Лаг', fontsize=12)
axes[1, 0].set_ylabel('Автокорреляция', fontsize=12)
axes[1, 0].grid(alpha=0.3)

# PACF для Y
plot_pacf(gibbs_y_series, lags=50, method='ywm', alpha=0.05, ax=axes[1, 1])
axes[1, 1].set_title('PACF для Y (метод Гиббса)', fontsize=14)
axes[1, 1].set_xlabel('Лаг', fontsize=12)
axes[1, 1].set_ylabel('Частичная автокорреляция', fontsize=12)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('stats/gibbs_acf_pacf.png')
plt.show()

# Вычислим несколько значений автокорреляции для проверки
print("Значения автокорреляции для X (метод Гиббса):")
for lag in range(1, 6):
    acf_value = gibbs_x_series.autocorr(lag=lag)
    print(f"Лаг {lag}: {acf_value:.6f}")

print("\nЗначения автокорреляции для Y (метод Гиббса):")
for lag in range(1, 6):
    acf_value = gibbs_y_series.autocorr(lag=lag)
    print(f"Лаг {lag}: {acf_value:.6f}")

# Отдельные графики для визуализации автокорреляции времянки
plt.figure(figsize=(10, 6))
plt.plot(range(1, 26), [gibbs_x_series.autocorr(lag=i) for i in range(1, 26)], 'bo-', label='X')
plt.plot(range(1, 26), [gibbs_y_series.autocorr(lag=i) for i in range(1, 26)], 'ro-', label='Y')
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
plt.title('Автокорреляция X и Y для метода Гиббса', fontsize=14)
plt.xlabel('Лаг', fontsize=12)
plt.ylabel('Автокорреляция', fontsize=12)
plt.grid(alpha=0.3)
plt.legend()
plt.savefig('stats/gibbs_autocorr_plot.png')
plt.show()

# Для метода Гиббса характерна высокая автокорреляция, особенно для 
# малых лагов. Это является следствием того, как работает алгоритм:
# каждое новое значение генерируется на основе предыдущего, что создает
# зависимость между последовательными значениями.
# 
# В отличие от других методов, для хорошей работы метода Гиббса
# не требуется отсутствие автокорреляции. Напротив, наличие автокорреляции
# ожидаемо и является частью алгоритма. Однако для получения независимых 
# выборок из совместного распределения, сгенерированного методом Гиббса,
# рекомендуется делать "прореживание" (thinning) - брать значения
# через определенные интервалы. 



# Метод Гиббса генерирует новые значения на основе предыдущих, 
# создавая очень сильную зависимость между соседними значениями


# Практический вывод:
# Для получения по-настоящему независимых выборок из распределения, 
# сгенерированного методом Гиббса, рекомендуется использовать "прореживание" 
# (thinning) - брать не все значения подряд, а например, каждое 50-е или 100-е значение. 
# Это снизит автокорреляцию между выбранными значениями.