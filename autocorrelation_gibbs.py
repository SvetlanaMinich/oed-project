import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# Загружаем данные из CSV
gibbs_data = pd.read_csv('datasets/gibbs_sampler.csv')


# -------------------------------- АНАЛИЗ X КООРДИНАТЫ --------------------------------
# Рассчитываем основные статистические показатели для X
mean_x = np.mean(gibbs_data['x'])
variance_x = np.var(gibbs_data['x'])
std_dev_x = np.std(gibbs_data['x'])
autocorrelation_x = pd.Series(gibbs_data['x']).autocorr(lag=1)

print("--- Статистические показатели для X ---")
print(f'Среднее X: {mean_x:.6f}')
print(f'Дисперсия X: {variance_x:.6f}')
print(f'Стандартное отклонение X: {std_dev_x:.6f}')
print(f'Автокорреляция X (лаг 1): {autocorrelation_x:.6f}')


# -------------------------------- АНАЛИЗ Y КООРДИНАТЫ --------------------------------
# Рассчитываем основные статистические показатели для Y
mean_y = np.mean(gibbs_data['y'])
variance_y = np.var(gibbs_data['y'])
std_dev_y = np.std(gibbs_data['y'])
autocorrelation_y = pd.Series(gibbs_data['y']).autocorr(lag=1)

print("\n--- Статистические показатели для Y ---")
print(f'Среднее Y: {mean_y:.6f}')
print(f'Дисперсия Y: {variance_y:.6f}')
print(f'Стандартное отклонение Y: {std_dev_y:.6f}')
print(f'Автокорреляция Y (лаг 1): {autocorrelation_y:.6f}')


# -------------------------------- КОРРЕЛЯЦИЯ МЕЖДУ X И Y --------------------------------
correlation = gibbs_data['x'].corr(gibbs_data['y'])
print(f'\nКорреляция между X и Y: {correlation:.6f}')


# -------------------------------- ВИЗУАЛИЗАЦИЯ --------------------------------
plt.figure(figsize=(15, 10))

# Гистограмма X с кривой нормального распределения
plt.subplot(2, 3, 1)
counts_x, bins_x, _ = plt.hist(gibbs_data['x'], bins=50, color='skyblue', edgecolor='black', alpha=0.7, density=True)
# Теоретическая кривая нормального распределения для X
x_range = np.linspace(min(gibbs_data['x']), max(gibbs_data['x']), 100)
y_norm_x = (1 / (std_dev_x * np.sqrt(2 * np.pi))) * np.exp(-(x_range - mean_x)**2 / (2 * std_dev_x**2))
plt.plot(x_range, y_norm_x, 'r-', linewidth=2)
plt.title(f'Распределение X\nСреднее: {mean_x:.4f}, СКО: {std_dev_x:.4f}')
plt.grid(alpha=0.3)

# Гистограмма Y с кривой нормального распределения
plt.subplot(2, 3, 2)
counts_y, bins_y, _ = plt.hist(gibbs_data['y'], bins=50, color='lightgreen', edgecolor='black', alpha=0.7, density=True)
# Теоретическая кривая нормального распределения для Y
y_range = np.linspace(min(gibbs_data['y']), max(gibbs_data['y']), 100)
y_norm_y = (1 / (std_dev_y * np.sqrt(2 * np.pi))) * np.exp(-(y_range - mean_y)**2 / (2 * std_dev_y**2))
plt.plot(y_range, y_norm_y, 'r-', linewidth=2)
plt.title(f'Распределение Y\nСреднее: {mean_y:.4f}, СКО: {std_dev_y:.4f}')
plt.grid(alpha=0.3)

# Диаграмма рассеяния X и Y
plt.subplot(2, 3, 3)
plt.scatter(gibbs_data['x'], gibbs_data['y'], alpha=0.5, s=3, c='purple')
plt.title(f'Диаграмма рассеяния X и Y\nКорреляция: {correlation:.4f}')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(alpha=0.3)

# Автокорреляция X для разных лагов
plt.subplot(2, 3, 4)
lags = range(1, 21)
autocorrs_x = [pd.Series(gibbs_data['x']).autocorr(lag=i) for i in lags]
plt.bar(lags, autocorrs_x, color='skyblue')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.title('Автокорреляция X для разных лагов')
plt.xlabel('Лаг')
plt.ylabel('Автокорреляция')
plt.grid(alpha=0.3)

# Автокорреляция Y для разных лагов
plt.subplot(2, 3, 5)
autocorrs_y = [pd.Series(gibbs_data['y']).autocorr(lag=i) for i in lags]
plt.bar(lags, autocorrs_y, color='lightgreen')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.title('Автокорреляция Y для разных лагов')
plt.xlabel('Лаг')
plt.ylabel('Автокорреляция')
plt.grid(alpha=0.3)

# Временной ряд для X и Y (первые 100 значений)
plt.subplot(2, 3, 6)
plt.plot(gibbs_data['x'][:100], color='blue', label='X', alpha=0.7)
plt.plot(gibbs_data['y'][:100], color='green', label='Y', alpha=0.7)
plt.title('Временной ряд X и Y (первые 100 значений)')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('stats/gibbs_sampler_stats.png')
plt.show()


# -------------------------------- ДОПОЛНИТЕЛЬНЫЕ РАСЧЕТЫ --------------------------------
# Дарбина-Уотсона для проверки автокорреляции
def durbin_watson(data):
    diff = np.diff(data)
    return np.sum(diff**2) / np.sum(data**2)

dw_stat_x = durbin_watson(gibbs_data['x'])
dw_stat_y = durbin_watson(gibbs_data['y'])
print(f'Критерий Дарбина-Уотсона для X: {dw_stat_x:.6f}')
print(f'Критерий Дарбина-Уотсона для Y: {dw_stat_y:.6f}')

# Q-Q тесты на нормальность (не отображаем графики, просто получаем статистику)
_, p_value_x = stats.normaltest(gibbs_data['x'])
_, p_value_y = stats.normaltest(gibbs_data['y'])
print(f'p-значение теста на нормальность для X: {p_value_x:.6f}')
print(f'p-значение теста на нормальность для Y: {p_value_y:.6f}')
print(f'(p-значение > 0.05 говорит о нормальном распределении)')

# Особенность метода Гиббса: он использует условные распределения 
# и постепенно сходится к целевому совместному распределению.
# При правильной настройке, полученное распределение X и Y должно 
# соответствовать двумерному нормальному распределению с корреляцией,
# близкой к заданному значению. В данном случае, автокорреляция между 
# соседними значениями обычно высокая, что является особенностью метода. 