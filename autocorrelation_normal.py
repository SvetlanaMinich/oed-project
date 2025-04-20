import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


normal_data = pd.read_csv('datasets/normal_distribution.csv')['values']

# Рассчитываем основные статистические показатели
# Среднее значение показывает центр распределения
mean = np.mean(normal_data)   
# Дисперсия (СКО) показывает, насколько данные рассеяны вокруг среднего
variance = np.var(normal_data) 
# Стандартное отклонение показывает, насколько данные рассеяны вокруг среднего
std_dev = np.std(normal_data) 
# Автокорреляция показывает, насколько данные зависят от предыдущих значений
# В контексте автокорреляции, лаг определяет, насколько далеко друг от друга находятся элементы, между которыми мы ищем связь
autocorrelation = pd.Series(normal_data).autocorr(lag=1) 

print(f'Среднее: {mean:.6f}')
print(f'Дисперсия: {variance:.6f}')
print(f'Стандартное отклонение: {std_dev:.6f}')
print(f'Автокорреляция (лаг 1): {autocorrelation:.6f}')




plt.figure(figsize=(12, 8))
# Гистограмма с кривой нормального распределения
plt.subplot(2, 2, 1)
counts, bins, _ = plt.hist(normal_data, bins=50, color='skyblue', edgecolor='black', alpha=0.7, density=True)
# Добавим теоретическую кривую нормального распределения
x = np.linspace(min(normal_data), max(normal_data), 100)
y = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-(x - mean)**2 / (2 * std_dev**2))
plt.plot(x, y, 'r-', linewidth=2)
plt.title(f'Нормальное распределение\nСреднее: {mean:.4f}, Дисперсия: {variance:.4f}')
plt.grid(alpha=0.3)




# График Q-Q (quantile-quantile) для проверки нормальности
# Если точки ложатся близко к прямой линии, данные очень близки к нормальному распределению
plt.subplot(2, 2, 2)
from scipy import stats
stats.probplot(normal_data, dist="norm", plot=plt)
plt.title('Q-Q график (проверка на нормальность)')
plt.grid(alpha=0.3)




# Автокорреляция для разных лагов
# Для настоящего случайного процесса все значения должны быть близки к нулю
# Если есть выбросы, значит генератор имеет скрытые зависимости
plt.subplot(2, 2, 3)
lags = range(1, 21)
autocorrs = [pd.Series(normal_data).autocorr(lag=i) for i in lags]
plt.bar(lags, autocorrs, color='lightgreen')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.title('Автокорреляция для разных лагов')
plt.xlabel('Лаг')
plt.ylabel('Автокорреляция')
plt.grid(alpha=0.3)




# Временной ряд (первые 100 значений)
plt.subplot(2, 2, 4)
plt.plot(normal_data[:100], color='purple')
plt.title('Временной ряд (первые 100 значений)')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('stats/normal_distribution_stats.png')
plt.show()




# Дополнительно: расчет критерия Дарбина-Уотсона для проверки автокорреляции
# Критерий близкий к 2 означает отсутствие автокорреляции
# Меньше 2 - положительная автокорреляция
# Больше 2 - отрицательная автокорреляция
def durbin_watson(data):
    diff = np.diff(data)
    return np.sum(diff**2) / np.sum(data**2)

dw_stat = durbin_watson(normal_data)
print(f'Критерий Дарбина-Уотсона: {dw_stat:.6f}') 