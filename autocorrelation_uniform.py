import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


uniform_data = pd.read_csv('datasets/uniform_distribution.csv')['values']




# Рассчитываем основные статистические показатели
# Среднее значение показывает центр распределения
mean = np.mean(uniform_data)   
# Дисперсия (СКО) показывает, насколько данные рассеяны вокруг среднего
variance = np.var(uniform_data) 
# Стандартное отклонение показывает, насколько данные рассеяны вокруг среднего
std_dev = np.std(uniform_data) 
# Автокорреляция показывает, насколько данные зависят от предыдущих значений
# В контексте автокорреляции, лаг определяет, насколько далеко друг от друга находятся элементы, между которыми мы ищем связь
autocorrelation = pd.Series(uniform_data).autocorr(lag=1) 

print(f'Среднее: {mean:.6f}')
print(f'Дисперсия: {variance:.6f}')
print(f'Стандартное отклонение: {std_dev:.6f}')
print(f'Автокорреляция (лаг 1): {autocorrelation:.6f}')




plt.figure(figsize=(12, 8))
# Гистограмма с теоретической кривой равномерного распределения
plt.subplot(2, 2, 1)
counts, bins, _ = plt.hist(uniform_data, bins=50, color='orange', edgecolor='black', alpha=0.7, density=True)
# Добавим теоретическую прямую линию для равномерного распределения
min_val, max_val = min(uniform_data), max(uniform_data)
x = np.linspace(min_val, max_val, 100)
y = np.ones_like(x) / (max_val - min_val)  # Плотность вероятности для равномерного распределения
plt.plot(x, y, 'r-', linewidth=2)
plt.title(f'Равномерное распределение\nСреднее: {mean:.4f}, Дисперсия: {variance:.4f}')
plt.grid(alpha=0.3)




# График Q-Q (quantile-quantile) для проверки на равномерность
# Для равномерного распределения используем uniform
plt.subplot(2, 2, 2)
from scipy import stats
stats.probplot(uniform_data, dist="uniform", plot=plt)
plt.title('Q-Q график (проверка на равномерность)')
plt.grid(alpha=0.3)




# Автокорреляция для разных лагов
# Для настоящего случайного процесса все значения должны быть близки к нулю
plt.subplot(2, 2, 3)
lags = range(1, 21)
autocorrs = [pd.Series(uniform_data).autocorr(lag=i) for i in lags]
plt.bar(lags, autocorrs, color='lightgreen')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.title('Автокорреляция для разных лагов')
plt.xlabel('Лаг')
plt.ylabel('Автокорреляция')
plt.grid(alpha=0.3)




# Временной ряд (первые 100 значений)
plt.subplot(2, 2, 4)
plt.plot(uniform_data[:100], color='purple')
plt.title('Временной ряд (первые 100 значений)')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('stats/uniform_distribution_stats.png')
plt.show()



# Дополнительно: расчет критерия Дарбина-Уотсона для проверки автокорреляции
# Критерий близкий к 2 означает отсутствие автокорреляции
# Меньше 2 - положительная автокорреляция
# Больше 2 - отрицательная автокорреляция
def durbin_watson(data):
    diff = np.diff(data)
    return np.sum(diff**2) / np.sum(data**2)

dw_stat = durbin_watson(uniform_data)
print(f'Критерий Дарбина-Уотсона: {dw_stat:.6f}')

# Для идеального равномерного распределения на отрезке [a,b]:
# - Теоретическое среднее = (a+b)/2
# - Теоретическая дисперсия = (b-a)²/12
# Для отрезка [1,100]: среднее = 50.5, дисперсия = 816.75 