import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


exponential_data = pd.read_csv('datasets/exponential_distribution.csv')['values']





# Рассчитываем основные статистические показатели
# Среднее значение показывает центр распределения
mean = np.mean(exponential_data)   
# Дисперсия показывает, насколько данные рассеяны вокруг среднего
variance = np.var(exponential_data) 
# Стандартное отклонение показывает, насколько данные рассеяны вокруг среднего
std_dev = np.std(exponential_data) 
# Автокорреляция показывает, насколько данные зависят от предыдущих значений
autocorrelation = pd.Series(exponential_data).autocorr(lag=1) 

print(f'Среднее: {mean:.6f}')
print(f'Дисперсия: {variance:.6f}')
print(f'Стандартное отклонение: {std_dev:.6f}')
print(f'Автокорреляция (лаг 1): {autocorrelation:.6f}')




plt.figure(figsize=(12, 8))
# Гистограмма с теоретической кривой экспоненциального распределения
plt.subplot(2, 2, 1)
counts, bins, _ = plt.hist(exponential_data, bins=50, color='salmon', edgecolor='black', alpha=0.7, density=True)
# Добавим теоретическую кривую для экспоненциального распределения
# Параметр лямбда (λ) определяет скорость убывания и равен 1/среднее
lambda_param = 1 / mean
x = np.linspace(0, max(exponential_data), 100)
y = lambda_param * np.exp(-lambda_param * x)  # Плотность вероятности для экспоненциального распределения
plt.plot(x, y, 'r-', linewidth=2)
plt.title(f'Экспоненциальное распределение\nСреднее: {mean:.4f}, Дисперсия: {variance:.4f}')
plt.grid(alpha=0.3)




# График Q-Q (quantile-quantile) для проверки на экспоненциальность
plt.subplot(2, 2, 2)
from scipy import stats
stats.probplot(exponential_data, dist="expon", plot=plt)
plt.title('Q-Q график (проверка на экспоненциальность)')
plt.grid(alpha=0.3)




# Автокорреляция для разных лагов
plt.subplot(2, 2, 3)
lags = range(1, 21)
autocorrs = [pd.Series(exponential_data).autocorr(lag=i) for i in lags]
plt.bar(lags, autocorrs, color='lightgreen')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.title('Автокорреляция для разных лагов')
plt.xlabel('Лаг')
plt.ylabel('Автокорреляция')
plt.grid(alpha=0.3)




# Временной ряд (первые 100 значений)
plt.subplot(2, 2, 4)
plt.plot(exponential_data[:100], color='purple')
plt.title('Временной ряд (первые 100 значений)')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('stats/exponential_distribution_stats.png')
plt.show()




# Дополнительно: расчет критерия Дарбина-Уотсона для проверки автокорреляции
def durbin_watson(data):
    diff = np.diff(data)
    return np.sum(diff**2) / np.sum(data**2)

dw_stat = durbin_watson(exponential_data)
print(f'Критерий Дарбина-Уотсона: {dw_stat:.6f}')

# Для идеального экспоненциального распределения с параметром λ:
# - Теоретическое среднее = 1/λ
# - Теоретическая дисперсия = 1/λ²
# - Теоретическое стандартное отклонение = 1/λ
# Для λ = 1: среднее = 1, дисперсия = 1