import matplotlib.pyplot as plt
import pandas as pd

# Загружаем данные из CSV файла
gibbs_data = pd.read_csv('datasets/gibbs_sampler.csv')

# Строим гистограмму для значений X
plt.figure(figsize=(10, 6))
plt.hist(gibbs_data['x'], bins=50, color='skyblue', edgecolor='black')
plt.title('Гистограмма метода Гиббса (значения X)')
plt.xlabel('Значение')
plt.ylabel('Частота')
plt.grid(alpha=0.3)
plt.savefig('histograms/gibbs_sampler_x.png')
plt.show()

# Строим гистограмму для значений Y
plt.figure(figsize=(10, 6))
plt.hist(gibbs_data['y'], bins=50, color='lightgreen', edgecolor='black')
plt.title('Гистограмма метода Гиббса (значения Y)')
plt.xlabel('Значение')
plt.ylabel('Частота')
plt.grid(alpha=0.3)
plt.savefig('histograms/gibbs_sampler_y.png')
plt.show()

# Строим диаграмму рассеяния (scatter plot) для визуализации пар значений
plt.figure(figsize=(10, 6))
plt.scatter(gibbs_data['x'], gibbs_data['y'], alpha=0.5, s=3)
plt.title('Диаграмма рассеяния метода Гиббса (X vs Y)')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(alpha=0.3)
plt.savefig('histograms/gibbs_sampler_scatter.png')
plt.show()