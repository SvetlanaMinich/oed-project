import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Энтропия - это мера неопределенности или случайности в данных.

# Загружаем данные из CSV
gibbs_data = pd.read_csv('datasets/gibbs_sampler.csv')

# Рассчитаем энтропию отдельно для X и Y координат
bins = 100

# Энтропия для X
hist_x, bin_edges_x = np.histogram(gibbs_data['x'], bins=bins)
prob_x = hist_x / len(gibbs_data['x'])
prob_x = prob_x[prob_x > 0]
entropy_x = -np.sum(prob_x * np.log2(prob_x))

# Энтропия для Y
hist_y, bin_edges_y = np.histogram(gibbs_data['y'], bins=bins)
prob_y = hist_y / len(gibbs_data['y'])
prob_y = prob_y[prob_y > 0]
entropy_y = -np.sum(prob_y * np.log2(prob_y))

# Визуализация энтропии X
plt.figure(figsize=(10, 6))
plt.hist(gibbs_data['x'], bins=bins, color='skyblue', edgecolor='black')
plt.title(f'Гистограмма метода Гиббса (X)\nЭнтропия X: {entropy_x:.4f} бит')
plt.xlabel('Значение X')
plt.ylabel('Частота')
plt.grid(alpha=0.3)
plt.savefig('entropy/gibbs_x_entropy.png')
plt.show()

# Визуализация энтропии Y
plt.figure(figsize=(10, 6))
plt.hist(gibbs_data['y'], bins=bins, color='lightgreen', edgecolor='black')
plt.title(f'Гистограмма метода Гиббса (Y)\nЭнтропия Y: {entropy_y:.4f} бит')
plt.xlabel('Значение Y')
plt.ylabel('Частота')
plt.grid(alpha=0.3)
plt.savefig('entropy/gibbs_y_entropy.png')
plt.show()

# Попробуем оценить совместную энтропию
# Создаем 2D гистограмму
H, xedges, yedges = np.histogram2d(gibbs_data['x'], gibbs_data['y'], bins=[bins, bins])
# Нормализуем для получения совместных вероятностей
p_joint = H / np.sum(H)
# Убираем нули
p_joint = p_joint[p_joint > 0]
# Считаем совместную энтропию
joint_entropy = -np.sum(p_joint * np.log2(p_joint))

print(f'Энтропия X: {entropy_x:.4f} бит')
print(f'Энтропия Y: {entropy_y:.4f} бит')
print(f'Совместная энтропия X и Y: {joint_entropy:.4f} бит')
print(f'Взаимная информация: {entropy_x + entropy_y - joint_entropy:.4f} бит')

# Визуализация совместного распределения
plt.figure(figsize=(10, 8))
plt.scatter(gibbs_data['x'], gibbs_data['y'], alpha=0.5, s=3)
plt.title(f'Диаграмма рассеяния метода Гиббса\nСовместная энтропия: {joint_entropy:.4f} бит')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(alpha=0.3)
plt.colorbar(label='Плотность вероятности')
plt.savefig('entropy/gibbs_joint_entropy.png')
plt.show() 