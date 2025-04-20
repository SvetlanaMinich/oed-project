import numpy as np
import pandas as pd


# Метод Гаусса (нормальное распределение)
random_numbers = np.random.normal(loc=0, scale=1, size=10_000)
# Сохраняем в CSV файл
pd.DataFrame(random_numbers, columns=['values']).to_csv('datasets/normal_distribution.csv', index=False)


# Метод Экспоненциального распределения
random_numbers = np.random.exponential(scale=1, size=10_000)
pd.DataFrame(random_numbers, columns=['values']).to_csv('datasets/exponential_distribution.csv', index=False)


# Метод Лептона (для равномерного распределения)
random_numbers = np.random.uniform(low=1, high=100, size=10_000)
pd.DataFrame(random_numbers, columns=['values']).to_csv('datasets/uniform_distribution.csv', index=False)


# Метод Гиббса 
def gibbs_sampler(num_samples):
    samples = []
    x = np.random.rand() # начальное значение
    for _ in range(num_samples):
        y = np.random.normal(x, 1) # условное для y
        x = np.random.normal(y, 1) # условное для x
        samples.append((x, y))
    return samples
samples = gibbs_sampler(10_000)
pd.DataFrame(samples, columns=['x', 'y']).to_csv('datasets/gibbs_sampler.csv', index=False)


# # Метод Марковских цепей 
# def markov_chain(num_steps):
#     states = [0, 1] # два состояния
#     current_state = 0 # начальное состояние
#     samples = []
#     for _ in range(num_steps):
#         if current_state == 0:
#             current_state = np.random.choice(states, p=[0.5, 0.5]) # переход
#         else:
#             current_state = np.random.choice(states, p=[0.1, 0.9]) # другой переход
#             samples.append(current_state)
#     return samples
# samples = markov_chain(10_000)
# pd.DataFrame(samples, columns=['values']).to_csv('datasets/markov_chain.csv', index=False)
