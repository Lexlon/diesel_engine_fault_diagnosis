import pandas as pd
import numpy as np
# 读取10个csv并合并
for num in range(1, 2):
    file_path = f'../发动机试验数据/高频信号/1800-57%-正常工况/1800-57%-{num}-Z1.csv'
    data = pd.read_csv(file_path)
    data = data.drop(columns=[data.columns[-1]])
    samples = data.iloc[:, 1:].values
    if num == 1:
        positive_samples = samples
    else:
        positive_samples = np.concatenate((positive_samples, samples), axis=1)
print("样本集的形状:", positive_samples.shape)
print("前几行的样本:")
print(positive_samples[:5])

for num in range(1, 2):
    file_path = f'../发动机试验数据/高频信号/1800-57%-断缸/1800-57%-{num}-Z1.csv'
    data = pd.read_csv(file_path)
    data = data.drop(columns=[data.columns[-1]])
    n_samples = data.iloc[:, 1:].values
    if num == 1:
        negetive_samples = n_samples
    else:
        negetive_samples = np.concatenate((negetive_samples, n_samples), axis=1)
print("样本集的形状:", negetive_samples.shape)
print("前几行的样本:")
print(negetive_samples[:5])
import numpy as np
import matplotlib.pyplot as plt

# 随机抽取一个样本
degree_values = data['Degree'].values
random_index = np.random.randint(0, negetive_samples.shape[1])
random_sample_pos = positive_samples[:, random_index]
random_sample_neg = negetive_samples[:, random_index]


# 绘制信号图
plt.figure(figsize=(10, 5))
plt.plot(degree_values, random_sample_pos, color = 'blue', label='Positive Sample')
plt.plot(degree_values, random_sample_neg, color = 'red', label='Negative Sample')
plt.title('Angular Domain Signal')
plt.xlabel('degrees')
plt.ylabel('Acceleration Amplitude')
plt.xlim(-360, 360)
plt.legend()
plt.grid(True)

plt.grid(True)
plt.savefig(f'raw_data.png')