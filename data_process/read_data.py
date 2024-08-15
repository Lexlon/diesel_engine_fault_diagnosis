import pandas as pd

file_path = '发动机试验数据/高频信号/1800-57%-正常工况/1800-57%-1-Z1.csv'
data = pd.read_csv(file_path)
data = data.drop(columns=[data.columns[-1]])
samples = data.iloc[:, 1:].values
print("样本集的形状:", samples.shape)
print("前几行的样本:")
print(samples[:5])
