import pandas as pd

# 加载数据
data = pd.read_csv('data/train.csv')

# 打印数据前5行
print("数据前5行:")
print(data.head())

# 打印数据形状
print("\\n数据形状:")
print(data.shape)

# 打印数据类型
print("\\n数据类型:")
print(data.dtypes)

# 打印数值列的统计摘要
print("\\n数值列的统计摘要:")
print(data.describe())

# 统计subscribe列的数量
print("\\nsubscribe列的数量:")
print(data['subscribe'].value_counts())

# 统计各列的缺失值数量
print("\\n各列的缺失值数量:")
print(data.isnull().sum())
