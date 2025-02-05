import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

# 加载数据
data = pd.read_csv('data/train.csv')

# 类别特征列表
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
numerical_features = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']

# 交互特征
data['age_job'] = data['age'].astype(str) + '_' + data['job'].astype(str)
data['education_marital'] = data['education'].astype(str) + '_' + data['marital'].astype(str)
categorical_features.extend(['age_job', 'education_marital'])

# 使用独热编码处理类别特征，并处理unknown值
for col in categorical_features:
    # 填充 unknown 值为 "unknown_category"
    data[col] = data[col].str.replace('unknown', 'unknown_category')
    # 进行独热编码
    dummies = pd.get_dummies(data[col], prefix=col)
    data = pd.concat([data, dummies], axis=1)
    data.drop(col, axis=1, inplace=True)

# 多项式特征
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(data[numerical_features])
poly_feature_names = poly.get_feature_names_out(numerical_features)
poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)
data = pd.concat([data.reset_index(drop=True), poly_df.reset_index(drop=True)], axis=1)
data.drop(numerical_features, axis=1, inplace=True)


# 将 'subscribe' 列转换为数值型 (yes: 1, no: 0)
data['subscribe'] = data['subscribe'].map({'yes': 1, 'no': 0})

# 打印处理后的数据前5行和形状
print("处理后的数据前5行:")
print(data.head())
print("\\n处理后的数据形状:")
print(data.shape)

# 保存处理后的数据到文件 processed_train.csv
data.to_csv('data/processed_train.csv', index=False)
print("\\n处理后的数据已保存到 data/processed_train.csv")
