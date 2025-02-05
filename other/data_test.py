import pandas as pd
import joblib
import numpy as np

# 加载模型和元数据
model = joblib.load('model/rf_optimized.pkl')
meta = joblib.load('model/feature_meta.pkl')

# 加载测试数据
data = pd.read_csv('data/train.csv')

# 二元变量处理
binary_map = {'no': 0, 'yes': 1}
data['default'] = data['default'].map(binary_map)
data['housing'] = data['housing'].map(binary_map)

# 周期性特征处理
month_map = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 
            'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
data['month'] = data['month'].map(month_map)
data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

day_map = {'mon':1, 'tue':2, 'wed':3, 'thu':4, 'fri':5}
data['day_of_week'] = data['day_of_week'].map(day_map)
data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 5)
data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 5)


# 预处理测试数据
X_test = data[meta['selected_features']]

# 取前10个
X_test = X_test.head(10)

# 预测
y_pred = model.predict(X_test)

# 输出预测结果
print(y_pred)