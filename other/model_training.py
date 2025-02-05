import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import StackingClassifier, VotingClassifier

# 加载处理后的数据
data = pd.read_csv('data/processed_train.csv')

# 准备特征和目标变量
X = data.drop(['id','subscribe'], axis=1)
y = data['subscribe']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

base_models = [
    ('xgb', XGBClassifier()),
    ('rf', RandomForestClassifier()),
    ('svm', SVC(probability=True))
]

from sklearn.model_selection import GridSearchCV

# 使用SMOTE进行过采样
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

param_grid = {
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0]
}

xgb = XGBClassifier(eval_metric='logloss')
grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_resampled, y_resampled)  # 使用过采样数据

best_xgb = grid_search.best_estimator_
print("最佳参数:", grid_search.best_params_)



# 定义要比较的模型
models = {
    '逻辑回归': LogisticRegression(max_iter=1000),
    '随机森林': RandomForestClassifier(random_state=42),
    '支持向量机': SVC(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
    'K近邻': KNeighborsClassifier(),
    '堆叠分类器': StackingClassifier(
                estimators=base_models,
                final_estimator=LogisticRegression(),
                stack_method='auto'
            ),
    '投票分类器': VotingClassifier(
                estimators=[('xgb', best_xgb), 
                        ('rf', RandomForestClassifier(n_estimators=200)),
                        ('svm', SVC(probability=True, C=1.0))],
                voting='soft'
            )
}

# 用于存储结果的字典
results = {}

# 循环训练和评估每个模型
for model_name, model in models.items():
    start_time = time.time()
    
    # 训练模型
    model.fit(X_train_scaled, y_train)
    
    # 预测测试集
    y_pred = model.predict(X_test_scaled)
    
    # 计算评估指标
    train_time = time.time() - start_time
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # 存储结果
    results[model_name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1': report['weighted avg']['f1-score'],
        'train_time': train_time
    }
    
    # 打印结果
    print(f"\n{model_name} 结果:")
    print(f"- 训练时间: {train_time:.2f}秒")
    print(f"- 测试集准确率: {accuracy:.4f}")
    print(f"- 加权平均F1分数: {results[model_name]['f1']:.4f}")

    # 获取特征权重 (仅适用于逻辑回归和随机森林)
    if model_name == '逻辑回归' or model_name == '随机森林':
        if hasattr(model, 'coef_'):  # 逻辑回归
            feature_weights = model.coef_[0]
        elif hasattr(model, 'feature_importances_'):  # 随机森林
            feature_weights = model.feature_importances_
        else:
            feature_weights = None

        if feature_weights is not None:
            feature_names = X.columns  # 获取特征名称
            feature_importance = pd.DataFrame({'特征': feature_names, '权重': feature_weights})
            feature_importance = feature_importance.sort_values(by='权重', ascending=False)
            print("\n特征权重 (前10个):")
            print(feature_importance.head(10))


# 创建结果比较表格
import joblib

results_df = pd.DataFrame.from_dict(results, orient='index')
results_df = results_df.sort_values(by='accuracy', ascending=False)

joblib.dump(models, 'models.pkl')
