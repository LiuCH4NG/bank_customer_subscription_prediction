"""
银行产品购买预测完整流程
包含：数据预处理、模型训练、特征选择、模型存储
"""

# ========== 环境配置 ==========
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

# 创建模型存储目录
os.makedirs('model', exist_ok=True)

# ========== 数据预处理 ==========
def load_and_preprocess(data_path):
    """数据加载与预处理"""
    data = pd.read_csv(data_path)
    
    # 目标变量编码
    data['subscribe'] = data['subscribe'].map({'yes': 1, 'no': 0})
    
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
    
    # 处理pdays特殊值
    data['contacted_before'] = np.where(data['pdays'] == 9999, 0, 1)
    data['pdays'] = np.where(data['pdays'] == 9999, 0, data['pdays'])
    
    return data

# ========== 特征工程配置 ==========
numerical_cols = ['age', 'duration', 'campaign', 'pdays', 'previous',
                 'emp_var_rate', 'cons_price_index', 'cons_conf_index',
                 'lending_rate3m', 'nr_employed', 'month_sin', 'month_cos',
                 'day_sin', 'day_cos', 'contacted_before']

categorical_cols = ['job', 'marital', 'contact', 'poutcome']

# ========== 模型训练 ==========
def train_models(X_train, y_train):
    """训练完整模型"""
    # 预处理管道
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    # 随机森林模型
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            class_weight='balanced',
            random_state=42
        ))
    ])

    # 使用专为不平衡数据设计的随机森林
    brf = BalancedRandomForestClassifier(
        n_estimators=200,
        sampling_strategy='auto',
        replacement=True,
        random_state=42
    )

    brf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', brf)
    ])

    brf_pipeline.fit(X_train, y_train)
    
    # 参数网格搜索
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5]
    }
    
    grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    
    # XGBoost模型
    xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        ))
    ])
    xgb_pipeline.fit(X_train, y_train)
    
    return best_rf, xgb_pipeline, brf_pipeline

# ========== 特征选择 ==========
def feature_selection(model, preprocessor, X_train, y_train):
    """特征重要性分析与选择"""
    # 获取特征名称
    preprocessor.fit(X_train)
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    all_features = numerical_cols + list(cat_features)
    
    # 获取模型特征重要性
    if isinstance(model.named_steps['classifier'], RandomForestClassifier):
        importances = model.named_steps['classifier'].feature_importances_
    else:
        importances = model.named_steps['classifier'].feature_importances_
    
    # 可视化重要性
    plt.figure(figsize=(12, 8))
    plt.barh(all_features, importances)
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('model/feature_importance.png')
    plt.close()
    
    # 自动选择特征（重要性>中位数的2倍）
    threshold = np.median(importances) * 2
    selected_features = [f for f, imp in zip(all_features, importances) if imp > threshold]
    print(f'Selected {len(selected_features)}/{len(all_features)} features')
    
    return selected_features

# ========== 优化模型训练 ==========
def train_optimized_model(selected_features, X_train, y_train):
    """训练优化后的模型"""
    # 解析选择的特征
    num_features = [f for f in selected_features if f in numerical_cols]
    cat_features = list(set([f.split('_')[0] for f in selected_features if f.startswith(tuple(categorical_cols))]))
    
    # 新预处理流程
    optimized_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])
    
    # 优化后的随机森林
    optimized_rf = Pipeline([
        ('preprocessor', optimized_preprocessor),
        ('classifier', RandomForestClassifier(
            class_weight='balanced',
            n_estimators=200,
            max_depth=None,
            random_state=42
        ))
    ])
    optimized_rf.fit(X_train, y_train)
    return optimized_rf

# ========== 主执行流程 ==========
if __name__ == "__main__":
    # 数据准备
    data = load_and_preprocess('data/train.csv')
    X = data.drop('subscribe', axis=1)
    y = data['subscribe']
    # 使用train_test_split函数将数据集X和y分割为训练集和测试集
    # X_train: 训练集的特征数据
    # X_test: 测试集的特征数据
    # y_train: 训练集的目标数据
    # y_test: 测试集的目标数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 训练完整模型
    print("训练完整模型...")
    rf_model, xgb_model, brf_pipeline = train_models(X_train, y_train)
    
    # 评估完整模型
    print("\n完整模型性能：")
    print(f"随机森林测试集准确率: {accuracy_score(y_test, rf_model.predict(X_test)):.4f}")
    print(f"XGBoost测试集准确率: {accuracy_score(y_test, xgb_model.predict(X_test)):.4f}")
    print(f"Bagging随机森林测试集准确率: {accuracy_score(y_test, brf_pipeline.predict(X_test)):.4f}")
    
    # 特征选择
    print("\n进行特征选择...")
    selected_features = feature_selection(rf_model, rf_model.named_steps['preprocessor'], X_train, y_train)
    
    # 训练优化模型
    print("\n训练优化模型...")
    optimized_model = train_optimized_model(selected_features, X_train, y_train)
    
    # 评估优化模型
    optimized_acc = accuracy_score(y_test, optimized_model.predict(X_test))
    print(f"\n优化模型测试集准确率: {optimized_acc:.4f}")

    
    # 模型存储
    print("\n保存模型...")
    joblib.dump(rf_model, 'model/rf_full.pkl')
    joblib.dump(xgb_model, 'model/xgb_full.pkl')
    joblib.dump(optimized_model, 'model/rf_optimized.pkl')
    joblib.dump({
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'selected_features': selected_features
    }, 'model/feature_meta.pkl')

    
    print("训练流程完成！保存模型至model/目录")