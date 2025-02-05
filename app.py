from flask import Flask, request, render_template
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# 加载模型和元数据
model = joblib.load('model/rf_optimized.pkl')
meta = joblib.load('model/feature_meta.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 获取表单数据
        form_data = request.form.to_dict()
        
        # 转换数值类型
        numeric_fields = ['age', 'duration', 'campaign', 'pdays', 'previous',
                         'emp_var_rate', 'cons_price_index', 'cons_conf_index',
                         'lending_rate3m', 'nr_employed', 'month_sin', 'month_cos',
                         'day_sin']

        # 月份星期转换
        month = form_data.pop('month')
        day = form_data.pop('day')
        
        form_data['month_sin'] = np.sin(2 * np.pi * int(month) / 12)
        form_data['month_cos'] = np.cos(2 * np.pi * int(month) / 12)
        form_data['day_sin'] = np.sin(2 * np.pi * int(day) / 5)
    
        for field in numeric_fields:
            form_data[field] = float(form_data[field])
        
        # 创建DataFrame
        input_df = pd.DataFrame([form_data])
        
        # 特征验证
        missing = set(meta['selected_features']) - set(input_df.columns)
        if missing:
            return render_template('error.html', 
                                message=f"缺少必要特征: {', '.join(missing)}")
        
        # 选择特征
        input_df = input_df[meta['selected_features']]

        # 预测
        prediction = model.predict(input_df)[0]
        result = "会购买" if prediction == 1 else "不会购买"
        probability = model.predict_proba(input_df)[0][1]
        
        return render_template('result.html', 
                             result=result,
                             probability=f"{probability*100:.1f}%")
    
    except Exception as e:
        return render_template('error.html', 
                             message=f"预测错误: {str(e)}")

# 错误页面模板
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', message="页面不存在"), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8010, debug=True)