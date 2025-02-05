# 赛题背景

本赛题以银行产品认购预测为背景，旨在预测客户是否会购买银行的产品。在与客户沟通的过程中，我们记录了以下信息：

- **客户沟通数据**：联系次数、上一次联系的时长、上一次联系的时间间隔等。
- **客户基本信息**：年龄、职业、婚姻状况、是否有违约记录、是否有房贷等。
- **市场环境数据**：就业情况、消费信息、银行同业拆借率等。

---

# 赛题任务

**To DO**：预测用户是否会购买银行产品。

---

# 数据字段说明

| 字段 | 说明 |
| --- | --- |
| `age` | 年龄 |
| `job` | 职业：admin, unknown, unemployed, management… |
| `marital` | 婚姻状况：married, divorced, single |
| `default` | 信用卡是否有违约：yes or no |
| `housing` | 是否有房贷：yes or no |
| `contact` | 联系方式：unknown, telephone, cellular |
| `month` | 上一次联系的月份：jan, feb, mar, … |
| `day_of_week` | 上一次联系的星期几：mon, tue, wed, thu, fri |
| `duration` | 上一次联系的时长（秒） |
| `campaign` | 活动期间联系客户的次数 |
| `pdays` | 上一次与客户联系后的间隔天数 |
| `previous` | 在本次营销活动前，与客户联系的次数 |
| `poutcome` | 之前营销活动的结果：unknown, other, failure, success |
| `emp_var_rate` | 就业变动率（季度指标） |
| `cons_price_index` | 消费者价格指数（月度指标） |
| `cons_conf_index` | 消费者信心指数（月度指标） |
| `lending_rate3m` | 银行同业拆借率 3个月利率（每日指标） |
| `nr_employed` | 雇员人数（季度指标） |
| `subscribe` | 客户是否购买：yes 或 no |

---

# 评价标准

**Accuracy（准确率）**：所有分类正确的百分比。