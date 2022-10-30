# 导入需要的第三方库
# 用于保存和提取模型
import joblib
import lightgbm as lgbm
import numpy as np
# 导入相关库
import optuna
import pandas as pd
from lightgbm import log_evaluation, early_stopping
# 对数据进行训练之前检测出不太好地超参数集，从而显着减少搜索时间
from optuna.integration import LightGBMPruningCallback
# 导入方差过滤
from sklearn.feature_selection import VarianceThreshold
# K折交叉验证
from sklearn.model_selection import StratifiedKFold
# 训练集和测试集分割
from sklearn.model_selection import train_test_split
# 评价指标
# 导入LabelEncoder
from sklearn.preprocessing import LabelEncoder

# 训练集路径
train_path = r'D:\python\机器学习\机器学习课程设计\train.csv'
# 测试集路径
test_path = r'D:\python\机器学习\机器学习课程设计\evaluation_public.csv'

# 读入训练集、测试集数据
traindatas = pd.read_csv(train_path)
testdatas = pd.read_csv(test_path)
# 结果：取数据data_y（最后一行0，1）
data_y = traindatas[['is_risk']]
# 训练集删除data_y
traindatas = traindatas.drop(['is_risk'], axis=1)
# 合并训练集和测试集
totaldf = pd.concat([traindatas, testdatas], axis=0)

totaldf = pd.read_csv(r'D:\python\机器学习\机器学习课程设计\totaldf.csv')
# 将特征值和结果表进行合并
totaldf = pd.concat([totaldf, data_y], axis=1)

# 根据该代码可以发现内网是无用数据，所以去除，可以发现id、uscname、ip_transform也有可能无用，以及浏览器操作系统
# 先保留进行观察，后续考虑是否删除
# region
# def train(x, y, z):
#     # 47660*13
#     m, n = x.shape
#     # 拉普拉斯修正   p0 无风险概率
#     p1 = (len(y[y == 1])+1) / (m+2)
#     p0 = (len(y[y == 0])+1) / (m+2)
#     # 有、无风险概率   有7696  无39964
#     m0 = len(y[y == 0])
#     m1 = len(y[y == 1])
#
#     # 概率
#     p_0 = p0
#     p_1 = p1
#     #
#     X1 = x[y == 1]
#     X0 = x[y == 0]
#     #
#     # 遍历13列
#     for i in range(n):
#         # 第n列数据
#         data_xi = x[:, i]
#         for j in range(m):
#             # 去空
#             if data_xi[j] != 'nan':
#                 if j == 0:
#                     # 存入第一个数据
#                     data_xk = np.array([data_xi[j]])
#                 # 如果不存在则存入数组
#                 if len(data_xk[data_xk == data_xi[j]]) == 0:
#                     data_xk = np.append(data_xk, data_xi[j])
#         # 通过观察op_datetime和ip_type对数据无任何影响，所以在输入数据中去除
#         r = len(data_xk)
#         # X0[:, i]  第一列的所有数据
#         # X0[:, 1]第一列数据
#         # x[0, :] 第一行所有数据
#         p_i0 = (len(X0[X0[:, i] == z[i]])+1) / (m0+r)
#         p_i1 = (len(X1[X1[:, i] == z[i]])+1) / (m1+r)
#         p_0 = p_0 * p_i0
#         p_1 = p_1 * p_i1
#     if p_1 > p_0:
#         result = 1
#     else:
#         result = 0
#     return result
#
#
# if __name__ == '__main__':
#     data1 = pd.read_csv(r'D:\python\机器学习\机器学习课程设计\submitdf-7.csv').values
#     data_y = data1[:, 1]
#     print(data_y)
#     m, n = data1.shape
#     # 拉普拉斯修正   p0 无风险概率
#     p1 = len(data_y[data_y == 1]) / m
#     # 0.5034
#     print(p1)
#     data_x1 = data1[:, 1:9].astype(str)
#     data_x2 = data1[:, 11:15].astype(str)
#     data_x = np.hstack((data_x1, data_x2))
#
#     data_y = data1[:, 16]
#
#     data2 = pd.read_csv(r'D:\python\机器学习\机器学习课程设计\evaluation_public.csv').values
#     data_z1 = data2[:, 1:9].astype(str)
#     data_z2 = data2[:, 11:15].astype(str)
#     data_z = np.hstack((data_z1, data_z2))
#     # 25710 * 13
#     u, _ = data_z.shape
#     ids = list(range(100))
#     results = []
#     for i in range(100):
#         result = train(data_x, data_y, data_z[i, :])
#         results.append(result)
#         print(i)
#     dataframe = pd.DataFrame({'id': ids, 'is_risk': results})
#
#     dataframe.to_csv("test.csv", index=False, sep=',')
# endregion
# 去除掉ip_type特征值
totaldf['ip_type'].value_counts()
totaldf = totaldf.drop(['ip_type'], axis=1)

# 对数据进行处理，将str类型的数据转换为离散类型
object_list = ['ip_transform', 'device_num_transform', 'browser', 'browser_version', 'department',
               'log_system_transform', 'op_city', 'os_type', 'os_version', 'url', 'http_status_code']
for feature in object_list:
    totaldf[f'{feature}'] = LabelEncoder().fit_transform(totaldf[feature])

# 对于op_datetime需要进行转换为时间格式，求出两个特征之间的关联
totaldf['op_datetime'] = pd.to_datetime(totaldf['op_datetime'])
# 然后再转化为离散型
totaldf['op_ts'] = totaldf['op_datetime'].values.astype(np.int64)
# 对于同用户名的浏览记录，按时间排序
totaldf = totaldf.sort_values(by=['user_name', 'op_ts']).reset_index(drop=True)
totaldf.head()
# 记录登录时间
totaldf['log1'] = totaldf.groupby(['user_name'])['op_ts'].shift(1)
totaldf['log2'] = totaldf.groupby(['user_name'])['op_ts'].shift(2)

# 计算与第一、二次登陆时间差值
totaldf['diff1'] = totaldf['op_ts'] - totaldf['log1']
totaldf['diff2'] = totaldf['op_ts'] - totaldf['log2']

# 用新的时间特征（平均值和标准差）来替换原先的时间特征
totaldf = totaldf.drop(['op_datetime', 'op_month'], axis=1)
totaldf.head()
deal_list = ['department', 'ip_transform', 'device_num_transform', 'browser_version', 'browser', 'os_type',
             'os_version', 'op_city', 'log_system_transform', 'url']
for feature in deal_list:
    totaldf[feature + 'mean'] = totaldf.groupby(feature)['diff1'].transform('mean')
    totaldf[feature + 'std'] = totaldf.groupby(feature)['diff1'].transform('std')

# 统计用户系统访问次数，并将用户名进行数据处理
totaldf['access_time'] = totaldf.groupby('user_name')['department'].transform('count')
totaldf.head()
totaldf['user_name'] = LabelEncoder().fit_transform(totaldf['user_name'])
totaldf.head()

# 对训练集和测试集进行划分，is_risk为空即为测试集，反之就为训练集
traindatas = totaldf[totaldf['is_risk'].notna()]
testdatas = totaldf[totaldf['is_risk'].isna()]
# 测试集id
test_id = testdatas[['id']]
# 提取出训练集结果
data_y = traindatas[['is_risk']]
features = [feature for feature in totaldf.columns if
            feature not in ['id', 'is_risk', 'year', 'min', 'op_ts', 'log1', 'log2']]
traindatas = traindatas[features]
testdatas = testdatas[features]

# 进行方差过滤
v = VarianceThreshold(threshold=0)
# 对训练集处理
vt_train = v.fit_transform(traindatas)
vt_train = pd.DataFrame(vt_train)
# 对测试集进行处理
vt_test = v.transform(testdatas)
vt_test = pd.DataFrame(vt_test)
x = vt_train
y = data_y['is_risk']


# 缺失值模型：梯度提升决策树lightgbm
def objective(trial, x, y, fold_time):
    # 参数
    params = {'n_estimators': trial.suggest_int('n_estimators', 1, 1000),
              'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),  # 学习率
              'num_leaves': trial.suggest_int('num_leaves', 0, 50),  # 一棵树的叶子节点数
              'max_depth': trial.suggest_int('max_depth', 1, 10),  # 树的最大深度，控制过拟合
              "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 1.0, step=0.01),  # 选择特征比例
              "reg_alpha": trial.suggest_int("reg_alpha", 0, 10, step=1),  # 正则化
              "reg_lambda": trial.suggest_int("reg_lambda", 0, 10, step=1),
              "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0, step=0.01),  # 随机选择部分数据不重新采样
              "bagging_freq": trial.suggest_int("bagging_freq", 2, 20),  # 每k次迭代执行
              }
    # 分类
    cv = StratifiedKFold(n_splits=fold_time, shuffle=True, random_state=2022)
    cv_scores = np.zeros(fold_time)
    # 划分训练集和测试集
    for i, (train_idx, test_idx) in enumerate(cv.split(x, y)):
        X_train, X_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        # lightGBM的分类器初始化
        model = lgbm.LGBMClassifier(boosting='gbdt',   # gbdt提升决策树,rf随机森林
                                    objective='binary',    # 二分类
                                    n_jobs=-1,
                                    force_row_wise=True,
                                    random_state=2022,
                                    **params)
        # 填充训练数据
        callbacks = [log_evaluation(period=100), LightGBMPruningCallback(trial, 'auc'),
                     early_stopping(stopping_rounds=50)]
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='auc',
            callbacks=callbacks
        )
        # 获得预测分数
        pred_score = model.score(X_test, y_test)
        cv_scores[i] = pred_score
    # 返回平均值
    return np.mean(cv_scores)

study = optuna.create_study(study_name='LGBMClassifier', direction='maximize')
func = lambda trial: objective(trial, x, y, fold_time=7)
# 运行数目
study.optimize(func, n_trials=6)
values = []
# 将得到最优时的参数存入数组中。
for key, value in study.best_params.items():
    values.append(value)
print(values)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
# 填入参数
LGBMC = lgbm.LGBMClassifier(boosting_type='gbdt',
                            objective='binary',
                            n_jobs=-1,
                            force_row_wise=True,
                            random_state=2022,
                            n_estimators=values[0],
                            learning_rate=values[1],
                            num_leaves=values[2],
                            max_depth=values[3],
                            feature_fraction=values[4],
                            reg_alpha=values[5],
                            reg_lambda=values[6],
                            bagging_fraction=values[7],
                            bagging_freq=values[8])
LGBMC.fit(x_train, y_train)

# 获得每个样本的预测值,并且写入文件
y_pred = LGBMC.predict(vt_test)
submitdf = pd.DataFrame({'id': test_id['id'], 'is_risk': y_pred})
submitdf = submitdf.sort_values(['id']).reset_index(drop=True)
submitdf.to_csv(r'D:\python\机器学习\机器学习课程设计\submitdf.csv', index=False, encoding='utf-8')
submitdf.head()
