# coding=gbk
# 使用ARIMA 模型对非平稳时间序列记性建模操作
# 差分运算具有强大的确定性的信息提取能力， 许多非平稳的序列差分后显示出平稳序列的性质， 这是称这个非平稳序列为差分平稳序列。
# 对差分平稳序列可以还是要ARMA 模型进行拟合， ARIMA 模型的实质就是差分预算与 ARMA 模型的结合。


# 导入数据
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# 导入数据
cityname = input("____")
filename = cityname + '.xlsx'

# 采用pandas中提供的read_excel（）直接返回dataframe格式的数据
# 而xlrd中返回的是原始数据 更适合于对excel中的数据进行定制化操作（更底层的操作）
# index_col：指定作为行索引的列，可以是列名、列索引或列表
data = pd.read_excel(filename, index_col=u'year')

# 画出输入excel时序图 设置绘图的一些字符编码问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 定义使其正常显示中文字体黑体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示表示负号
data.plot()
plt.show()

# 平稳性检测
print('原始序列的检验结果为：', adfuller(data[u'tanhui']))
# 原始序列的检验结果为： (1.8137710150945268, 0.9983759421514264, 10, 26, {'1%': -3.7112123008648155,
#  '10%': -2.6300945562130176, '5%': -2.981246804733728}, 299.46989866024177) 说明此处为非平稳序列
# 返回值依次为：adf, pvalue p值， usedlag, nobs, critical values临界值 , icbest, regresults, resstore
# adf 分别大于3中不同检验水平的3个临界值，单位检测统计量对应的p 值显著大于 0.05 ， 说明序列可以判定为 非平稳序列


# 对数据进行差分后并删除缺失值得到 自相关图和 偏相关图
D_data = data.diff(1).dropna()
D_data.columns = [u'碳汇差分']
D_data.plot()  # 画出差分后的时序图
plt.show()
plot_acf(D_data)  # 画出自相关图
plt.show()
plot_pacf(D_data, lags=1)  # 画出偏相关图
plt.show()
print(u'差分序列的ADF 检验结果为： ', adfuller(D_data[u'碳汇差分']))
# 平稳性检验，单位根检验
# 差分序列的ADF 检验结果为：  (-3.1560562366723537, 0.022673435440048798, 0, 35, {'1%': -3.6327426647230316,
# '10%': -2.6130173469387756, '5%': -2.9485102040816327}, 287.5909090780334)
# 一阶差分后的序列的时序图在均值附近比较平稳的波动， 自相关性有很强的短期相关性， 单位根检验 p值小于 0.05 ，所以说一阶差分后的序列是平稳序列


# 对一阶差分后的序列做白噪声检验
print(u'差分序列的白噪声检验结果：', acorr_ljungbox(D_data, lags=1))  # 返回统计量和 p 值
# 差分序列的白噪声检验结果： (array([11.30402222]), array([0.00077339])) p值为第二项， 远小于 0.05

# 找到最优模型对于的pq值
# 初始化一个BIC矩阵
# 通过拟合 ARIMA 模型来获取最优模型，以便后续的时间序列分析和预测
# fit() 方法对构建的 ARIMA 模型进行拟合，即估计模型的参数
bic_matrix = []
for p in range(4):
    temp = []
    for q in range(10):
        try:
            temp.append(sm.tsa.arima.ARIMA(data, order=(p, 1, q)).fit().aic)
            # print('3',sm.tsa.arima.ARIMA(data,order=(p,1,q)).fit().aic)
        except:
            temp.append(None)
        bic_matrix.append(temp)

bic_matrix = pd.DataFrame(bic_matrix)  # 将其转换成Dataframe 数据结构
# 在贝叶斯矩阵中找到最小值
p, q = bic_matrix.stack().astype('float64').idxmin()  # 先使用stack 展*获得多级索引， 然后使用 idxmin 找出最小值的位置
print(u'BIC 最小的p值 和 q 值：%s,%s' % (p, q))  # BIC 最小的p值 和 q 值：0,1
# 所以可以建立ARIMA 模型，ARIMA(0,1,1)
# arima_model=sm.tsa.ARIMA()
model = sm.tsa.ARIMA(data, order=(p, 1, q))
results = model.fit()
predict_sunspots = results.predict()  # In-sample prediction and out-of-sample forecasting
predict_sunspots_2 = results.forecast(5)  # Out-of-sample forecasts预测
print('The result of the predicton,next 1:', predict_sunspots_2)

print("predict past data", predict_sunspots)
x = [2025, 2030, 2035, 2040, 2045]
y = []
for i in range(6, 11):
    y.append(predict_sunspots_2[i])
print(y)
