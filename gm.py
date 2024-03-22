import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import array as arr
import arcpy
import xlrd


value=arcpy.GetParameterAsText(0)
wb = xlrd.open_workbook("tanhui.xlsx")
sheet = wb.sheet_by_index(0)
rows = sheet.nrows
columns = sheet.ncols


# inputfc=arcpy.GetParameterAsText(0)
# wb=xlrd.open_workbook(intputfc)

def GM11(x, n):
    '''
    灰色预测
    x：序列，numpy对象
    n:需要往后预测的个数
    '''
    x1 = x.cumsum()  # 一次累加
    z1 = (x1[:len(x1) - 1] + x1[1:]) / 2.0  # 紧邻均值
    z1 = z1.reshape((len(z1), 1))
    B = np.append(-z1, np.ones_like(z1), axis=1)
    Y = x[1:].reshape((len(x) - 1, 1))
    # a为发展系数 b为灰色作用量
    [[a], [b]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Y)  # 计算参数
    result = (x[0] - b / a) * np.exp(-a * (n - 1)) - (x[0] - b / a) * np.exp(-a * (n - 2))  # 预测方程
    S1_2 = x.var()  # 原序列方差
    e = list()  # 残差序列
    for index in range(1, x.shape[0] + 1):
        predict = (x[0] - b / a) * np.exp(-a * (index - 1)) - (x[0] - b / a) * np.exp(-a * (index - 2))
        e.append(x[index - 1] - predict)
        print(predict)  # 预测值
    S2_2 = np.array(e).var()  # 残差方差
    C = S2_2 / S1_2  # 后验差比
    if C <= 0.35:
        assess = '后验差比<=0.35，模型精度等级为好'
    elif C <= 0.5:
        assess = '后验差比<=0.5，模型精度等级为合格'
    elif C <= 0.65:
        assess = '后验差比<=0.65，模型精度等级为勉强'
    else:
        assess = '后验差比>0.65，模型精度等级为不合格'
    # 预测数据
    predict = list()
    for index in range(x.shape[0] + 1, x.shape[0] + n + 1):
        predict.append((x[0] - b / a) * np.exp(-a * (index - 1)) - (x[0] - b / a) * np.exp(-a * (index - 2)))
        # print((x[0]-b/a)*np.exp(-a*(index-1)))
        # print((x[0]-b/a)*np.exp(-a*(index-2)))
    predict = np.array(predict)

    return {
        'a': {'value': a, 'desc': '发展系数'},
        'b': {'value': b, 'desc': '灰色作用量'},
        'predict': {'value': result, 'desc': '第%d个预测值' % n},
        'C': {'value': C, 'desc': assess},
        'predict': {'value': predict, 'desc': '往后预测%d个的序列' % (n)},
    }


data = arr.array('f', [])
head = []
a=0
# if __name__="__main__":
# head=str(sheet.row_values(0))
for i in range(0,rows):
    head.append(str(sheet.cell(i, 0)))
    #print(head[i])
for j in range(1,rows):
    city=head[j][6:].strip('\'')
    #print(city)
    if value==city:
        a=j
for k in range(1,7):
    idata = sheet.cell(a, k).value
    data.append(idata)
datas = np.array(data)
print(datas)
x = datas[0:5]  # 输入数据
y = datas[0:6]  # 需要预测的数据
result = GM11(x, len(y))
predict = result['predict']['value']
predict = np.round(predict, 1)
arcpy.AddMessage(result)
print('真实值:', y)
print('预测值:', predict)
print(result)


# 作图
x1 = np.array([1995, 2000, 2005, 2010, 2015,2020])
y1 = np.array(y)
x2 = np.array([2025, 2030, 2035, 2040, 2045, 2050])
y2 = np.array(predict)
plt.plot(x1, y1, 'y*-', label='true value')  # 真实值
plt.plot(x2, y2, 'b+-', label='predicted value')  # 预测值
plt.xlabel('year')
plt.ylabel('value')
plt.title('tu')
plt.legend()
plt.plot()
plt.show()
