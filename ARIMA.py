# coding=gbk
# ʹ��ARIMA ģ�ͶԷ�ƽ��ʱ�����м��Խ�ģ����
# ����������ǿ���ȷ���Ե���Ϣ��ȡ������ ����ƽ�ȵ����в�ֺ���ʾ��ƽ�����е����ʣ� ���ǳ������ƽ������Ϊ���ƽ�����С�
# �Բ��ƽ�����п��Ի���ҪARMA ģ�ͽ�����ϣ� ARIMA ģ�͵�ʵ�ʾ��ǲ��Ԥ���� ARMA ģ�͵Ľ�ϡ�


# ��������
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# ��������
cityname = input("____")
filename = cityname + '.xlsx'

# ����pandas���ṩ��read_excel����ֱ�ӷ���dataframe��ʽ������
# ��xlrd�з��ص���ԭʼ���� ���ʺ��ڶ�excel�е����ݽ��ж��ƻ����������ײ�Ĳ�����
# index_col��ָ����Ϊ���������У����������������������б�
data = pd.read_excel(filename, index_col=u'year')

# ��������excelʱ��ͼ ���û�ͼ��һЩ�ַ���������
plt.rcParams['font.sans-serif'] = ['SimHei']  # ����ʹ��������ʾ�����������
plt.rcParams['axes.unicode_minus'] = False  # ����������ʾ��ʾ����
data.plot()
plt.show()

# ƽ���Լ��
print('ԭʼ���еļ�����Ϊ��', adfuller(data[u'tanhui']))
# ԭʼ���еļ�����Ϊ�� (1.8137710150945268, 0.9983759421514264, 10, 26, {'1%': -3.7112123008648155,
#  '10%': -2.6300945562130176, '5%': -2.981246804733728}, 299.46989866024177) ˵���˴�Ϊ��ƽ������
# ����ֵ����Ϊ��adf, pvalue pֵ�� usedlag, nobs, critical values�ٽ�ֵ , icbest, regresults, resstore
# adf �ֱ����3�в�ͬ����ˮƽ��3���ٽ�ֵ����λ���ͳ������Ӧ��p ֵ�������� 0.05 �� ˵�����п����ж�Ϊ ��ƽ������


# �����ݽ��в�ֺ�ɾ��ȱʧֵ�õ� �����ͼ�� ƫ���ͼ
D_data = data.diff(1).dropna()
D_data.columns = [u'̼����']
D_data.plot()  # ������ֺ��ʱ��ͼ
plt.show()
plot_acf(D_data)  # ���������ͼ
plt.show()
plot_pacf(D_data, lags=1)  # ����ƫ���ͼ
plt.show()
print(u'������е�ADF ������Ϊ�� ', adfuller(D_data[u'̼����']))
# ƽ���Լ��飬��λ������
# ������е�ADF ������Ϊ��  (-3.1560562366723537, 0.022673435440048798, 0, 35, {'1%': -3.6327426647230316,
# '10%': -2.6130173469387756, '5%': -2.9485102040816327}, 287.5909090780334)
# һ�ײ�ֺ�����е�ʱ��ͼ�ھ�ֵ�����Ƚ�ƽ�ȵĲ����� ��������к�ǿ�Ķ�������ԣ� ��λ������ pֵС�� 0.05 ������˵һ�ײ�ֺ��������ƽ������


# ��һ�ײ�ֺ������������������
print(u'������еİ�������������', acorr_ljungbox(D_data, lags=1))  # ����ͳ������ p ֵ
# ������еİ������������� (array([11.30402222]), array([0.00077339])) pֵΪ�ڶ�� ԶС�� 0.05

# �ҵ�����ģ�Ͷ��ڵ�pqֵ
# ��ʼ��һ��BIC����
# ͨ����� ARIMA ģ������ȡ����ģ�ͣ��Ա������ʱ�����з�����Ԥ��
# fit() �����Թ����� ARIMA ģ�ͽ�����ϣ�������ģ�͵Ĳ���
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

bic_matrix = pd.DataFrame(bic_matrix)  # ����ת����Dataframe ���ݽṹ
# �ڱ�Ҷ˹�������ҵ���Сֵ
p, q = bic_matrix.stack().astype('float64').idxmin()  # ��ʹ��stack չ*��ö༶������ Ȼ��ʹ�� idxmin �ҳ���Сֵ��λ��
print(u'BIC ��С��pֵ �� q ֵ��%s,%s' % (p, q))  # BIC ��С��pֵ �� q ֵ��0,1
# ���Կ��Խ���ARIMA ģ�ͣ�ARIMA(0,1,1)
# arima_model=sm.tsa.ARIMA()
model = sm.tsa.ARIMA(data, order=(p, 1, q))
results = model.fit()
predict_sunspots = results.predict()  # In-sample prediction and out-of-sample forecasting
predict_sunspots_2 = results.forecast(5)  # Out-of-sample forecastsԤ��
print('The result of the predicton,next 1:', predict_sunspots_2)

print("predict past data", predict_sunspots)
x = [2025, 2030, 2035, 2040, 2045]
y = []
for i in range(6, 11):
    y.append(predict_sunspots_2[i])
print(y)
