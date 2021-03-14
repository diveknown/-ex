import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
A = np.eye(5)
A
#python不能识别反斜杠，只能用正斜杠读文件
path = 'C:/Users/guini/Desktop/todo/Andrew-NG-Meachine-Learning-master\Andrew-NG-Meachine-Learning-master\machine-learning-ex1\machine-learning-ex1\ex1\ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit']) #参数head表示第head行为head，从后一行开始读
data.head(5)      #后面加上.head(5)只显示前五行，不加.head()显示全部
data.describe()
#count：数量统计，此列共有多少有效值
#unipue：不同的值有多少个
#std：标准差（Standard Deviation）
#min：最小值
#25%：四分之一分位数
#50%：二分之一分位数
#75%：四分之三分位数
#max：最大值
#mean：均值
data.plot(kind='scatter', x='Population', y='Profit', figsize=(8,5))
plt.show()
def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) -  y), 2)            #numpy.power()的元素分别求n次方
    return np.sum(inner) / (2 * len(X))
#计算代价函数，除以2m而不是m，是因为方便后续求导，实际上并不影响
data.insert(0, 'Ones', 1)
data
#让我们在训练集中添加一列，以便我们可以使用向量化的解决方案来计算代价和梯度
# set X (training data) and y (target variable)
cols = data.shape[1]  # 列数
X = data.iloc[:,0:cols-1]  # 取前cols-1列，即输入向量
y = data.iloc[:,cols-1:cols] # 取最后一列，即目标向量
print(X,"\n",y)
type(X)
#pandas.iloc()：从pandas.core.frame.DataFrame这个数据类型，提取出子pandas.core.frame.DataFrame
X = np.matrix(X.values)        #X从pandas的统计数据类型转化为numpy的数组
y = np.matrix(y.values)
theta = np.matrix([0,0])
print(theta)
X.shape, theta.shape, y.shape
# ((97, 2), (1, 2), (97, 1))
computeCost(X, y, theta) # 32.072733877455676
count = np.sum(np.power(y,2))/194
print(count)
temp = np.matrix(np.zeros(theta.shape))
np.zeros(theta.shape)
def gradientDescent(X, y, theta, alpha, epoch):
    """reuturn theta, cost"""
    
    temp = np.matrix(np.zeros(theta.shape))  # 初始化一个 θ 临时矩阵(1, 2)
    parameters = int(theta.flatten().shape[1])  # 参数 θ的数量          numpy.flatten()将theta展开为一维数组，shape[1]列数，即为个数
    cost = np.zeros(epoch)  # 初始化一个ndarray，包含每次epoch的cost
    m = X.shape[0]  # 样本数量m
    
    for i in range(epoch):
        # 利用向量化一步求解
        temp =theta - (alpha / m) * (X * theta.T - y).T * X
        
# 以下是不用Vectorization求解梯度下降
#         error = (X * theta.T) - y  # (97, 1)
        
#         for j in range(parameters):
#             term = np.multiply(error, X[:,j])  # (97, 1)
#             temp[0,j] = theta[0,j] - ((alpha / m) * np.sum(term))  # (1,1)
            
        theta = temp
        cost[i] = computeCost(X, y, theta)
        
    return theta, cost
alpha = 0.01
epoch = 1000
#初始化一些附加变量 - 学习速率α和要执行的迭代次数
final_theta, cost = gradientDescent(X, y, theta, alpha, epoch)
cost[999]
computeCost(X, y, final_theta)
x = np.linspace(data.Population.min(), data.Population.max(), 100)  # 横坐标
f = final_theta[0, 0] + (final_theta[0, 1] * x)  # 纵坐标，利润

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data['Population'], data.Profit, label='Traning Data')
ax.legend(loc=2)  # 2表示在左上角
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(np.arange(epoch), cost, 'r')  # np.arange()返回等差数组
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()
 
