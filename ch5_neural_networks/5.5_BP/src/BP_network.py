# -*- coding: utf-8 -*
    
'''''
@author: PY131
'''''

'''
import data and pre-analysis through data visualization
'''
# using pandas dataframe for .csv read which contains chinese char.
import pandas as pd
import io
data_file_encode = "gb18030"  # the watermelon_3.csv codec type (Chinese characters)
with io.open("../data/watermelon_3.csv", mode = 'r', encoding = data_file_encode) as data_file:
    df = pd.read_csv(data_file)

print type(df), "\n", df
'''
# using seaborn for data visualization.
# load chinese font
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set(style="whitegrid", color_codes=True)
mpl.rcParams['font.sans-serif'] = ['Droid Sans Fallback']  # for chinese chararter visualization
mpl.rcParams['axes.unicode_minus'] = False 
sns.set_context("poster")
 
f1 = plt.figure(1)
sns.FacetGrid(df, hue="好瓜", size=5).map(plt.scatter, "密度", "含糖率").add_legend() 
sns.plt.show()
 
f2 = plt.figure(2)
sns.plt.subplot(221)
sns.swarmplot(x = "纹理", y = '密度', hue = "好瓜", data = df)
sns.plt.subplot(222)
sns.swarmplot(x = "敲声", y = '密度', hue = "好瓜", data = df)
sns.plt.subplot(223)
sns.swarmplot(x = "色泽", y = '含糖率', hue = "好瓜", data = df)
sns.plt.subplot(224)
sns.swarmplot(x = "敲声", y = '含糖率', hue = "好瓜", data = df)
sns.plt.show()    
'''
"""
另一种常用于统计建模或机器学习的转换方式是：
将分类变量（categorical variable）转换为
“哑变量矩阵”（dummy matrix）或“指标矩阵”（indicator matrix）。
如果DataFrame的某一列中含有k个不同的值，则可以派生出一个k列矩阵或DataFrame（其值全为1和0）。
pandas有一个get_dummies函数可以实现该功能
"""
# one-hot encoding  
wm_df = pd.get_dummies(df)
# print "wm_df: \n", wm_df
X = wm_df[wm_df.columns[1:-2]]  # input
# print "wm_df.columns: \n", wm_df.columns
# print "wm_df.columns[1:-2]: \n", wm_df.columns[1:-2]
# print "X:\n", X
# print "wm_df.columns[-2:]: \n", wm_df.columns[-2:]
Y = wm_df[wm_df.columns[-2:]]  # output
# print "Y: \n", Y
label = wm_df.columns._data[-2:] # class label
# print wm_df.columns._data
# construction of data in pybrain's formation
from pybrain.datasets import ClassificationDataSet
ds = ClassificationDataSet(19, 1, nb_classes=2, class_labels=label)
print "ds: \n", ds
for i in range(len(Y)): 
    y = 0
    if Y['好瓜_是'][i] == 1: y = 1
    ds.appendLinked(X.values[i], y)
    print X.values[i]
print "ds before calculateStatistics: \n", ds
ds.calculateStatistics()
print "ds after calculateStatistics: \n", ds

# generation of train set and test set (3:1)
tstdata_temp, trndata_temp = ds.splitWithProportion(0.25)  
tstdata = ClassificationDataSet(19, 1, nb_classes=2, class_labels=label)
for n in range(0, tstdata_temp.getLength()):
    tstdata.appendLinked( tstdata_temp.getSample(n)[0], tstdata_temp.getSample(n)[1] )

trndata = ClassificationDataSet(19, 1, nb_classes=2, class_labels=label)
for n in range(0, trndata_temp.getLength()):
    trndata.appendLinked( trndata_temp.getSample(n)[0], trndata_temp.getSample(n)[1] )

trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()

'''
implementation of BP network
'''    
from pybrain.tools.shortcuts import buildNetwork  # for building network raw model
from pybrain.structure import SoftmaxLayer  # for output layer activation function
from pybrain.supervised.trainers import BackpropTrainer  # for model trainer

# network structure
n_h = 5 # hidden layer nodes number
net = buildNetwork(19, n_h, 2, outclass = SoftmaxLayer)  

# 1.1 model training, using standard BP algorithm
trainer = BackpropTrainer(net, trndata)
trainer.trainEpochs(1) # training for once

# 1.2 model training, using accumulative BP algorithm
# trainer = BackpropTrainer(net, trndata, batchlearning=True)
# trainer.trainEpochs(50)
# err_train, err_valid = trainer.trainUntilConvergence(maxEpochs=50)

# convergence curve for accumulative BP algorithm process
# import matplotlib.pyplot as plt
# plt.plot(err_train,'b',err_valid,'r')
# plt.title('BP network classification')  
# plt.ylabel('accuracy')  
# plt.xlabel('epochs')  
# plt.show()

# 1.3 model testing
from pybrain.utilities import percentError
tstresult = percentError( trainer.testOnClassData(tstdata), tstdata['class'] )
print("epoch: %4d" % trainer.totalepochs, " test error: %5.2f%%" % tstresult)

er_sum = 0;
for i in range(20):
    # generation of train set and test set (3:1)
    tstdata_temp, trndata_temp = ds.splitWithProportion(0.25)  
    tstdata = ClassificationDataSet(19, 1, nb_classes=2, class_labels=label)
    for n in range(0, tstdata_temp.getLength()):
        tstdata.appendLinked( tstdata_temp.getSample(n)[0], tstdata_temp.getSample(n)[1] )
    trndata = ClassificationDataSet(19, 1, nb_classes=2, class_labels=label)
    for n in range(0, trndata_temp.getLength()):
        trndata.appendLinked( trndata_temp.getSample(n)[0], trndata_temp.getSample(n)[1] )
    trndata._convertToOneOfMany()
    tstdata._convertToOneOfMany() 
    # network structure
    n_h = 10 # hidden layer nodes number
    net = buildNetwork(19, n_h, 2, outclass = SoftmaxLayer)   
    
    # model training, using standard BP algorithm
    trainer = BackpropTrainer(net, trndata)
    trainer.trainEpochs(1) # training for once
    
    # 1.2 model training, using accumulative BP algorithm
#     trainer = BackpropTrainer(net, trndata, batchlearning=True)
#     trainer.trainEpochs(10)

    tstresult = percentError( trainer.testOnClassData(tstdata), tstdata['class'] )
    # print result
    print("%5.2f%%  " % tstresult)
    er_sum += tstresult
print("\naverage error rate: %5.2f%%" % (er_sum/20))
