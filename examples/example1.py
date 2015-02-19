import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pdastf.pdas.trendfilter import TF, TFsafe


# 1: Plot for segmenting constant-shaped time series
mp = pd.read_csv('/home/zhh210/workspace/seg/data/real/merged_22segs.csv')
mp = 0.1*mp['AI-253']
fig = plt.figure(figsize=(8,2))
ax = fig.add_subplot(111)
ax.plot(mp,'b-',label='observed')
prob = TF(np.array(mp)[:,np.newaxis],3000,order=1,mode = -1)
prob.pdas()
prob.plot('it_const.pdf')
ax.plot(prob.x,'r-',label='fitted')
plt.xlabel('Time Index', size = 18)
plt.ylabel('Value', size = 18)
ax.legend(prop={'size':6})
fig.savefig('merged-const.png',bbox_inches='tight')

# 2: Plot for segmenting linear-shaped time series
mp = pd.read_csv('/home/zhh210/workspace/seg/data/real/merged_22segs.csv')
mp = mp['AI-274']
fig = plt.figure(figsize=(8,2))
ax = fig.add_subplot(111)
ax.plot(mp,'b-',label='observed')
weights = np.ones(mp.size)
prob = TF(np.array(mp)[:,np.newaxis],1000,order=2,mode = -1)
prob.pdas()
prob.plot('it_linear.pdf')
ax.plot(prob.x,'r-',label='fitted')
plt.xlabel('Time Index', size = 18)
plt.ylabel('Value', size = 18)
ax.legend(prop={'size':6})
fig.savefig('merged-linear.png',bbox_inches='tight')


# 3: Plot for global temperature, isotonic regression
data = pd.read_table('temperature',delim_whitespace=True)
x = np.array(data['Year'],dtype='float')
y = np.array(data['Annual_Mean'],dtype='float')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x,y,'bo-',label='observed')
weights = np.ones(max(y.shape))
prob = TF(y[:,np.newaxis],3000,order=1,mode = 0)
prob.pdas()
ax.plot(x,prob.x,'r-',label='fitted')
plt.xlabel('Year',size = 8)
plt.ylabel('Temperature Index (C) (Anomaly with Base: 1951-1980)',size = 8)
ax.legend(prop={'size':6})
fig.savefig('neariso-temp.png')


# 4: Plot for synthetic data, almost convex fitting
x = np.arange(1,31)
np.random.seed(1)
y = np.square(x - 15) + 10*np.random.randn(len(x))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x,y,'bo-',label='observed')
weights = np.ones(max(y.shape))
prob = TF(y[:,np.newaxis],10,order=2,mode = 0)
prob.pdas()
ax.plot(x,prob.x,'r-',label='fitted')
plt.xlabel('$x$',size = 8)
plt.ylabel('$y$',size = 8)
ax.legend(prop={'size':6})
fig.savefig('nearconv.png')


