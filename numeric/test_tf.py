import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from pdastf.pdas.trendfilter import TF, TFsafe



def plot(x,collector):
    'Generate plot on tested results'
    plt.figure(figsize = (6,6))
    plt.title('Trend of indicators')
    plt.subplot(2,1,1)
    plt.plot(x,collector['iterations']['pdas'],'ro-',label='pdas')
    plt.plot(x,collector['iterations']['safe1'],'bo-',label='safe-1')
    plt.plot(x,collector['iterations']['safe2'],'go-',label='safe-2')
    plt.ylabel('Iteration')
    plt.xlabel('Size')
    plt.legend(prop={'size':6})

    plt.subplot(2,1,2)
    plt.plot(x,collector['times']['pdas'],'ro-',label='pdas')
    plt.plot(x,collector['times']['safe1'],'bo-',label='safe-1')
    plt.plot(x,collector['times']['safe2'],'go-',label='safe-2')
    plt.ylabel('Time')
    plt.xlabel('Size')
    plt.legend(prop={'size':6})
    plt.savefig('test_randomtf.png')


if __name__=='__main__':
    
    size = 36

    seed_partition = 0
    seed_data = 0

    np.random.seed(seed_data)

    iterations = {'pdas':[],'safe1':[],'safe2':[]}
    times = {'pdas':[],'safe1':[],'safe2':[]}

    for i in range(1,size):
        # Generate data
        dat = np.arange(i * 1e4).astype(np.float)
        dat += 2 * np.random.randn(i * 1e4)  # add noise

        # By pdas
        random.seed(seed_partition)
        prob = TF(dat[:,np.newaxis],1000,order=1,mode = 0)
        prob.pdas()
        iterations['pdas'].append(prob.info['iter'])
        times['pdas'].append(prob.info['time'])
        del prob

        # By pdas with safeguard
        random.seed(seed_partition)
        prob = TFsafe(dat[:,np.newaxis],1000,order=1,mode = 0)
        prob.pdas()
        iterations['safe1'].append(prob.info['iter'])
        times['safe1'].append(prob.info['time'])
        del prob

        # By pdas with safeguard 2
        random.seed(seed_partition)
        prob = TFsafe(dat[:,np.newaxis],1000,order=1,mode = 0)
        prob.pdas2()
        iterations['safe2'].append(prob.info['iter'])
        times['safe2'].append(prob.info['time'])
        del prob
    collector = {'iterations':iterations, 'times':times}
    plot([i for i in range(1,size)],collector)
