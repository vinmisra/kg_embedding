import numpy as np
from scipy.stats import rv_discrete
from math import log
import matplotlib.pyplot as plt
from est_entro import est_entro_JVHW
from est_entro import est_entro_MLE
from entropy_true import entropy_true

if __name__ == '__main__':
    C = 1
    num = 15
    mc_times = 20
    record_S = np.ceil(np.logspace(2, 5, num))
    record_n = np.ceil(C*record_S/np.log(record_S))
    true_S = np.array([])
    JVHW_S = np.array([])
    MLE_S = np.array([])
    twonum = np.random.rand(2, 1)
    for i in range(num):
        S = record_S[i]
        n = record_n[i]
        dist = np.random.beta(twonum[0], twonum[1], S)
        dist = dist / sum(dist)
        true_S = np.append(true_S, entropy_true(dist))
        record_H = np.zeros(mc_times)
        record_MLE = np.zeros(mc_times)
        for mc in range(mc_times):
            samp = rv_discrete(values=(np.arange(S), dist)).rvs(size=n)
            record_MLE[mc] = est_entro_MLE(samp)
            record_H[mc] = est_entro_JVHW(samp)
        JVHW_S = np.append(JVHW_S, np.mean(abs(record_H - true_S[-1])))
        MLE_S = np.append(MLE_S, np.mean(abs(record_MLE - true_S[-1])))
    
    plot_JVHW, = plt.plot(record_S/record_n, JVHW_S, 'b-s', linewidth=2.0, label='JVHW estimator')
    plot_MLE, = plt.plot(record_S/record_n, MLE_S, 'r-.o', linewidth=2.0, label = 'MLE')
    plt.xlabel('S/n')
    plt.ylabel('Mean Absolute Error')
    plt.legend(handles=[plot_JVHW, plot_MLE], loc=2)
    plt.show()