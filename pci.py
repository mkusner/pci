import numpy as np
import pdb
import os
import math
import random
import time
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel
import pandas as pd
import scipy.io as io


# from this: http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

# spearman's rho
def spearmans_rho(a, b):
    sort_a = np.argsort(a)
    sort_b = np.argsort(b)
    ordering_a = np.argsort(sort_a)
    ordering_b = np.argsort(sort_b)
    d = (ordering_a - ordering_b)**2
    m = float(len(a))
    s_rho = np.abs(1.0 - ((6*np.sum(d))/(m*(m**2 - 1.0))))
    return s_rho

# kendall's tau
def kendalls_tau(a, b):
    m = len(a)
    C = 0.0
    D = 0.0
    for i in range(m):
        for j in range(i):
            ai = a[i] 
            bi = b[i]
            aj = a[j]
            bj = b[j]
            if ai > aj:
                if bi > bj:
                    C += 1.0
                else:
                    D += 1.0
            else:
                if bi > bj:
                    D += 1.0
                else:
                    C += 1.0
    k_tau = np.abs(C - D)/(0.5*m*(m - 1.0))
    return k_tau

# hsic
def hsic(a, b):
    m = len(a)
    a = a.reshape(m,1)
    b = b.reshape(m,1)
    K = rbf_kernel(a)
    L = rbf_kernel(b)
    H = np.eye(m) - (1.0/m)
    score = (1.0/(m - 1)**2)*np.trace(np.dot(np.dot(np.dot(K, H), L), H))
    return score

def compute_residuals(xtr, xte, ytr, yte, alpha):
    m = len(xtr)
    xtr = xtr.reshape(m,1)
    xte = xte.reshape(m,1)
    ytr = ytr.reshape(m,1)
    yte = yte.reshape(m,1)
    # train model
    clf = KernelRidge(alpha=alpha, kernel='rbf', gamma=0.0001)
    f = clf.fit(xtr, ytr)
    g = clf.fit(ytr, xtr)
    r_y = yte - g.predict(xte)
    r_x = xte - f.predict(yte)
    return r_y.reshape(m,), r_x.reshape(m,)

if __name__ == '__main__':
    # load some data here (and scale to be in [-1,1])
    # here's a simple example
    m = 1000
    X = (np.random.rand(m)*2.0) - 1.0 # in [-1,1]
    Y = X**4
    # labels
    # 1  is X->Y
    # -1 is Y->X
    label = 1

    # select dependence score
    scores = ["spearman", "kendall", "hsic"]

    
    trials = 10
    eps = np.logspace(0.1,10,20)/2
    priv_results = np.zeros((len(scores), trials, len(eps)))
    results      = np.zeros((len(scores), trials))
    for i in range(len(scores)):
        dep_score = scores[i]

        # dependence scores
        if dep_score == "spearman":
            s = lambda x, y: spearmans_rho(x,y)
            sense = (30.0/m)
        elif dep_score == "kendall":
            s = lambda x, y: kendalls_tau(x,y)
            sense = (4.0/m)
        elif dep_score == "hsic":
            s = lambda x, y: hsic(x,y)
            sense = (12.0*m - 11)/(m - 1.0)**2
        else:
            print('nope, this score is not currently implemented')
            pdb.set_trace()

        # split train/test
        for j in range(trials):
            shuffle = np.random.permutation(m)
            xtr = X[shuffle[:m//2]]
            xte = X[shuffle[m//2:]]
            ytr = Y[shuffle[:m//2]]
            yte = Y[shuffle[m//2:]]
            r_y, r_x = compute_residuals(xtr, xte, ytr, yte, 1.0)
            s_xy = s(xte, r_y)
            s_yx = s(yte, r_x)

            # check if we get the direction right
            if s_xy < s_yx:
                pred = 1
                gamma = s_yx - s_xy
            else:
                pred = -1
                gamma = s_xy - s_yx
            correct = (pred == label)
            results[i,j] = float(correct)
            if not correct:
                print('wrong')
                priv_results[i,j,:] = np.nan
            else:
                for e in range(len(eps)):
                    sig = sense/eps[e]
                    prob_correct = 1.0 - (np.exp(-gamma/sig)*(gamma + 2*sig)/(4*sig))
                    priv_results[i,j,e] = prob_correct
    pdb.set_trace()


