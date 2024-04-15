#!/usr/bin/env python3
import sklearn
import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt
import tqdm

X, y = sklearn.datasets.load_digits(return_X_y=True)

#Arrange into chunks with one example of each class
one_hot = y[:,np.newaxis]==np.arange(10)
occurrence = np.choose(np.round(y),one_hot.cumsum(axis=0).T)
idx = np.lexsort((y,occurrence))
X, y = X[idx], y[idx]
min_freq = one_hot.sum(axis=0).min()
X, y = X[:10*min_freq], y[:10*min_freq]

#One-hot encoding
y = (y[:,np.newaxis]==np.arange(10))*1.0

def standardise(x, axis=None):
    return (x-x.mean(axis=axis)) * np.nan_to_num(1/x.std(axis=axis))

def one_hot(key, idx):
    X_ = X[idx]
    y_ = y[idx,key]
    return [standardise(x,axis=0) for x in (X_,y_)]

def get_beta(X, y, penalty = None):
    XTX = np.cov(X.T)
    XTy = np.cov(X.T, y.reshape(1,-1))[-1,:-1]

    if penalty is None:
        penalty = np.zeros_like(XTX)

    try:
        return np.linalg.solve(XTX + penalty, XTy)
    except np.linalg.LinAlgError:
        return np.nan

def loocv_mse(X, y, penalty = None):
    def loocv_residual(i):
        X_train = np.delete(X,i,axis=0)
        y_train = np.delete(y,i)
        return y[i] - np.dot(X[i],get_beta(X_train,y_train,penalty))
    return np.mean([loocv_residual(i)**2 for i in range(len(y))])

if __name__ == '__main__':
    fig, ax = plt.subplots(5,2, sharex=True, sharey=True)
    fig.suptitle('MSE from LOOCV')

    chunks = [slice(30*i,30*(i+1)) for i in range(len(y)//30)]
    penalty_strengths = np.linspace(0,1,5)

    for key in range(10):
        print(key)
        f = ax[key//2,key%2]

        """
        ols = loocv_mse(*one_hot(key))
        f.hlines(ols, penalty_strengths[0], penalty_strengths[-1], label='OLS')
        """

        print('Ridge')
        ridge_penalty = np.eye(64)
        ridge = [np.mean([loocv_mse(*one_hot(key, idx), strength*ridge_penalty) for idx in chunks]) for strength in tqdm.tqdm(penalty_strengths)]
        f.plot(penalty_strengths, ridge, label='Ridge')

        for lengthscale in (0.5,0.75,1):
            print(f'Off-diagonal Penalty, {lengthscale=}')
            offdiagonal_penalty = np.array([[np.exp(-((x1-x2)**2+(y1-y2)**2)/lengthscale**2) for x1 in range(8) for y1 in range(8)] for x2 in range(8) for y2 in range(8)])
            offdiagonal = [np.mean([loocv_mse(*one_hot(key, idx), strength*offdiagonal_penalty) for idx in chunks]) for strength in tqdm.tqdm(penalty_strengths)]
            f.plot(penalty_strengths, offdiagonal, label=f'Non-Orthogonal Quadratic Penalty ({lengthscale=})')

        f.set_xlabel('Penalty Strength')
        f.set_ylabel('MSE')

        f.set_yscale('log')

        f.set_title(f'{key}')

        f.legend()

    fig.set_size_inches(20, 20)
    fig.savefig('output/mse_loocv.png')
