import numpy as np
import pandas as pd
import methods
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
import time


def reg_Euro_Call(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T):

    cor_mat = np.ones((d, d)) * 0.3 + np.identity(d) * 0.7
    cov_mat = cor_mat * sigma**2

    start = time.time()
    sample_outer_train = methods.GBM_front(n_front = n_front, d = d, S_0 = S_0,
                                           drift_vec = mu, diffusion_mat = cov_mat, tau = tau)
    sample_inner = methods.GBM_back(n_front = n_front, n_back = n_back, d = d, S_tau = sample_outer_train,
                                    drift = r, diffusion = cov_mat, T = T - tau)

    payoff = np.zeros([d, n_front])
    for j in range(len(K)):
        price = np.mean(np.maximum(sample_inner - K[j], 0), axis=2) * np.exp(-r * (T - tau))
        payoff = payoff + price

    portfolio_value_0 = 0
    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_CP(S_0, T, sigma, r, K[j], 0, "C", "long")

    X_train = methods.generate_basis(sample_outer_train)
    y_train = np.sum(payoff, axis=0)

    reg = LinearRegression().fit(X_train, y_train)

    sample_outer_test = methods.GBM_front(n_front=n_front, d=d, S_0=S_0,
                                          drift_vec=mu, diffusion_mat=cov_mat, tau=tau)

    X_test = methods.generate_basis(sample_outer_test)
    y_test = reg.predict(X_test)

    loss_reg = d * portfolio_value_0 - y_test

    total_time = time.time() - start

    return loss_reg, total_time


def cal_RRMSE(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, alpha, L0):

    loss, total_time = reg_Euro_Call(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T)

    # indicator = np.mean((loss > L0))
    # hockey = np.mean(np.maximum(loss - L0, 0))
    # quadratic = np.mean((loss - L0) ** 2)

    # loss.sort()
    # VaR = loss[int(np.ceil((1 - alpha) * n_front))]
    # CVaR = np.mean(loss[loss >= VaR])

    return total_time


d = 2
sigma = 0.1
S_0 = 100
K = [90, 100, 110]
mu = 0.08
r = 0.05
tau = 3/50
T = 1
alpha = 0.1

result_true = np.array(pd.read_csv(f"./trueValue_{d}.csv")).flatten()[1:]

L0 = result_true[3]
# exponential growth: 1000, 2000, 4000, 8000, 16000
n_front_list = [1000, 2000, 4000, 8000, 16000, 32000]
n_front_vec = np.array(n_front_list)
n_trials = len(n_front_list)
n_back_vec = [1] * n_trials

result_table = []

n_rep = 10

for _j in range(n_trials):

    n_front_11 = n_front_vec[_j]
    n_back_11 = n_back_vec[_j]

    time_cv = 0
    res = Parallel(n_jobs=20, verbose=10)(delayed(cal_RRMSE)(n_front_11, n_back_11, d, S_0, K,
                                                         mu, sigma, r, tau, T,
                                                         alpha, L0)
                                          for _n in range(n_rep))
    res = np.array(res)

    total_time = np.mean(res)

    result_table.append(total_time)

    print("Regression Estimation Done for:", n_front_11, "x", n_back_11)
    print(total_time)

print(result_table)
np.save(f"./result_2_time_reg.npy", result_table)
