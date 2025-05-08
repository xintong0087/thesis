import numpy as np
import pandas as pd
import methods
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process.kernels import Matern
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from skopt import BayesSearchCV
from skopt.space import Real
import time


def krr_Euro_Call(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, alpha_opt, l_opt, nu_opt):

    cor_mat = np.ones((d, d)) * 0.3 + np.identity(d) * 0.7
    cov_mat = cor_mat * sigma ** 2

    start = time.time()
    sample_outer_train = methods.GBM_front(n_front=n_front, d=d, S_0=S_0,
                                           drift_vec=mu, diffusion_mat=cov_mat, tau=tau)
    sample_inner = methods.GBM_back(n_front=n_front, n_back=n_back, d=d, S_tau=sample_outer_train,
                                    drift=r, diffusion=cov_mat, T=T - tau)

    payoff = np.zeros([d, n_front])
    for j in range(len(K)):
        price = np.mean(np.maximum(sample_inner - K[j], 0), axis=2) * np.exp(-r * (T - tau))
        payoff = payoff + price

    portfolio_value_0 = 0
    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_CP(S_0, T, sigma, r, K[j], 0, "C", "long")

    X_train = methods.generate_basis(sample_outer_train, option_type="Vanilla", basis_function="None")
    y_train = np.sum(payoff, axis=0)

    kernel = 1.0 * Matern(length_scale=l_opt, nu=nu_opt)
    krr = KernelRidge(alpha=alpha_opt, kernel=kernel).fit(X_train, y_train)

    sample_outer_test = methods.GBM_front(n_front=n_front, d=d, S_0=S_0,
                                          drift_vec=mu, diffusion_mat=cov_mat, tau=tau)

    X_test = methods.generate_basis(sample_outer_test, option_type="Vanilla", basis_function="None")

    y_test = krr.predict(X_test)

    loss_krr = d * portfolio_value_0 - y_test

    total_time = time.time() - start

    return loss_krr, total_time


def cv_krr(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T):
    cor_mat = np.ones((d, d)) * 0.3 + np.identity(d) * 0.7
    cov_mat = cor_mat * sigma ** 2

    sample_outer_train = methods.GBM_front(n_front=n_front, d=d, S_0=S_0,
                                           drift_vec=mu, diffusion_mat=cov_mat, tau=tau)
    sample_inner = methods.GBM_back(n_front=n_front, n_back=n_back, d=d, S_tau=sample_outer_train,
                                    drift=r, diffusion=cov_mat, T=T - tau)

    payoff = np.zeros([d, n_front])
    for j in range(len(K)):
        price = np.mean(np.maximum(sample_inner - K[j], 0), axis=2) * np.exp(-r * (T - tau))
        payoff = payoff + price

    portfolio_value_0 = 0
    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_CP(S_0, T, sigma, r, K[j], 0, "C", "long")

    X_train = methods.generate_basis(sample_outer_train, option_type="Vanilla", basis_function="None")
    y_train = np.sum(payoff, axis=0)

    param_distributions = {
        "alpha": Real(1e-5, 1e-1, "log-uniform"),
        "kernel__length_scale": Real(1e-3, 1e3, "log-uniform"),
        "kernel__nu": Real(5e-1, 5e0, "log-uniform"),
    }

    cv_kf = KFold(n_splits=5)

    bayesian_search = BayesSearchCV(estimator=KernelRidge(kernel=Matern()),
                                    search_spaces=param_distributions, n_jobs=20, cv=cv_kf)

    bayesian_search.fit(X_train, y_train)

    alpha = bayesian_search.best_params_["alpha"]
    l = bayesian_search.best_params_["kernel__length_scale"]
    nu = bayesian_search.best_params_["kernel__nu"]

    print("End of CV, optimal hyperparameter =", alpha, l, nu)

    # krr = KernelRidge(alpha = 1, kernel=Matern(length_scale=100, nu=5/2))
    # krr.fit(X_train, y_train)
    #
    # param_distributions = {
    #     "alpha": loguniform(1e-5, 1e-1),
    #     "kernel__length_scale": loguniform(1e-3, 1e3),
    #     "kernel__nu": loguniform(5e-1, 5e0),
    # }
    # krr_tuned = RandomizedSearchCV(
    #     krr,
    #     param_distributions=param_distributions,
    #     n_iter=500,
    #     random_state=22,
    # )
    #
    # krr_tuned.fit(X_train, y_train)
    # alpha = krr_tuned.best_params_["alpha"]
    # l = krr_tuned.best_params_["kernel__length_scale"]
    # nu = krr_tuned.best_params_["kernel__nu"]
    #
    # print("End of CV, optimal hyperparameter =", alpha, l, nu)

    return alpha, l, nu


def cal_RRMSE(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, alpha, L0, alpha_opt, l_opt, nu_opt):

    loss, total_time = krr_Euro_Call(n_front, n_back,
                                     d, S_0, K, mu, sigma, r, tau, T,
                                     alpha_opt, l_opt, nu_opt)

    # indicator = np.mean((loss > L0))
    # hockey = np.mean(np.maximum(loss - L0, 0))
    # quadratic = np.mean((loss - L0) ** 2)

    # loss.sort()
    # VaR = loss[int(np.ceil((1 - alpha) * n_front))]
    # CVaR = np.mean(loss[loss >= VaR])

    return total_time


np.random.seed(22)

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
n_front_list = [125, 250, 500, 1000, 2000, 4000]
n_front_vec = np.array(n_front_list)
n_trials = len(n_front_list)
n_back_vec = [1] * n_trials

result_table = []   

n_rep = 1

for _j in range(n_trials):

    n_front = n_front_vec[_j]
    n_back = n_back_vec[_j]

    start_cv = time.time()
    alpha_opt, l_opt, nu_opt = cv_krr(n_front, n_back,
                                               d, S_0, K, mu, sigma, r, tau, T)
    time_cv = time.time() - start_cv

    res = cal_RRMSE(n_front, n_back, d, S_0, K,
                     mu, sigma, r, tau, T,
                     alpha, L0, alpha_opt, l_opt, nu_opt)

    total_time = res

    result_table.append(total_time + time_cv)

    print("KRR Estimation Done for:", n_front, "x", n_back)
    print(total_time + time_cv)

print(result_table)
np.save(f"./result_{d}_time_krr.npy", result_table)
