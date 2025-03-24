import numpy as np
import methods
import pandas as pd
from joblib import Parallel, delayed


def ComputeTrueLoss(n_front, d, S_0, K, mu, sigma, r, tau, T, alpha=0.1):

    rho = 0.3
    cor_mat = methods.generate_cor_mat(d, rho)
    cov_mat_call = cor_mat * sigma ** 2

    print("European Call: Simulating Front Paths...")
    sample_outer_call = methods.GBM_front(n_front=n_front, d=d, S_0=S_0,
                                          drift_vec=mu, diffusion_mat=cov_mat_call, tau=tau)

    print("Calculating Value...")
    call_tau = np.zeros(n_front)
    for j in range(len(K)):
        call_tau_vec = Parallel(n_jobs=d, verbose=10)(
            delayed(methods.price_CP)(sample_outer_call[k, :], T - tau, sigma, r, K[j], 0, "C", "long")
            for k in range(d))
        call_tau = call_tau + np.sum(call_tau_vec, axis=0)
    print("End of European Call.")

    portfolio_value_0 = 0

    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_CP(S_0, T, sigma, r, K[j], 0, "C", "long")

    loss_true = d * portfolio_value_0 - call_tau

    loss_true.sort()

    L0 = loss_true[int(np.ceil((1 - alpha) * n_front))]

    indicator_true = alpha
    hockey_true = np.mean(np.maximum(loss_true - L0, 0))
    quadratic_true = np.mean((loss_true - L0) ** 2)
    CVaR = np.mean(loss_true[loss_true > L0])

    return L0, indicator_true, hockey_true, quadratic_true, CVaR


d = 2
sigma = 0.1
n_cmc = 10**7
S_0 = 100
K = [90, 100, 110]
mu = 0.08
r = 0.05
tau = 3/50
T = 1

L0, indicator_true, hockey_true, quadratic_true, CVaR = ComputeTrueLoss(n_front=n_cmc,
                                                                                       d=d,
                                                                                       S_0=S_0,
                                                                                       K=K,
                                                                                       mu=mu,
                                                                                       sigma=sigma,
                                                                                       r=r,
                                                                                       tau=tau,
                                                                                       T=T,
                                                                                       alpha=0.1)

df = pd.DataFrame([indicator_true, hockey_true, quadratic_true, L0, CVaR],
                  index=["Indicator",
                         "Hockey",
                         "Quadratic",
                         "VaR",
                         "CVaR"],
                  columns=[n_cmc]).T
print(df)
df.to_csv("./trueValue_2.csv")
