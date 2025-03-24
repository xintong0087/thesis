import numpy as np

import matplotlib.pyplot as plt


methods = ["snsbs", "reg", "knn", "lr", "krr"]
methods_name = ["Standard with Bootstrap", "Regression", "Kernel Smoothing", "Likelihood Ratio", "Kernel Ridge Regression"]
Gammas = [1000, 2000, 4000, 8000, 16000, 32000]

plt.figure(figsize=(8, 4.5))

for method, name in zip(methods, methods_name):
    data = np.load(f"./result_2_time_{method}.npy")

    plt.loglog(Gammas, data, label=name, marker="o", linestyle="--")

plt.legend(loc="upper left")
plt.xlabel("Simulation Budget $\Gamma$")
plt.ylabel("Time (s)")
plt.title("European Call - d = 2")
plt.savefig("time_comparison.png", dpi=300)











