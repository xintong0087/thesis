# Pareto random variable

Consider a random variable $(\rho_\Gamma-\rho)$  that follows a Pareto distribution with parameters $\Gamma^{-1}$ and $\alpha$.

Its survival function is given by:

$$
\frac{1}{(x \Gamma)^\alpha} \; \text{for} \; x \geq \Gamma^{-1}
$$

The variance of the random variable is given by:

$$
\Gamma^{-2} \frac{\alpha^2}{(\alpha-1)^2(\alpha-2)}
$$

Let's state the definition of convergence in MSE:

**Definition 4 (convergence in MSE)**:

If there exists a constant $C$ such that:
$$
\limsup_{\Gamma \to \infty} \;\; \mathbb{E}[\frac{(\rho_\Gamma-\rho)^2}{\Gamma^{-2}}] \leq C
$$

The MSE converges in the order of $\mathcal{O}(\Gamma^{-2})$.

Hence, it can be observed that above Pareto random variable converges in MSE in the order of $\mathcal{O}(\Gamma^{-2})$.

Let's state the definition of convergence in probability:

**Definition 5 (Convergence in probability)**:

For any $\epsilon > 0$, there exists a constant $C$ such that:
$$
\mathbb{P} \left(|\rho_\Gamma-\rho| \geq C \Gamma^{-1}\right) \leq \epsilon
$$

For this Pareto random variable, we are able to compute the probability using the survival function:

$$
\mathbb{P} \left(|\rho_\Gamma-\rho| \geq C \Gamma^{-1}\right) = \frac{1}{(C \Gamma^{-1} \Gamma )^\alpha} = C^{-\alpha}
$$

The right hand side of the above equation is a constant, so it can not be smaller than any $\epsilon > 0$.

Hence, **Definition 5** is not satisfied.

Therefore, the above Pareto random variable does not converge in probability in the order of $\mathcal{O}(\Gamma^{-1})$.

Since Professor Weng's comment is related to the convergence in probability, we can easily verify that the above Pareto random variable converges in probability in the order of $\mathcal{O}(\Gamma^{-1-\delta})$ for any $\delta > 0$.


## Normal random variable

If we try to show the same result for a normal random variable, we can observe that the normal random variable satisfies **Definition 4** with $C=1$. Therefore, we have:

$$
\mathbb{E}[(\rho_\Gamma-\rho)^2] = \mathcal{O}(\Gamma^{-2})
$$

We observe that:

$$
\begin{align}
\mathbb{P}(|\rho_\Gamma-\rho| \geq C \Gamma^{-1}) 
& = 2 \phi\left(\frac{C \Gamma^{-1}}{\Gamma^{-1}}\right)  \\
& = 2 \phi\left(C\right)  
\end{align}
$$

Hence, **Definition 5** is not satisfied for the normal random variable.














