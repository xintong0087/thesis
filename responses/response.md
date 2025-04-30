# List of Changes and Responses

The page numbers are from the revised draft, and the page numbers in brackets are from the original draft.

### 1. Numerical Experiments Details
- **Added in Chapter 2, Page 44(46):** Added a paragraph to explain the numerical experiments in Chapter 2.
- **Added in Chapter 2, Page 45(47):** Added a paragraph to explain maturity date of the options in the numerical experiments in Chapter 2.
- **Added in Chapter 2, Page 45-46(47):** Computing platform details for the numerical experiments in Chapter 2.
- **Added in Chapter 2, Page 56(58):** Added a paragraph to explain the Heston model parameters.
- The numerical experiments in Chapter 3 and Chapter 4 now have sufficient details.

### 2. Appendix Addition
- **New Addition (After Page xiv):** Added an appendix for all acronyms.
- After careful consideration, only one acronym table is added. The reason is that the number of acronyms used in this thesis is relatively small, making a single consolidated table more appropriate and efficient.

### 3. Clarification of Expected Values
- **Chapter 3, Pages 72-74(72):** 
    - Clarified definition for GMMB liability, showing how the conditional expectation works.
    - Added explanation for pathwise delta calculations.
- **Chapter 3, Page 75(74):** 
    - Corrected the mistake in the definition $H_t^{bf}$. 
    - Showed the derivation of the formula for loss random variable $L$.

### 4. Neural Network Approximation Clarification
- **Chapter 3, Page 66** 
    - Added a paragraph to explain how the neural network approximates the relationship between the outer scenarios and the VA loss.
- **Chapter 3, Page 78(76):** 
    - Added explanation that neural networks approximate L in Equation (3.4).
    - Added explanation that L depends on all hedging weights (Δ₀,...,ΔT₋₁) and each Δt depends on St.
    - Explained that $L(\mathbf{S}_T)$ is both a random variable and a function of the underlying path.

### 5. Regression References
- **Chapter 2, Page 52(54):** 
    - Added one line to explain the high empirical convergence rate reported in the graph legends.
    - I think in the thesis, the high empirical convergence rate is well-explained using a separate set of experiments. 

### 6. Multi-level Monte Carlo Sections
- **Chapter 2, Page 21(21):** Rewrote sections on multi-level Monte Carlo with clearer descriptions.
- **Chapter 2, Page 57(59):** Added more implementation details for replication purposes.

### 7. Cauchy-Schwarz Inequality
- **Chapter 2, Page 30:** 
    - Added explanation between the inequality:
        $$\mathbb{E}[(\hat{\rho}_{M,N} - \rho)^2] \leq 2\mathbb{E}[(\hat{\rho}_{M,N} - \rho_M)^2] + 2\mathbb{E}[(\rho_M - \rho)^2]$$
    - Included proper explanation for Cauchy-Schwarz inequality.

### 8. Taylor Expansion Clarification
- **Chapter 2, Page 31(33):** 
    - Clearly defined variable $z$ when applying Taylor expansion.
- **Global check, Pages 34(36):** Verified all other Taylor expansions are clearly explained.

### 9. Section 2.3.1 Rewrite
- **Chapter 2, (Page 28-30):** The section is completely rewritten.
    - The original section was not clear and the proof of Theorem 1 was not rigorous.
    - The section is moved to the appendix due to the lack of theoretical contributions.
    - Chapter 2 is rewritten to be more consistent with the rest of the thesis.
        - **Page 17, 35, 37**: Deleted the part about the connection between the convergence rates.
        - **Page 19, 28, 44**: Reworded a sentence to be more consistent with the rest of the thesis.
        - **Page 25**: Added a sentence to be more consistent with the rest of the thesis.
        - **Page 27**: Deleted sentences that are not relevant to the thesis anymore.
        - **Page 63**: Added a sentence to clarify the contribution of this chapter.
- I am fully aware that these changes do not fully address the comments from the committee. A supporting document is provided to show the difficulty in drawing the connections between the convergence rates. I suggest that the connection between the convergence rates to be left for future consideration. 

### 10. Minor Items
- **Page i:** Moved copyright statement to first page.
- **Page ii:** Added paragraph under the "sole-author" declaration noting that part of this thesis has been published in a WSC proceeding.
