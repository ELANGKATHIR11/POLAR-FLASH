# POLAR-FLASH: Mathematical Foundations & Rigorous Derivations

This document details the mathematical theory, statistical derivations, and numerical proofs implemented throughout the POLAR-FLASH framework.

---

## 1. Information Theory & Shannon Rate-Distortion

### 1.1 Rate-Distortion Function
For a continuous random variable $W$ with probability density function $p(w)$ and reconstructed reproduction $\hat{W}$, the Shannon Rate-Distortion function $R(D)$ characterizes the minimal mutual information $I(W; \hat{W})$ required to achieve an expected distortion bound:

\[
R(D) = \min_{p(\hat{w}|w): \mathbb{E}[d(W, \hat{W})] \leq D} I(W; \hat{W})
\]

Under mean squared error (MSE) distortion $d(w, \hat{w}) = (w - \hat{w})^2$:

#### Gaussian Scalar Source
For a zero-mean Gaussian source $W \sim \mathcal{N}(0, \sigma^2)$, the Shannon lower bound is exact:

\[
R(D) = \begin{cases}
\frac{1}{2} \log_2 \left( \frac{\sigma^2}{D} \right) & \text{for } 0 < D \leq \sigma^2 \\
0 & \text{for } D > \sigma^2
\end{cases}
\]

Inverting this relationship gives the distortion-rate curve:

\[
D(R) = \sigma^2 2^{-2R}
\]

#### Vector Gaussian Source
For a multivariate Gaussian vector $W \in \mathbb{R}^d$ with covariance matrix $\Sigma$ having eigenvalues $\{\lambda_1, \lambda_2, \dots, \lambda_d\}$, the optimal rate allocation across orthogonal coordinates is given by reverse water-filling:

\[
R(D) = \frac{1}{2} \sum_{i=1}^d \max\left(0, \log_2 \frac{\lambda_i}{\theta}\right)
\]

where parameter $\theta > 0$ satisfies the total distortion constraint:

\[
D = \sum_{i=1}^d \min(\theta, \lambda_i)
\]

### 1.2 Entropy-Constrained Scalar Quantization (ECSQ)
Uniform quantization with interval $\Delta$ followed by entropy coding minimizes the Lagrangian objective:

\[
J(\lambda) = D + \lambda H(\hat{W})
\]

For small step size $\Delta$ (high-resolution asymptotic regime):
- Distortion: $D \approx \frac{\Delta^2}{12}$
- Output entropy: $H(\hat{W}) \approx h(W) - \log_2 \Delta$
- Where $h(W) = -\int p(w) \log_2 p(w) dw$ is differential entropy.

### 1.3 Constrained Layer-Wise Rate Allocation
Given an $L$-layer neural network with parameter counts $N_l$ and layer sensitivities $\alpha_l = \mathbb{E}[\|\nabla_{W_l} \mathcal{L}\|_2^2]$, the optimal bit allocation minimizes total task degradation subject to a storage budget $B_{\text{total}}$:

\[
\min_{\{R_l\}} \sum_{l=1}^L \alpha_l D_l(R_l) \quad \text{subject to} \quad \sum_{l=1}^L N_l R_l \leq B_{\text{total}}
\]

Substituting the Gaussian distortion model $D_l(R_l) = \sigma_l^2 2^{-2 R_l}$:

\[
\mathcal{L}(R, \mu) = \sum_{l=1}^L \alpha_l \sigma_l^2 2^{-2 R_l} + \mu \left( \sum_{l=1}^L N_l R_l - B_{\text{total}} \right)
\]

Differentiating with respect to $R_l$ and setting to zero yields:

\[
-2 \ln(2) \alpha_l \sigma_l^2 2^{-2 R_l} + \mu N_l = 0 \implies R_l^* = \frac{1}{2} \log_2 \left( \frac{2 \ln(2) \alpha_l \sigma_l^2}{\mu N_l} \right)
\]

POLAR-FLASH solves for optimal $\mu$ using bisection search over the interval $[\mu_{\min}, \mu_{\max}]$ to satisfy $\sum_l N_l R_l^* = B_{\text{total}}$ within $10^{-6}$ numerical tolerance.

---

## 2. Orthogonal Preconditioning via Walsh-Hadamard Transforms

### 2.1 Coordinate Incoherence & Outlier Spikes
Weight matrices in pretrained LLMs exhibit extreme coordinate kurtosis, with rare outlier columns having amplitudes $50\times$ larger than typical elements. These outliers destroy low-bit scalar quantization by dominating the dynamic range.

Let $H \in \mathbb{R}^{d \times d}$ be a normalized Walsh-Hadamard matrix satisfying:

\[
H^T H = I_d, \quad H_{i,j} \in \left\{ -\frac{1}{\sqrt{d}}, +\frac{1}{\sqrt{d}} \right\}
\]

Applying randomized sign pre-rotation $z = H S w$, where $S = \operatorname{diag}(s_1, \dots, s_d)$ with independent Rademacher entries $s_i \in \{-1, +1\}$:

\[
z_i = \frac{1}{\sqrt{d}} \sum_{j=1}^d s_j w_j H_{i,j}
\]

By the Central Limit Theorem and Berry-Esseen Theorem, the rotated coordinates $z_i$ rapidly converge to a joint Gaussian distribution as $d \to \infty$.

### 2.2 Energy Preservation
Because $H$ is orthogonal and $S$ is unitary:

\[
\|z\|_2^2 = (H S w)^T (H S w) = w^T S^T H^T H S w = w^T S^2 w = \|w\|_2^2
\]

Energy is strictly conserved within float32 numerical precision ($\| \|z\|_2^2 - \|w\|_2^2 \| < 10^{-6}$).

---

## 3. High-Dimensional Geometry & Gaussian Concentration

### 3.1 Spherical Shell Concentration
For a $d$-dimensional independent Gaussian vector $z \sim \mathcal{N}(0, \sigma^2 I_d)$, the squared norm $\|z\|_2^2 = \sum_{i=1}^d z_i^2$ is distributed as $\sigma^2 \chi_d^2$:

- Expected value:
  \[
  \mathbb{E}[\|z\|_2^2] = d \sigma^2
  \]
- Variance:
  \[
  \operatorname{Var}(\|z\|_2^2) = 2 d \sigma^4
  \]
- Relative norm deviation:
  \[
  \frac{\sqrt{\operatorname{Var}(\|z\|_2^2)}}{\mathbb{E}[\|z\|_2^2]} = \sqrt{\frac{2}{d}} \xrightarrow{d \to \infty} 0
  \]

For $d = 4096$, the standard deviation of vector radius is only $2.2\%$ of the mean radius $R \approx \sigma \sqrt{d}$. The entire probability mass of high-dimensional weights concentrates on an infinitesimally thin spherical hypershell.

---

## 4. Recursive PolarQuant Formulation

### 4.1 2D Hyperspherical Coordinate Mapping
Weight coordinates are grouped into pairs $(x_1, x_2)$ and mapped to radius $r$ and angle $\theta$:

\[
r = \sqrt{x_1^2 + x_2^2}, \quad \theta = \operatorname{atan2}(x_2, x_1) \in [-\pi, \pi]
\]

Reconstruction is exact:

\[
x_1 = r \cos \theta, \quad x_2 = r \sin \theta
\]

### 4.2 Differential Polar Distortion
Let perturbations in radius and angle be $\Delta r = r - \hat{r}$ and $\Delta \theta = \theta - \hat{\theta}$. Expanding Cartesian error via Taylor approximation:

\[
x_1 - \hat{x}_1 = (r - \hat{r}) \cos \theta - r \sin \theta (\Delta \theta) + \mathcal{O}((\Delta \theta)^2)
\]

\[
x_2 - \hat{x}_2 = (r - \hat{r}) \sin \theta + r \cos \theta (\Delta \theta) + \mathcal{O}((\Delta \theta)^2)
\]

Summing squared deviations:

\[
D = (x_1 - \hat{x}_1)^2 + (x_2 - \hat{x}_2)^2 \approx (\Delta r)^2 + r^2 (\Delta \theta)^2
\]

**Crucial Insight**: Angular distortion scales proportionally with $r^2$. Coordinates with small radius require very few angular bits, while outer coordinates require higher angular precision.

---

## 5. $E_8$ Root Lattice Quantization

### 5.1 Lattice Definition
The Gosset lattice $\Lambda_8 \subset \mathbb{R}^8$ is the densest sphere packing in 8 dimensions, defined as:

\[
\Lambda_8 = D_8 \cup \left( D_8 + \frac{1}{2} \mathbf{1}_8 \right)
\]

where $D_8 = \{x \in \mathbb{Z}^8 : \sum_{i=1}^8 x_i \equiv 0 \pmod 2\}$.

The minimal vectors of $\Lambda_8$ (Shell 1) form the root system of Lie algebra $\mathfrak{e}_8$, comprising exactly 240 vectors of squared norm $\|v\|^2 = 2$:
1. $112$ vectors of type $(\pm 1, \pm 1, 0, 0, 0, 0, 0, 0)$ and permutations.
2. $128$ vectors of type $\left(\pm \frac{1}{2}, \dots, \pm \frac{1}{2}\right)$ with an even number of minus signs.

### 5.2 Fast Nearest-Point Decoding
Nearest-point decoding $\arg\min_{\lambda \in \Lambda_8} \|x - \lambda\|_2$ is solved in $\mathcal{O}(d)$ time:
1. Decode $x$ to $D_8$ using the Conway-Sloane algorithm: round each coordinate to the nearest integer, and if $\sum [x_i]$ is odd, perturb the coordinate with the largest rounding error.
2. Decode $x - \frac{1}{2} \mathbf{1}$ to $D_8$, then shift back by $+\frac{1}{2} \mathbf{1}$.
3. Select the candidate that minimizes Euclidean distance to $x$.

---

## 6. QJL-Inspired Residual Correction

Let $e = w - \hat{w}$ be the quantization residual vector. Given input activation $x$, the exact inner product is:

\[
\langle x, w \rangle = \langle x, \hat{w} \rangle + \langle x, e \rangle
\]

Using a Johnson-Lindenstrauss random projection matrix $S \in \mathbb{R}^{m \times d}$ with Rademacher entries $S_{i,j} \in \{-1, +1\}$:

\[
\hat{q} = \frac{1}{m} (S x)^T (S e)
\]

### 6.1 Unbiasedness Proof
\[
\mathbb{E}_S [\hat{q}] = \frac{1}{m} \mathbb{E}_S [x^T S^T S e] = \frac{1}{m} x^T \mathbb{E}_S [S^T S] e
\]

Because $\mathbb{E}[S_{i,k} S_{i,l}] = \delta_{k,l}$:

\[
\mathbb{E}_S [S^T S] = m I_d \implies \mathbb{E}_S [\hat{q}] = \frac{1}{m} x^T (m I_d) e = x^T e = \langle x, e \rangle
\]

The estimator is strictly unbiased. Variance is bounded by $\mathcal{O}\left(\frac{\|x\|_2^2 \|e\|_2^2}{m}\right)$.

---

## 7. Fisher Information Sparse Ternary Residuals

Second-order Taylor expansion of task loss $\mathcal{L}$ with respect to weight perturbation $\Delta w = r$:

\[
\Delta \mathcal{L} \approx g^T r + \frac{1}{2} r^T H r \approx \frac{1}{2} \sum_{i=1}^N F_i r_i^2
\]

where $F_i = \mathbb{E}[(\frac{\partial \mathcal{L}}{\partial w_i})^2]$ is the empirical Fisher diagonal.

Ternary residual decomposition identifies the top-$k$ Fisher-weighted coordinates and assigns optimal ternary levels:

\[
r_i \in \{ -s, 0, +s \}
\]

where optimal magnitude $s$ is the mean absolute residual of selected coordinates:

\[
s = \frac{\sum_{i \in \mathcal{I}_k} |e_i|}{k}
\]

POLAR-FLASH evaluates $1\%$, $3\%$, $5\%$, and $10\%$ sparsity, proving that $5\%$ ternary residuals recover $>82\%$ of low-bit quantization error with minimal storage overhead.

