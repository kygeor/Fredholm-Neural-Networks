# Fredholm Neural Networks
This repository contains the codes in Python and MATLAB that were developed and used for the Fredholm Neural Network (FNN) and Potential Fredholm Neural Network (PFNN) framework.

The theoretical framework, used for both the forward and inverse problems, is briefly described below. For the full details see the papers:

1. Fredholm Neural Networks - https://epubs.siam.org/doi/full/10.1137/24M1686991?casa_token=LUOO2mbMhAcAAAAA%3AQUFO1UaeNBfHdXzGBU2c_oZFy2vwIea8jtON46KL_TC_wkjEke7VEW-lLoQ9bY0Gw9BZcFy1
2. Fredholm Neural Networks for forward and inverse problems in elliptic PDEs - https://arxiv.org/abs/2507.06038

To reference this code please refer to:

@article{georgiou2025fredholm,
  title={Fredholm neural networks},
  author={Georgiou, Kyriakos and Siettos, Constantinos and Yannacopoulos, Athanasios N},
  journal={SIAM Journal on Scientific Computing},
  volume={47},
  number={4},
  pages={C1006--C1031},
  year={2025},
  publisher={SIAM}
}

and/or 

@article{georgiou2025fredholm,
  title={Fredholm Neural Networks for forward and inverse problems in elliptic PDEs},
  author={Georgiou, Kyriakos and Siettos, Constantinos and Yannacopoulos, Athanasios N},
  journal={arXiv preprint arXiv:2507.06038},
  year={2025}
}

# 1.  Fredholm Neural Networks for Integral Equations

## Background
The basis of FNNs is the method of successive approximations (fixed point iterations) to approximate the fixed-point solution to Fredholm Integral Equations (FIEs). Specifically, the framework is built upon linear FIEs of the second kind, which are of the form:

$$f(x) = g(x) + \int_{\Omega}K(x,z) f(z)dz, $$

as well as the non-linear counterpart,

$$f(x) = g(x) + \int_{\Omega}K(x,z) G(f(z))dz,$$

for some function $G: \mathbb{R} \rightarrow\mathbb{R}$ considered to be a Lipschitz function. 

We consider the cases where the integral operators are either contractive or non-expansive. This allows linear FIE defined by a non-expansive operator $\mathcal{T}$, and a sequence $\{\kappa_n\}, \kappa_n \in (0,1]$ such that $\sum_n \kappa_n(1-\kappa_n) = \infty$. Then, the iterative scheme:

$$f_{n+1}(x) = f_n(x) + \kappa_n(\mathcal{T}f_n(x) -f_n(x)) = (1-\kappa_n)f_n(x) + \kappa_n \mathcal{T} f_n(x),$$

with $f_0(x) = g(x)$, converges to the fixed point solution of the FIE, $f^{*}(x)$.

When $\mathcal{T}$ is a contraction, we can obtain the iterative process:
$$f_n(x)= g(x) +  \int_{\Omega}f_{n-1})(x), \,\,\ n \geq 1,$$
which converges to the fixed point solution. This is often referred to as the method of successive approximations.

## FNN construction
Fredholm Neural Networks are based on the observation that the FIE approximation $f_K(x)$ can be implemented as a deep neural network with a one-dimensional input $x$, $M$ hidden layers, a linear activation function and a single output node corresponding to the estimated solution $f(x)$. The weights and biases are:

$$
W_1 =
\begin{bmatrix}
\kappa g(z_1) \\
\vdots \\
\kappa g(z_{N})
\end{bmatrix},
\qquad
b_1 =
\begin{bmatrix}
0 \\
\vdots \\
0
\end{bmatrix}.
$$

for the first hidden layer,

$$
W_m =
\begin{bmatrix}
K_D(z_1) & K(z_1,z_2)\,\Delta z & \cdots & K(z_1,z_N)\,\Delta z \\
K(z_2,z_1)\,\Delta z & K_D(z_2) & \cdots & K(z_2,z_N)\,\Delta z \\
\vdots & \vdots & \ddots & \vdots \\
K(z_N,z_1)\,\Delta z & K(z_N,z_2)\,\Delta z & \cdots & K_D(z_N)
\end{bmatrix},
$$

and

$$
b_m =
\begin{bmatrix}
\kappa g(z_1) \\
\vdots \\
\kappa g(z_N)
\end{bmatrix},
\qquad m=2,\dots,M-1,
$$

where $K_D(z) := K(z,z)\,\Delta z + (1-\kappa_m)$. Finally,

$$
W_M =
\begin{bmatrix}
K(z_1,x)\,\Delta z \\
\vdots \\
K(z_{i-1},x)\,\Delta z \\
K_D(x) \\
K(z_{i+1},x)\,\Delta z \\
\vdots \\
K(z_N,x)\,\Delta z
\end{bmatrix},
\qquad
b_M = \kappa g(x),
$$

assuming $z_i = x$.


<img width="324" height="290" alt="Screenshot 2025-10-08 at 11 45 05 AM" src="https://github.com/user-attachments/assets/2cdfd98b-7c52-4119-999d-b1bc40732a6b" /> 
<img width="575" height="248" alt="Screenshot 2025-10-08 at 11 45 33 AM" src="https://github.com/user-attachments/assets/bbda1e93-36b5-4c83-afa3-8b86d9459996" />  

*Figure 1: Architecture of the Fredholm Neural Network (FNN). Outputs can be considered across the entire (or a subset of the) input grid, or for an arbitrary output vector as shown in the second graph, by applying the integral mapping one last time.*


## Application to non-linear FIEs 

We can create an iterative process that "linearizes" the integral equation and allows us to solve a linear FIE at each step. To this end, consider the non-linear, non-expansive integral operator:

$$(\mathcal{T}f)(x) := g(x) + \int_{\Omega}K(x,z) G(f(z))dz.$$

Then, the iterative scheme $f_n(x) = \tilde{f}_n(x)$, where $\tilde{f}_n(x)$ is the solution to the linear FIE:

$$\tilde{f}_{n}(x) = ({L}\tilde{f}_{n-1})(x) + \int_{\Omega}K(x,z) \tilde{f}_{n}(z))dz,$$
    
where: 

$$(\mathcal{L}\tilde{f}_{n-1})(x) := g(x) + \int_{\Omega} K(x,y)\big( G(\tilde{f}_{n-1}(y)) - \tilde{f}_{n-1}(y)\big)dy,$$ 

for $n \geq 1$, converges to the fixed point $f^*$  which is a solution of the non-linear FIE.

<img width="636" height="207" alt="Screenshot 2025-10-08 at 1 35 31 PM" src="https://github.com/user-attachments/assets/f692d52e-21a0-4f2a-b668-f8a938527a3f" />

*Figure 2: Iterative process to solve the non-linear FIE using the Fredholm NN architecture.*



## Application to BVP ODEs

Consider a BVP of the form:

$$y''(x) + g(x)y(x) = h(x), 0<x<1,$$ 
    
with $y(0) = \alpha, y(1) = \beta$. Then we can solve the BVP by obtaining the following FIE:

$$u(x) = f(x) + \int_{0}^{1} K(x,t) u(t)dt,$$

where $u(x) = y''(x), f(x) = h(x) - \alpha g(x) - (\beta - \alpha) x g(x)$, and the kernel is given by:

$$ K(x,t) = 
    \begin{cases}
        t(1-x)g(x), \,\,\, 0 \leq t \leq x \\
        x(1-t)g(x), \,\,\, x\leq t \leq 1.
    \end{cases}$$
    
Finally, by definition of $u(x)$, we can obtain the solution to the BVP by:

$$y(x) = \frac{h(x) - u(x)}{g(x)}.$$


# Potential Fredholm Neural Networks for elliptic PDEs


