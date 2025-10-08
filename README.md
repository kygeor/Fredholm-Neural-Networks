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

1. Fredholm Neural Networks for Integral Equations

The $M$-layer FIE approximation $f_K(x)$ can be implemented as a deep neural network with a one-dimensional input $x$, $M$ hidden layers, a linear activation function and a single output node corresponding to the estimated solution $f(x)$. The weights and biases are:

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

