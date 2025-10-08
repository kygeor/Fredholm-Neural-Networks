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

The $M$-layer FIE approximation $f_K(x)$ can be implemented as a Deep Neural Network with a one-dimensional input $x$, $M$ hidden layers, a linear activation function and a single output node corresponding the estimated solution $f(x)$, where the weights and biases are given by:
\begin{flalign}
    W_1 = \left(\begin{array}{ccc}
		\kappa g(z_1), \dots, \kappa g(z_{N})
	\end{array}\right)^{\top}, \,\,\,\,\    b_1 = \left(\begin{array}{ccc}
		0, 0, \dots, 0
	\end{array}\right)^{\top}
 \end{flalign} 
for the first hidden layer,  
\begin{eqnarray}	\label{inner-weight}
W_m=
%\left\{\Tilde{K}\left(z_i^{(m-1)}, z_j^{(m)}\right) \Delta z\right\}_{\substack{i \in\left\{1, \ldots, N_{m-1}\right\} \\ j \in\left\{1, . ., N_m\right\}}}= %
\left(\begin{array}{cccc}
	K_D\left(z_1\right) & {K}\left(z_1, z_2\right)\Delta z & \cdots & {K}\left(z_1, z_{N}\right)\Delta z \\
 {K}\left(z_2, z_1\right)\Delta z  & K_D\left(z_2\right) & \cdots & {K}\left(z_2, z_{N}\right)\Delta z \\
	\vdots & \vdots & \ddots & \vdots \\
	\vdots & \vdots & \vdots & \vdots \\
	{K}\left(z_{N}, z_1\right)\Delta z & {K}\left(z_{N}, z_2\right)\Delta z & \cdots & K_D\left(z_{N}\right) 
\end{array}\right),
\end{eqnarray}
and
\begin{eqnarray}
	b_m=\left(\begin{array}{ccc}
		\kappa g(z_1), \dots, \kappa g(z_{N})
	\end{array}\right)^{\top},
\end{eqnarray}
for hidden layers $m= 2, \dots, M-1$, where $K_D\left(z\right) := {K}\left(z, z\right)\Delta z + (1-\kappa_m)$, and:
\begin{flalign} \label{outer-weight}
	\begin{gathered}
		W_M=\left(\begin{array}{ccc}
			K(z_1, x)\Delta z, \dots, K(z_{i-1},x)\Delta z, K_D(x), K(z_{i+1}, x)\Delta z, \dots, K(z_{N}, x)\Delta z
		\end{array}\right)^{\top},
	\end{gathered}
\end{flalign}
$b_M =\big(\kappa g(x) \big)$, for the final layer, assuming $z_i = x$.

<img width="324" height="290" alt="Screenshot 2025-10-08 at 11 45 05 AM" src="https://github.com/user-attachments/assets/2cdfd98b-7c52-4119-999d-b1bc40732a6b" />
<img width="575" height="248" alt="Screenshot 2025-10-08 at 11 45 33 AM" src="https://github.com/user-attachments/assets/bbda1e93-36b5-4c83-afa3-8b86d9459996" />

