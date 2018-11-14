"""
Este script resuelve un problema simple de diffusion en 1D.
La ecuación a resover es:
    dT/dt = d2T/dx2; 
    T(0,x) = sin(pi * x);
    T(t, 0) = T(t, 1) = 0
"""

import numpy as np
import matplotlib.pyplot as plt


# SETUP
Nx = 25
N_temporales = 400
h = 1 / (Nx - 1)
# eps = h**2 / 2 # Max para metodo explicito estable
# eps = eps / 2
eps = 0.001
s = eps / h**2 / 2


T = np.zeros(Nx)
T_archive = np.zeros((N_temporales, Nx))

# Inicializamos T
T = np.sin(np.pi * h * np.arange(Nx))
b = np.zeros(Nx)
alpha = np.zeros(Nx)
beta = np.zeros(Nx)


# UN PASO TEMPORAL

# llenar b
def llena_b(b, T, Nx, s):
    for i in range(1, Nx-1):
        b[i] = s * T[i+1] + (1 - 2 * s) * T[i] + s * T[i-1]

# llenar alpha y beta
def llena_alpha_y_beta(alpha, beta, b, Nx, s):
    alpha[0] = 0
    beta[0] = 0 # por cond de borde
    for i in range(1, Nx):
        alpha[i] = s / (-s * alpha[i-1] + (2 * s + 1)) 
        beta[i] = (b[i] + s * beta[i-1]) / (-s * alpha[i-1] + (2 * s + 1))

# calcular T^(n+1)
def avanza_T(alpha, beta, Nx):
    T[-1] = 0 # cond de borde <=> T[Nx - 1] = 0
    T[0] = 0
    for i in range(Nx-2, 0, -1):
        T[i] = alpha[i] * T[i+1] + beta[i]


T_archive[0] = T.copy()
for j in range(1, N_temporales):
    llena_b(b, T, Nx, s)
    llena_alpha_y_beta(alpha, beta, b, Nx, s)
    avanza_T(alpha, beta, Nx)
    T_archive[j] = T.copy()

# Plots
plt.figure(1)
plt.clf()

for i in range(0, N_temporales, 100):
    label = "t = {:.3f}".format(i * eps)
    plt.plot(np.arange(Nx) * h, T_archive[i], label=label)

plt.xlabel("X")
plt.ylabel("Temp")
plt.legend()
plt.show()

plt.figure(2)
plt.clf()

t = np.arange(0, N_temporales) * eps
x = np.arange(Nx) * h
X, T = np.meshgrid(x, t)
plt.pcolormesh(X, T, T_archive)

plt.show()

plt.figure(3)
plt.clf()
plt.plot(np.arange(400) * eps, T_archive[:, 12])
plt.xlabel('tiempo')
plt.ylabel('Temp en pixel 12')
plt.show()