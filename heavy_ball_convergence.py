import numpy as np
from numpy.linalg import norm, solve, multi_dot, eigvals
import matplotlib.pyplot as plt

def f(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def Df(x):
    dx1 = -400 * x[0] * (x[1] - x[0]**2) + 2 * (x[0] - 1)
    dx2 = 200 * (x[1] - x[0]**2)
    return np.array([dx1, dx2])

def D2f(x):
    dx1dx1 = 1200 * x[0]**2 - 400 * x[1] + 2
    dx1dx2 = -400 * x[0]
    dx2dx2 = 200
    return np.array([[dx1dx1, dx1dx2], [dx1dx2, dx2dx2]])


def compute_alpha_beta(x, D2f):
    e_vals = np.linalg.eigvals(D2f(x))
    lambda_min, lambda_max = np.min(e_vals), np.max(e_vals)
    if (lambda_min >= 0) and (lambda_max >= 0):
        alpha = 4 / (np.sqrt(lambda_min) + np.sqrt(lambda_max))**2
        beta = ((np.sqrt(lambda_min) - np.sqrt(lambda_max))**2 / (np.sqrt(lambda_min) + np.sqrt(lambda_max))**2)
        return alpha, beta
    return 1e-3, 0.9

# Settings for heavy ball method
"""
 @description: heavy ball method
 @parameters :
    @f    : objective function  
    @Df: gradient of objective function
    @D2f: hessian of objective function
    @x0         : starting point 
    @tol
        :
 tolerace for stopping criteria 
    @maxIter    : maximum iteration for stopping criteria
 """

def heavyBall_default(f, Df, D2f, x0, alpha0, tol, maxIter):
    path      = [x0]
    grad_norms = [norm(f(x0))]                           
    k         = 0
    xk        = x0    
    pk        = -Df(xk)
    alpha = alpha0 # variable initial step length
    beta  = 0.9  # large initial heavy ball
    # Compute the first step separately. 
    if norm(pk) < tol:
        return xk, 0, path, grad_norms
    else:
        k = k + 1
        xk = xk + alpha * pk 
        path.append(xk)
        grad_norms.append(norm(f(xk)))
    # The rest of iterations
    pk = -Df(xk)
    while norm(pk) > tol and k <= maxIter: 
        xk  = xk + alpha * pk + beta * (xk - path[-2])
        pk  = -Df(xk)
        k   = k + 1
        path.append(xk)
        grad_norms.append(norm(f(xk)))
        alpha, beta = compute_alpha_beta(xk, D2f)
    path = np.array(path)
    if norm(pk) <= tol:
        print("Found the minimizer at {x} with {iter} iterations successfully, gradient's norm is {nrm}.".format(x=xk,iter=k,nrm=norm(pk)))
    else:
        print("Unable to locate minimizer within maximum iterations, last position is at {x}, gradient's norm is {nrm}".format(x=xk,nrm=norm(pk)))
    return xk, k, path, grad_norms

# FletcherReeves' Heavy Ball method
def heavyBall_FletcherReeves(f, Df, x0, alpha0, tol, maxIter):
    path = [x0]
    grad_norms = [norm(Df(x0))]
    k = 0
    xk = x0    
    pk = -Df(xk)
    alpha = alpha0  # variable initial step length
    
    if norm(pk) < tol:
        return xk, k, path, grad_norms
    else:
        k = k + 1
        xk_1 = xk + alpha * pk 
        path.append(xk_1)
        grad_norms.append(norm(Df(xk_1)))
    
    while norm(Df(xk_1)) > tol and k <= maxIter:
        beta_k = (norm(Df(xk_1))**2) / (norm(Df(xk))**2)
        xk, xk_1 = xk_1, xk_1 + alpha * (-Df(xk_1)) + beta_k * (xk_1 - xk)
        k += 1
        path.append(xk_1)
        grad_norms.append(norm(Df(xk_1)))
    path = np.array(path)
    if norm(Df(xk_1)) <= tol:
        print("Found the minimizer at {x} with {iter} iterations successfully, gradient's norm is {nrm}.".format(x=xk,iter=k,nrm=norm(pk)))
    else:
        print("Unable to locate minimizer within maximum iterations, last position is at {x}, gradient's norm is {nrm}".format(x=xk,nrm=norm(pk)))
    return xk_1, k, path, grad_norms


x0 = np.array([1.2, 1.2]) # Harder to converge 
tol = 1e-11 
maxIter = 1e6 
alpha0 = 1e-3

x_default, iter_default, path_default, grad_norms_default = heavyBall_default(f, Df, D2f, x0, alpha0, tol, maxIter) 

x_FletcherReeves, iter_FletcherReeves, path_FletcherReeves, grad_norms_FletcherReeves = heavyBall_FletcherReeves(f, Df, x0, alpha0, tol, maxIter)

def plot_heavyBall(method, grad_norms):
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(grad_norms)), grad_norms)
    plt.yscale("log")
    plt.xlabel("Iteration Number")
    plt.ylabel("Norm of Gradient (log scale)")
    plt.title(f"Convergence of {method} Method")
    plt.grid(True)
    plt.show()
plot_heavyBall("Heavy Ball", grad_norms_default)
plot_heavyBall("Fletcher-Reeves Adaptive Heavy Ball", grad_norms_FletcherReeves)
