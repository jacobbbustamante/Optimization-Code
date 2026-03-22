import numpy as np
from numpy.linalg import norm, eigvals
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
    eigenvalues = eigvals(D2f(x))
    lambda_min, lambda_max = np.min(eigenvalues), np.max(eigenvalues)
    
    if (lambda_min >= 0) and (lambda_max >= 0):
        alpha = 4 / (np.sqrt(lambda_min) + np.sqrt(lambda_max))**2
        beta = (
            (np.sqrt(lambda_min) - np.sqrt(lambda_max))**2 
             / (np.sqrt(lambda_min) + np.sqrt(lambda_max))**2
        )
        return alpha, beta
    return 1e-3, 0.9

def gradient_descent(f, Df, x0, alpha, tol, max_iter):
    path = [x0.copy()]
    grad_norms = [norm(Df(x0))]
    xk = x0.copy()
    k = 0

    while norm(Df(xk)) > tol and k <= max_iter:
        xk = xk - alpha * Df(xk)
        k += 1
        path.append(xk.copy())
        grad_norms.append(norm(Df(xk)))

    path = np.array(path)

    if norm(Df(xk)) <= tol:
        print(
            "Gradient Descent found the minimizer at {x} in {iter} iterations successfully, gradient norm is {nrm}."
            .format(x=xk, iter=k, nrm=norm(Df(xk)))
        )
    else:
        print(
            "Gradient Descent was unable to locate minimizer after {max_iterations} iterations, last position is at {x}, gradient norm is {nrm}"
            .format(max_iterations=int(max_iter), x=xk, nrm=norm(Df(xk)))
        )

    return xk, k, path, grad_norms


def heavy_ball(f, Df, D2f, x0, alpha0, tol, max_iter):
    path       = [x0.copy()]
    grad_norms = [norm(Df(x0))]                           
    k          = 0
    xk         = x0.copy()    
    alpha      = alpha0 # variable initial step length
    beta       = 0.9  # large initial heavy ball
    k          = 0
    pk         = -Df(xk)

    # Compute the first step separately. 
    if norm(pk) < tol:
        return xk, k, np.array(path), grad_norms
    
    xk = xk + alpha * pk 
    k = k + 1

    path.append(xk.copy())
    grad_norms.append(norm(Df(xk)))

    # The rest of iterations
    pk = -Df(xk)
    while norm(pk) > tol and k <= int(max_iter): 
        xk  = xk + alpha * pk + beta * (xk - path[-2])
        pk  = -Df(xk)
        k  += 1
        path.append(xk.copy())
        grad_norms.append(norm(Df(xk)))
        alpha, beta = compute_alpha_beta(xk, D2f)
    path = np.array(path)
    if norm(pk) <= tol:
        print("Heavy Ball found the minimizer at {x} in {iter} iterations successfully, gradient norm is {nrm}.".format(x=xk,iter=k,nrm=norm(pk)))
    else:
        print("Heavy Ball was unable to locate minimizer after {max_iterations} iterations, last position is at {x}, gradient norm is {nrm}".format(max_iterations=int(max_iter), x=xk,nrm=norm(pk)))
    return xk, k, path, grad_norms

# FletcherReeves' Heavy Ball method
def heavy_ball_fletcher_reeves(f, Df, x0, alpha0, tol, max_iter):
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
    
    while norm(Df(xk_1)) > tol and k <= max_iter:
        beta_k = (norm(Df(xk_1))**2) / (norm(Df(xk))**2)
        xk, xk_1 = xk_1, xk_1 + alpha * (-Df(xk_1)) + beta_k * (xk_1 - xk)
        k += 1
        path.append(xk_1)
        grad_norms.append(norm(Df(xk_1)))
    path = np.array(path)
    if norm(Df(xk_1)) <= tol:
        print("Fletcher-Reeves Heavy Ball found the minimizer at {x} in {iter} iterations successfully, gradient norm is {nrm}.".format(x=xk,iter=k,nrm=norm(pk)))
    else:
        print("Fletcher-Reeves Heavy Ball was unable to locate minimizer after {max_iterations} iterations, last position is at {x}, gradient norm is {nrm}".format(max_iterations=int(max_iter), x=xk,nrm=norm(pk)))
    return xk_1, k, path, grad_norms

def plotting(method_name, grad_norms):
        plt.figure(figsize=(8, 6))
        plt.plot(range(len(grad_norms)), grad_norms)
        plt.yscale("log")
        plt.xlabel("Iteration Number")
        plt.ylabel("Norm of Gradient (log scale)")
        plt.title(f"Convergence of {method_name} Method")
        plt.grid(True)
        plt.savefig(f"{method_name}.png")
        plt.close()

if __name__ == "__main__":
    x0 = np.array([1.2, 1.2]) # Harder to converge 
    tol = 1e-11 
    max_iter = 1e6 
    alpha0 = 1e-3

    x_gd, iter_gd, path_gd, grad_norms_gd = gradient_descent(
        f, Df, x0, alpha0, tol, max_iter
    )

    x_heavy_ball, iter_default, path_default, grad_norms_heavy_ball = heavy_ball(
        f, Df, D2f, x0, alpha0, tol, max_iter
    ) 

    x_fletcher_reeves, iter_fletcher_reeves, path_fletcher_reeves, grad_norms_fletcher_reeves = heavy_ball_fletcher_reeves(
        f, Df, x0, alpha0, tol, max_iter
    )

    plotting("gradient_descent", grad_norms_gd)
    plotting("heavy_ball", grad_norms_heavy_ball)
    plotting("fletcher_reeves", grad_norms_fletcher_reeves)