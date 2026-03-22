# Optimization Code: Heavy Ball Method

## Overview
This project implements the Heavy Ball method for optimization and analyzes its convergence behavior on a test function. This experiment focuses on ill-conditioned optimization behavior using the Rosenbrock function.

The objective function used is:

f(x) = 100(x_2 - x_1^2)^2 + (1 - x_1)^2

This is the Rosenbrock function, a standard test problem in optimization.

---

## Purpose
The goal of this project is to study the convergence behavior of momentum-based optimization methods and compare them against baseline methods.

---

## Methods Implemented
- Gradient (first derivative)
- Hessian (second derivative)
- Heavy Ball optimization method
- Fletcher-Reeves variant
- Convergence visualization

## Algorithm Details

The Heavy Ball method updates iterates using:

x_{k+1} = x_k - α ∇f(x_k) + β (x_k - x_{k-1})

where:
- α = step size (learning rate)
- β = momentum parameter

## Parameters

- Initial point: x0 = [1.2, 1.2]
- Tolerance: 1e-11
- Max iterations: 1e6
- Initial step size: 1e-3

## Comparison

Two variants are implemented:
- Standard Heavy Ball method with adaptive parameters
- Fletcher-Reeves inspired momentum update

The Heavy Ball method demonstrates faster convergence compared to gradient descent, as seen by the steeper decay in gradient norm.


## Output

The script produces:
- Iterative convergence behavior
- Trajectory of optimization
- Plots of function value vs iterations

## File Structure
- `heavy_ball_convergence.py`  
  Contains:
  - function definition
  - gradient and Hessian
  - optimization algorithms
  - convergence plotting

---

## Example Output

![Gradient Descent](gradient_descent.png)

![Heavy Ball](heavy_ball.png)

![Fletcher-Reeves](fletcher_reeves.png)

---

## Requirements

Install dependencies:

```bash
pip install numpy matplotlib
