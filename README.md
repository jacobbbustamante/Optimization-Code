# Optimization Code: Heavy Ball Method

## Overview
This project implements the Heavy Ball method for optimization and analyzes its convergence behavior on a test function.

The objective function used is:

f(x) = 100(x2 - x1^2)^2 + (1 - x1)^2

This is the Rosenbrock function, a standard test problem in optimization.

---

## Algorithm Details

The Heavy Ball method updates iterates using:

x_{k+1} = x_k - α ∇f(x_k) + β (x_k - x_{k-1})

where:
- α = step size (learning rate)
- β = momentum parameter

## Output

The script produces:
- Iterative convergence behavior
- Trajectory of optimization
- Plots of function value vs iterations

## Comparison

Two variants are implemented:
- Standard Heavy Ball method with adaptive parameters
- Fletcher-Reeves inspired momentum update

Their convergence behaviors are compared using gradient norm decay.

## Methods Implemented

- Gradient (first derivative)
- Hessian (second derivative)
- Heavy Ball optimization method
- Convergence visualization

---

## File Structure

- `heavy_ball_convergence.py`  
  Main script containing:
  - function definition
  - gradient and Hessian
  - optimization algorithm
  - plotting of convergence behavior

---

## Example Output

## Example Output

![Heavy Ball](Heavy%20Ball.png)

![Fletcher-Reeves](Fletcher-Reeves%20Adaptive%20Heavy%20Ball.png)

## Requirements

Install required Python packages:

```bash
pip install numpy matplotlib
