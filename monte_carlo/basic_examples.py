import numpy as np
import matplotlib.pyplot as plt

def estimate_pi(n_points):
    """Оценка числа π методом Монте-Карло"""
    x = np.random.uniform(0, 1, n_points)
    y = np.random.uniform(0, 1, n_points)
    
    distances = np.sqrt(x**2 + y**2)
    points_inside = np.sum(distances <= 1)
    pi_estimate = 4 * points_inside / n_points
    
    plt.figure(figsize=(8, 8))
    plt.scatter(x[distances > 1], y[distances > 1], c='red', alpha=0.6, label='Вне круга')
    plt.scatter(x[distances <= 1], y[distances <= 1], c='blue', alpha=0.6, label='Внутри круга')
    plt.axis('equal')
    plt.legend()
    plt.title(f'Оценка π = {pi_estimate:.6f}')
    plt.savefig('images/pi_estimation.png')
    plt.close()
    
    return pi_estimate

def integrate_monte_carlo(n_points):
    """Оценка определенного интеграла sin(x) от 0 до π"""
    x = np.random.uniform(0, np.pi, n_points)
    y = np.random.uniform(0, 1, n_points)
    
    points_under_curve = np.sum(y <= np.sin(x))
    integral_estimate = (np.pi * 1.0 * points_under_curve) / n_points
    
    plt.figure(figsize=(10, 6))
    x_plot = np.linspace(0, np.pi, 1000)
    plt.plot(x_plot, np.sin(x_plot), 'b-', label='sin(x)')
    plt.scatter(x[y > np.sin(x)], y[y > np.sin(x)], c='red', alpha=0.6, label='Вне функции')
    plt.scatter(x[y <= np.sin(x)], y[y <= np.sin(x)], c='blue', alpha=0.6, label='Под функцией')
    plt.legend()
    plt.title(f'Оценка интеграла = {integral_estimate:.6f}')
    plt.savefig('images/integral_estimation.png')
    plt.close()
    
    return integral_estimate

def estimate_circle_area(radius, n_points):
    """Оценка площади круга методом Монте-Карло"""
    x = np.random.uniform(-radius, radius, n_points)
    y = np.random.uniform(-radius, radius, n_points)
    
    distances = np.sqrt(x**2 + y**2)
    points_inside = np.sum(distances <= radius)
    area_estimate = (4 * radius * radius * points_inside) / n_points
    
    plt.figure(figsize=(8, 8))
    plt.scatter(x[distances > radius], y[distances > radius], c='red', alpha=0.6)
    plt.scatter(x[distances <= radius], y[distances <= radius], c='blue', alpha=0.6)
    plt.axis('equal')
    plt.title(f'Оценка площади круга = {area_estimate:.2f}')
    plt.savefig('images/circle_area.png')
    plt.close()
    
    return area_estimate 