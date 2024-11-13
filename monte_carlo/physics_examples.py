import numpy as np
import matplotlib.pyplot as plt

def brownian_motion(n_steps):
    """Симуляция броуновского движения"""
    steps = np.random.normal(0, 1, n_steps)
    position = np.cumsum(steps)
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(n_steps), position)
    plt.title('Броуновское движение')
    plt.xlabel('Шаги')
    plt.ylabel('Положение')
    plt.savefig('images/brownian_motion.png')
    plt.close()
    
    return position

def estimate_random_walk_2d(n_steps):
    """2D случайное блуждание"""
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)
    
    angles = np.random.uniform(0, 2*np.pi, n_steps-1)
    x[1:] = np.cumsum(np.cos(angles))
    y[1:] = np.cumsum(np.sin(angles))
    
    plt.figure(figsize=(10, 10))
    plt.plot(x, y, 'b-', alpha=0.6)
    plt.plot(x[0], y[0], 'go', label='Старт')
    plt.plot(x[-1], y[-1], 'ro', label='Финиш')
    plt.title('2D случайное блуждание')
    plt.legend()
    plt.axis('equal')
    plt.savefig('images/random_walk_2d.png')
    plt.close()
    
    return x, y 