import numpy as np
import matplotlib.pyplot as plt

def estimate_stock_price(S0, mu, sigma, T, n_paths):
    """Симуляция цен акций (модель Блэка-Шоулза)"""
    dt = 0.01
    t = np.arange(0, T, dt)
    paths = np.zeros((n_paths, len(t)))
    paths[:, 0] = S0
    
    for i in range(1, len(t)):
        dW = np.random.normal(0, np.sqrt(dt), n_paths)
        paths[:, i] = paths[:, i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
    
    plt.figure(figsize=(10, 6))
    for i in range(min(5, n_paths)):
        plt.plot(t, paths[i, :])
    plt.title('Симуляция цен акций')
    plt.xlabel('Время')
    plt.ylabel('Цена')
    plt.savefig('images/stock_price.png')
    plt.close()
    
    return paths 