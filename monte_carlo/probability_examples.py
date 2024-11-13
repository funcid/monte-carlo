import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def estimate_normal_distribution(n_points):
    """Генерация нормального распределения методом Монте-Карло"""
    # Используем метод Бокса-Мюллера для генерации нормального распределения
    u1 = np.random.uniform(0, 1, n_points)
    u2 = np.random.uniform(0, 1, n_points)
    
    # Преобразование равномерного распределения в нормальное
    z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    
    plt.figure(figsize=(12, 7))
    # Строим гистограмму сгенерированных значений
    plt.hist(z1, bins=50, density=True, alpha=0.8, color='royalblue', 
             label='Смоделированное распределение')
    
    # Теоретическая кривая стандартного нормального распределения
    x = np.linspace(-4, 4, 1000)
    plt.plot(x, stats.norm.pdf(x, 0, 1), 'r-', lw=3, 
            label='Теоретическое нормальное\nраспределение')
    
    plt.title('Генерация нормального распределения\nметодом Монте-Карло', 
             fontsize=12, pad=20)
    plt.xlabel('Значение', fontsize=10)
    plt.ylabel('Плотность вероятности', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    plt.xlim(-4, 4)
    plt.savefig('images/normal_dist.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return z1

def estimate_birthday_problem(n_people, n_simulations):
    """Парадокс дней рождения"""
    matches = 0
    for _ in range(n_simulations):
        birthdays = np.random.randint(0, 365, n_people)
        if len(np.unique(birthdays)) < len(birthdays):
            matches += 1
    
    probability = matches / n_simulations
    
    plt.figure(figsize=(10, 6))
    plt.bar(['Совпадения', 'Нет совпадений'], 
            [probability, 1-probability],
            color=['green', 'red'])
    plt.title(f'Вероятность совпадения дней рождения: {probability:.3f}')
    plt.savefig('images/birthday_problem.png')
    plt.close()
    
    return probability

def estimate_monte_hall(n_simulations):
    """Парадокс Монти Холла"""
    stay_wins = 0
    switch_wins = 0
    
    for _ in range(n_simulations):
        doors = [1, 2, 3]
        prize_door = np.random.choice(doors)
        initial_choice = np.random.choice(doors)
        
        remaining_doors = [d for d in doors if d != initial_choice and d != prize_door]
        opened_door = np.random.choice(remaining_doors)
        
        if initial_choice == prize_door:
            stay_wins += 1
        else:
            switch_wins += 1
    
    stay_prob = stay_wins / n_simulations
    switch_prob = switch_wins / n_simulations
    
    plt.figure(figsize=(10, 6))
    plt.bar(['Остаться', 'Поменять'], [stay_prob, switch_prob])
    plt.title('Парадокс Монти Холла')
    plt.ylabel('Вероятность выигрыша')
    plt.savefig('images/monty_hall.png')
    plt.close()
    
    return stay_prob, switch_prob

def estimate_geometric_distribution(p, n_simulations):
    """Моделирование геометрического распределения"""
    # Моделируем количество попыток до первого успеха
    trials = []
    for _ in range(n_simulations):
        count = 1
        while np.random.random() > p:
            count += 1
        trials.append(count)
    
    plt.figure(figsize=(12, 7))
    # Строим гистограмму
    plt.hist(trials, bins=range(1, max(trials) + 2), density=True, 
             alpha=0.8, color='royalblue', label='Смоделированное распределение')
    
    # Теоретическое распределение
    x = np.arange(1, max(trials) + 1)
    theoretical = p * (1-p)**(x-1)
    plt.plot(x, theoretical, 'r-', lw=3, label='Теоретическое распределение')
    
    plt.title(f'Геометрическое распределение (p={p})')
    plt.xlabel('Число попыток до успеха')
    plt.ylabel('Вероятность')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('images/geometric_dist.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return np.mean(trials)

def estimate_buffon_needle(n_needles, L=1.0, d=2.0):
    """Задача Бюффона о бросании иглы"""
    # L - длина иглы, d - расстояние между параллельными линиями
    # Генерируем случайные позиции центров игл и углы
    y = np.random.uniform(0, d/2, n_needles)  # расстояние до ближайшей линии
    theta = np.random.uniform(0, np.pi, n_needles)  # угол наклона иглы
    
    # Подсчет пересечений
    # Игла пересекает линию, если y ≤ (L/2)*sin(θ)
    crosses = np.sum(y <= (L/2) * np.sin(theta))
    
    # Формула для оценки π: π ≈ (2*L*n)/(d*crosses)
    pi_estimate = (2 * L * n_needles) / (d * crosses) if crosses > 0 else float('inf')
    
    # Визуализация
    plt.figure(figsize=(12, 7))
    # Разделяем точки на пересекающие и не пересекающие линии
    crossing = y <= (L/2) * np.sin(theta)
    
    plt.scatter(theta[~crossing], y[~crossing],
               c='blue', alpha=0.6, label='Не пересекает линию')
    plt.scatter(theta[crossing], y[crossing],
               c='red', alpha=0.6, label='Пересекает линию')
    
    # Добавляем теоретическую кривую
    theta_curve = np.linspace(0, np.pi, 1000)
    plt.plot(theta_curve, (L/2) * np.sin(theta_curve), 'g-', 
            label='Теоретическая граница', alpha=0.8)
    
    plt.title(f'Задача Бюффона: оценка π = {pi_estimate:.6f}\n'
             f'(Теоретическое π = 3.141593)', pad=20)
    plt.xlabel('Угол θ')
    plt.ylabel('Расстояние до линии')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('images/buffon_needle.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return pi_estimate

def estimate_coupon_collector(n_types, n_simulations):
    """Задача о коллекционере купонов"""
    collection_sizes = []
    
    for _ in range(n_simulations):
        collected = set()
        draws = 0
        while len(collected) < n_types:
            draws += 1
            coupon = np.random.randint(0, n_types)
            collected.add(coupon)
        collection_sizes.append(draws)
    
    plt.figure(figsize=(12, 7))
    plt.hist(collection_sizes, bins=30, density=True, alpha=0.8,
             color='royalblue', label='Смоделированное распределение')
    
    plt.title(f'Задача о коллекционере купонов\n(n={n_types} типов)')
    plt.xlabel('Количество покупок до полной коллекции')
    plt.ylabel('Частота')
    plt.grid(True, alpha=0.3)
    
    # Теоретическое среднее
    theoretical_mean = n_types * sum(1/i for i in range(1, n_types + 1))
    plt.axvline(theoretical_mean, color='r', linestyle='--',
                label=f'Теоретическо�� среднее: {theoretical_mean:.1f}')
    
    plt.legend()
    plt.savefig('images/coupon_collector.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return np.mean(collection_sizes)

def estimate_poisson_process(lambda_rate, T, n_simulations):
    """Моделирование пуассоновского процесса"""
    times = []
    for _ in range(n_simulations):
        t = 0
        events = []
        while t < T:
            # Время до следующего события
            dt = np.random.exponential(1/lambda_rate)
            t += dt
            if t < T:
                events.append(t)
        times.append(len(events))
    
    plt.figure(figsize=(12, 7))
    # Гистограмма числа событий
    plt.hist(times, bins=range(min(times), max(times) + 2), density=True,
             alpha=0.8, color='royalblue', label='Смоделированное распределение')
    
    # Теоретическое распределение Пуассона
    x = np.arange(min(times), max(times) + 1)
    theoretical = stats.poisson.pmf(x, lambda_rate * T)
    plt.plot(x, theoretical, 'r-', lw=3, label='Теоретическое распределение')
    
    plt.title(f'Пуассоновский процесс (λ={lambda_rate}, T={T})')
    plt.xlabel('Число событий')
    plt.ylabel('Вероятность')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('images/poisson_process.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return np.mean(times)

def estimate_galton_board(n_balls, n_levels):
    """Моделирование доски Гальтона"""
    final_positions = []
    for _ in range(n_balls):
        position = 0
        for _ in range(n_levels):
            # На каждом уровне шарик идет влево или вправо
            position += np.random.choice([-1, 1])
        final_positions.append(position)
    
    plt.figure(figsize=(12, 7))
    
    # Вычисляем границы для гистограммы
    min_pos = min(final_positions)
    max_pos = max(final_positions)
    bins = range(min_pos, max_pos + 2)  # +2 чтобы включить последний бин
    
    # Гистограмма конечных позиций
    plt.hist(final_positions, bins=bins, density=True, alpha=0.8,
             color='royalblue', label='Смоделированное распределение',
             align='left')  # Выравнивание по левому краю для соответствия теории
    
    # Теоретическое биномиальное распределение
    x = np.arange(min_pos, max_pos + 1)
    # Сдвигаем распределение для соответствия с данными
    theoretical = stats.binom.pmf(x - min_pos, n_levels, 0.5)
    plt.plot(x, theoretical, 'r-', lw=3, label='Теоретическое распределение')
    
    plt.title(f'Доска Гальтона (n_levels={n_levels})')
    plt.xlabel('Конечная позиция')
    plt.ylabel('Вероятность')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('images/galton_board.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return np.mean(final_positions)

def estimate_random_walk_hitting_time(barrier, n_simulations):
    """Моделирование времени достижения барьера в случайном блуждании"""
    hitting_times = []
    for _ in range(n_simulations):
        position = 0
        time = 0
        while abs(position) < barrier:
            position += np.random.choice([-1, 1])
            time += 1
        hitting_times.append(time)
    
    plt.figure(figsize=(12, 7))
    plt.hist(hitting_times, bins=30, density=True, alpha=0.8,
             color='royalblue', label='Смоделированное распределение')
    
    plt.title(f'Время достижения барьера ±{barrier}')
    plt.xlabel('Время достижения')
    plt.ylabel('Частота')
    plt.grid(True, alpha=0.3)
    
    # Теоретическое среднее время
    theoretical_mean = barrier**2
    plt.axvline(theoretical_mean, color='r', linestyle='--',
                label=f'Теоретическое среднее: {theoretical_mean}')
    
    plt.legend()
    plt.savefig('images/hitting_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return np.mean(hitting_times)

def estimate_hypergeometric(N, K, n, n_simulations):
    """Моделирование гипергеометрического распределения"""
    successes = []
    for _ in range(n_simulations):
        # Создаем популяцию из N элементов, где K - успешные
        population = np.array([1] * K + [0] * (N - K))
        # Выбираем n элементов без возвращения
        sample = np.random.choice(population, size=n, replace=False)
        successes.append(np.sum(sample))
    
    plt.figure(figsize=(12, 7))
    # Гистограмма результатов
    plt.hist(successes, bins=range(min(successes), max(successes) + 2),
             density=True, alpha=0.8, color='royalblue',
             label='Смоделированное распределение')
    
    # Теоретическое распределение
    x = np.arange(max(0, n - (N - K)), min(n, K) + 1)
    theoretical = stats.hypergeom.pmf(x, N, K, n)
    plt.plot(x, theoretical, 'r-', lw=3, label='Теоретическое распределение')
    
    plt.title(f'Гипергеометрическое распределение\n(N={N}, K={K}, n={n})')
    plt.xlabel('Число успехов')
    plt.ylabel('Вероятность')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('images/hypergeometric.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return np.mean(successes)

def estimate_chi_square(df, n_simulations):
    """Моделирование распределения хи-квадрат"""
    # Генерируем df стандартных нормальных величин и возводим в квадрат
    values = np.sum(np.random.normal(0, 1, (n_simulations, df))**2, axis=1)
    
    plt.figure(figsize=(12, 7))
    plt.hist(values, bins=50, density=True, alpha=0.8,
             color='royalblue', label='Смоделированное распределение')
    
    # Теоретическое распределение
    x = np.linspace(0, max(values), 1000)
    theoretical = stats.chi2.pdf(x, df)
    plt.plot(x, theoretical, 'r-', lw=3, label='Теоретическое распределение')
    
    plt.title(f'Распределение χ² (df={df})')
    plt.xlabel('Значение')
    plt.ylabel('Плотность вероятности')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('images/chi_square.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return np.mean(values)

def estimate_waiting_time(arrival_rate, service_rate, n_simulations):
    """Моделирование времени ожидания в системе массового обслуживания M/M/1"""
    waiting_times = []
    for _ in range(n_simulations):
        # Генерируем интервалы между прибытиями
        arrivals = np.random.exponential(1/arrival_rate, 1000)
        # Генерируем времена обслуживания
        service_times = np.random.exponential(1/service_rate, 1000)
        
        # Вычисляем времена ожидания
        arrival_times = np.cumsum(arrivals)
        completion_times = np.zeros_like(arrival_times)
        waiting_time = 0
        
        for i in range(len(arrival_times)):
            service_start = max(arrival_times[i], 
                              completion_times[i-1] if i > 0 else 0)
            completion_times[i] = service_start + service_times[i]
            waiting_time = service_start - arrival_times[i]
            waiting_times.append(waiting_time)
    
    plt.figure(figsize=(12, 7))
    plt.hist(waiting_times, bins=50, density=True, alpha=0.8,
             color='royalblue', label='Смоделированное распределение')
    
    # Теоретическое среднее время ожидания
    rho = arrival_rate/service_rate
    theoretical_mean = rho/(service_rate * (1 - rho))
    plt.axvline(theoretical_mean, color='r', linestyle='--',
                label=f'Теоретическое среднее: {theoretical_mean:.2f}')
    
    plt.title(f'Время ожидания в очереди M/M/1\n(ρ={rho:.2f})')
    plt.xlabel('Время ожидания')
    plt.ylabel('Плотность')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('images/waiting_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return np.mean(waiting_times)

def estimate_bootstrap_mean(data, n_bootstrap):
    """Бутстрап-оценка среднего значения и доверительного интервала"""
    bootstrap_means = []
    for _ in range(n_bootstrap):
        # Случайная выборка с возвращением
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    plt.figure(figsize=(12, 7))
    plt.hist(bootstrap_means, bins=50, density=True, alpha=0.8,
             color='royalblue', label='Распределение бутстрап-оценок')
    
    # Доверительный интервал
    ci_lower, ci_upper = np.percentile(bootstrap_means, [2.5, 97.5])
    plt.axvline(np.mean(data), color='r', linestyle='-',
                label=f'Исходное среднее: {np.mean(data):.2f}')
    plt.axvline(ci_lower, color='g', linestyle='--',
                label=f'95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]')
    plt.axvline(ci_upper, color='g', linestyle='--')
    
    plt.title('Бутстрап-оценка среднего значения')
    plt.xlabel('Среднее значение')
    plt.ylabel('Плотность')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('images/bootstrap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return np.mean(bootstrap_means), (ci_lower, ci_upper)