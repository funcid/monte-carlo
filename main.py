import os
from monte_carlo.basic_examples import (
    estimate_pi, integrate_monte_carlo, estimate_circle_area
)
from monte_carlo.financial_examples import estimate_stock_price
from monte_carlo.physics_examples import (
    brownian_motion, estimate_random_walk_2d
)
from monte_carlo.probability_examples import (
    estimate_normal_distribution, estimate_birthday_problem,
    estimate_monte_hall, estimate_geometric_distribution,
    estimate_buffon_needle, estimate_coupon_collector,
    estimate_poisson_process, estimate_galton_board,
    estimate_random_walk_hitting_time, estimate_hypergeometric,
    estimate_chi_square, estimate_waiting_time, estimate_bootstrap_mean
)
from report.pdf_generator import MonteCarloReport
import numpy as np

def ensure_dir(directory):
    """Создание директории, если она не существует"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    # Создаем директорию для изображений
    ensure_dir('images')
    
    # Создаем отчет
    report = MonteCarloReport()
    report.add_introduction()
    
    # Базовые примеры
    pi_estimate = estimate_pi(10000)
    report.add_example(
        "1. Оценка числа π",
        "Метод основан на отношении площади четверти круга к площади квадрата.",
        "images/pi_estimation.png",
        f"Полученная оценка π: {pi_estimate:.6f}"
    )
    
    integral = integrate_monte_carlo(10000)
    report.add_example(
        "2. Оценка определенного интеграла",
        "Вычисление интеграла sin(x) от 0 до π методом Монте-Карло.",
        "images/integral_estimation.png",
        f"Оценка интеграла: {integral:.6f}"
    )
    
    # Физические примеры
    brownian_motion(1000)
    report.add_example(
        "3. Броуновское движение",
        "Симуляция случайного движения частицы.",
        "images/brownian_motion.png"
    )
    
    # Остальные примеры...
    circle_area = estimate_circle_area(5, 10000)
    report.add_example(
        "4. Оценка площади круга",
        "Оценка площади круга методом Монте-Карло.",
        "images/circle_area.png",
        f"Оценка площади круга радиусом 5: {circle_area:.2f}"
    )
    
    estimate_normal_distribution(10000)
    report.add_example(
        "5. Генерация нормального распределения",
        "Получение нормального распределения из равномерного.",
        "images/normal_dist.png"
    )
    
    estimate_stock_price(100, 0.1, 0.3, 1, 10)
    report.add_example(
        "6. Симуляция цен акций",
        "Моделирование движения цен акций по модели Блэка-Шоулза.",
        "images/stock_price.png"
    )
    
    estimate_random_walk_2d(1000)
    report.add_example(
        "7. 2D случайное блуждание",
        "Симуляция случайного блуждания на плоскости.",
        "images/random_walk_2d.png"
    )
    
    birthday_prob = estimate_birthday_problem(23, 10000)
    report.add_example(
        "8. Парадокс дней рождения",
        "Вероятность совпадения дней рождения в группе людей.",
        "images/birthday_problem.png",
        f"Вероятность совпадения в группе из 23 человек: {birthday_prob:.3f}"
    )
    
    stay_prob, switch_prob = estimate_monte_hall(10000)
    report.add_example(
        "9. Парадокс Монти Холла",
        "Классическая задача о выборе дверей.",
        "images/monty_hall.png",
        f"Вероятность выигрыша:\nОстаться: {stay_prob:.3f}\nПоменять выбор: {switch_prob:.3f}"
    )
    
    # Добавляем новые примеры
    mean_trials = estimate_geometric_distribution(0.3, 10000)
    report.add_example(
        "10. Геометрическое распределение",
        "Моделирование числа попыток до первого успеха.",
        "images/geometric_dist.png",
        f"Среднее число попыток: {mean_trials:.2f} (теоретическое: {1/0.3:.2f})"
    )
    
    mean_purchases = estimate_coupon_collector(6, 10000)
    report.add_example(
        "11. Задача о коллекционере",
        "Среднее число покупок для сбора полной коллекции.",
        "images/coupon_collector.png",
        f"Среднее число покупок: {mean_purchases:.1f}"
    )
    
    mean_time = estimate_random_walk_hitting_time(10, 10000)
    report.add_example(
        "12. Время достижения барьера",
        "Моделирование времени достижения заданного уровня в случайном блуждании.",
        "images/hitting_time.png",
        f"Среднее время достижения: {mean_time:.2f} (теоретическое: 100)"
    )
    
    mean_successes = estimate_hypergeometric(50, 20, 10, 10000)
    report.add_example(
        "13. Гипергеометрическое распределение",
        "Моделирование выборки без возвращения.",
        "images/hypergeometric.png",
        f"Среднее число успехов: {mean_successes:.2f}"
    )
    
    chi_square_mean = estimate_chi_square(4, 10000)
    report.add_example(
        "14. Распределение χ²",
        "Моделирование суммы квадратов нормальных величин.",
        "images/chi_square.png",
        f"Среднее значение: {chi_square_mean:.2f} (теоретическое: 4.00)"
    )
    
    mean_wait = estimate_waiting_time(0.5, 1.0, 10000)
    report.add_example(
        "15. Система массового обслуживания",
        "Моделирование очереди M/M/1.",
        "images/waiting_time.png",
        f"Среднее время ожидания: {mean_wait:.2f}"
    )
    
    # Генерируем некоторые данные для бутстрапа
    data = np.random.normal(10, 2, 100)
    bootstrap_mean, ci = estimate_bootstrap_mean(data, 10000)
    report.add_example(
        "16. Бутстрап-анализ",
        "Оценка среднего значения методом бутстрапа.",
        "images/bootstrap.png",
        f"Среднее: {bootstrap_mean:.2f}\n95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]"
    )
    
    # Сохраняем отчет
    report.save()

if __name__ == "__main__":
    main()
