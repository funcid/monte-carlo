from fpdf import FPDF

class MonteCarloReport:
    def __init__(self, font_path='fonts/RobotoMono[wght].ttf'):
        self.pdf = FPDF()
        self.font_path = font_path
        self._setup_pdf()
    
    def _setup_pdf(self):
        """Настройка базовых параметров PDF"""
        self.pdf.add_font('RobotoMono', '', self.font_path, uni=True)
        self.pdf.set_font('RobotoMono', '', 10)
    
    def add_introduction(self):
        """Добавление введения в отчет"""
        self.pdf.add_page()
        self.pdf.set_font('RobotoMono', '', 14)
        self.pdf.cell(0, 10, 'Метод Монте-Карло: Теория и Примеры', ln=True, align='C')
        
        # Добавляем информацию об авторе
        self.pdf.ln(2)
        self.pdf.set_font('RobotoMono', '', 10)
        self.pdf.cell(0, 10, 'Царюк Артем Владимирович', ln=True)
        self.pdf.cell(0, 10, '21 ИТ-МО', ln=True)
        self.pdf.ln(10)
        
        self.pdf.set_font('RobotoMono', '', 10)
        intro_text = """Метод Монте-Карло - это широкий класс вычислительных алгоритмов, основанных на многократной случайной выборке для получения численных результатов. Основная идея метода заключается в использовании случайности для решения задач, которые могут быть детерминированными в принципе.

Ключевые особенности метода:
1. Использование случайных чисел
2. Многократное повторение эксперимента
3. Статистическая обработка результатов
4. Повышение точности с увеличением числа испытаний

Метод применяется в:
- Физике (моделирование частиц, квантовая механика)
- Математике (вычисление интегралов, оптимизация)
- Экономике (оценка рисков, прогнозирование)
- Оптимизации (поиск глобальных минимумов)
- Теории вероятностей (проверка гипотез)

История метода:
Метод был разработан в 1940-х годах в рамках Манхэттенского проекта. Название "Монте-Карло" предложил Николас Метрополис, вдохновленный казино в Монако, где дядя его коллеги Станислава Улама часто играл в рулетку.
"""
        self.pdf.multi_cell(0, 10, intro_text)
    
    def add_example(self, title, description, image_path, results=None):
        """Добавление примера в отчет"""
        self.pdf.add_page()
        self.pdf.set_font('RobotoMono', '', 12)
        self.pdf.cell(0, 10, title, ln=True)
        self.pdf.set_font('RobotoMono', '', 10)
        
        descriptions = {
            "1. Оценка числа π": """Метод основан на отношении площади четверти круга к площади квадрата. Случайно генерируются точки в квадрате 1x1, и подсчитывается доля точек, попавших в четверть круга радиусом 1.

Теоретическое значение π = 3.141592653589793...""",

            "2. Оценка определенного интеграла": """Вычисление интеграла sin(x) от 0 до π методом Монте-Карло. Метод основан на оценке площади под кривой путем случайного выбра точек.

Теоретическое значение интеграла = 2.000000...""",

            "3. Броуновское движение": """Броуновское движение - это случайное движение частиц, взвешенных в жидкости или газе. Названо в честь ботаника Роберта Броуна, впервые описавшего это явление в 1827 году.

Модель описывает множество физических процессов, от движения молекул до колебаний цен на финансовых рынках.""",

            "4. Оценка площади круга": """Оценка площади круга методом Монте-Карло. Точки случайно размещаются в квадрате, описанном вокруг круга.

Теоретическая площадь круга радиусом r = πr²
Для r = 5: площадь = 78.539816...""",

            "5. Генерация нормального распределения": """Демонстрация центральной предельной теоремы: сумма множества независимых случайных величин стремится к нормальному распределению.

График показывает сравнение полученного распределения с теоретической кривой нормального распределения.""",

            "6. Симуляция цен акций": """Моделирование движения цен акций по модели Блэка-Шоулза. Модель описывает изменение цены с учетом волатильности и тренда.

Параметры модели:
- Начальная цена (S0)
- Тренд (μ)
- Волатильность (σ)""",

            "7. 2D случайное блуждание": """Двумерное случайное блуждание - обобщение одномерного случайного процесса. На каждом шаге частица движется в случайном направлении.

Среднее расстояние от начала координат после n шагов пропорционально √n.""",

            "8. Парадокс дней рождения": """Вероятность того, что в группе из n человек у кого-то совпадут дни рождения. 
            
Для группы из 23 человек теоретическая вероятность ≈ 0.507
Для группы из 30 человек ≈ 0.706
Для группы из 50 человек ≈ 0.970""",

            "9. Парадокс Монти Холла": """Классическая вероятностная задача, основанная на телеигре. Игрок выбирает одну из трех дверей, за одной из которых приз. Ведущий открывает пустую дверь и предлагает игроку изменить выбор.

Теоретические вероятности:
- Остаться при своем выборе: 1/3 (≈ 0.333)
- Изменить выбор: 2/3 (≈ 0.667)""",

            "10. Геометрическое распределение": """Моделирование геометрического распределения - распределения числа испытаний Бернулли до первого успеха.

Теоретическое среднее значение для p = 0.3:
E(X) = 1/p = 3.333...""",

            "11. Задача Бюффона": """Классический метод оценки числа π путем бросания иглы на разлинованную поверхность. 
            
Вероятность пересечения иглой линии связана с числом π через формулу:
P(пересечение) = (2L)/(πd), где:
L - длина иглы
d - расстояние между линиями""",

            "12. Задача о коллекционере": """Классическая вероятностная задача: сколько нужно сделать покупок, чтобы собрать полную коллекцию из n различных предметов?

Теоретическое среднее число покупок для n типов:
E(X) = n * (1 + 1/2 + 1/3 + ... + 1/n)""",

            "10. Пуассоновский процесс": """Пуассоновский процесс описывает случайные события, происходящие с постоянной средней интенсивностью λ.

Теоретическое среднее число событий за ��ремя T:
E(N(T)) = λT

Примеры применения:
- Число клиентов в очереди
- Распад радиоактивных частиц
- Число аварий на участке дороги""",

            "11. Доска Гальтона": """Механическое устройство, демонстрирующее биномиальное распределение. Шарики падают через систему штифтов, на каждом уровне случайно отклоняясь влево или вправо.

При большом числе шариков их распределение стремится к нормальному (демонстрация центральной предельной теоремы).""",

            "12. Время достижения барьера": """Исследование времени, необходимого частице в случайном блуждании для достижения заданного уровня (барьера).

Теоретическое среднее время достижения барьера h:
E(T) = h²

Это классическая задача теории случайных процессов.""",

            "13. Гипергеометрическое распределение": """Моделирование выборки без возвращения из конечной популяции.

Параметры:
N - размер популяции
K - число успешных элементов
n - размер выборки

Применяется в:
- Контроле качества
- Экологии
- Социологических исследованиях""",

            "14. Распределение χ²": """Распределение суммы квадратов независимых стандартных нормальных величин.

Степени свободы (df) определяют форму распределения.

Применяется в:
- Проверке гипотез
- Анализе категориальных данных
- Оценке качества подгонки""",

            "15. Система массового обслуживания": """Моделирование очереди типа M/M/1 (пуассоновский поток заявок, экспоненциальное время обслуживания, один сервер).

Параметры системы:
- λ (интенсивность прибытия)
- μ (интенсивность обслуживания)
- ρ = λ/μ (коэффициент загрузки)""",

            "16. Бутстрап-анализ": """Метод статистического вывода, основанный на многократной генерации выборок из имеющихся данных.

Позволяет:
- Оценивать параметры распределения
- Строить доверительные интервалы
- Проводить статистические тесты

Не требует предположений о форме распределения."""
        }
        
        if description:
            self.pdf.multi_cell(0, 10, descriptions.get(title, description))
        if results:
            self.pdf.ln(5)
            self.pdf.multi_cell(0, 10, results)
        if image_path:
            self.pdf.image(image_path, x=10, w=190)
    
    def save(self, output_path='monte_carlo_report.pdf'):
        """Сохранение отчета"""
        self.pdf.output(output_path) 