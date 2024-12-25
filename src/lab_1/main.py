import numpy as np
from scipy.stats import kstest, t, norm, chi2, ttest_1samp
from statsmodels.distributions.empirical_distribution import ECDF
import pathlib
import math
import time
import matplotlib.pyplot as plt

# Створення теки для збереження графіків
data_folder = pathlib.Path('./../../data/lab_1')
output_data_folder = data_folder / 'output'

# Крок 1: Генерація рівномірно розподілених випадкових чисел за заданою формулою
def generate_uniform_random_numbers_with_formula(n):
    # Константи для формули
    modulus = 10**11
    k = 2
    multiplier = 7**(4 * k + 1)

    # Початкове значення на основі системного часу
    b = int(time.time() * 1000) % modulus

    uniform_numbers = []
    for _ in range(n):
        b = (multiplier * b) % modulus
        r = b / modulus  # Перетворення у діапазон [0, 1]
        uniform_numbers.append(r)

    return uniform_numbers

# Крок 2: Генерація нормально розподілених випадкових чисел за методом полярних координат
def generate_normal_random_numbers(n):
    results = []
    while len(results) < n:
        u1, u2 = np.random.uniform(0, 1), np.random.uniform(0, 1)
        v1, v2 = 2 * u1 - 1, 2 * u2 - 1
        s = v1**2 + v2**2
        if s >= 1 or s == 0:
            continue
        multiplier = math.sqrt(-2 * math.log(s) / s)
        results.append(v1 * multiplier)
        if len(results) < n:
            results.append(v2 * multiplier)
    return results

# Крок 3: Тести для перевірки якості генерованих чисел

def aperiodicity_test(data):
    """Перевірка аперіодичності послідовності"""
    n = len(data)
    for period in range(1, n // 2):
        if np.allclose(data[:n-period], data[period:n]):
            return False
    return True

def moments_test(data):
    """Перевірка збігу моментів: середнього та дисперсії"""
    mean = np.mean(data)
    variance = np.var(data, ddof=1)
    expected_mean = 0.5
    expected_variance = 1 / 12
    return abs(mean - expected_mean) < 0.01 and abs(variance - expected_variance) < 0.01


def covariance_test(data, alpha=0.05):
    """Перевірка коваріації між сусідніми значеннями в послідовності"""
    n = len(data)
    mean = np.mean(data)
    covariance = np.cov(data[:-1], data[1:])[0, 1]
    variance = np.var(data, ddof=1)

    # Обчислення критичного значення
    t_crit = t.ppf(1 - alpha / 2, df=n - 2)
    r_crit = t_crit / np.sqrt(n - 2 + t_crit**2)
    max_covariance = r_crit * variance

    return abs(covariance) < max_covariance

def kolmogorov_smirnov_test(data):
    """Тест Колмогорова-Смірнова для нормального розподілу"""
    statistic, p_value = kstest(data, 'norm')
    return statistic, p_value

def hypothesis_test_normal(data, theoretical_mean, theoretical_std):
    """Перевірка гіпотези для нормального розподілу: середнє та дисперсія"""
    # Перевірка середнього значення
    t_statistic, t_p_value = ttest_1samp(data, theoretical_mean)

    # Перевірка дисперсії
    variance = np.var(data, ddof=1)
    n = len(data)
    chi2_statistic = (n - 1) * variance / (theoretical_std**2)
    chi2_p_value = chi2.sf(chi2_statistic, df=n-1)

    mean_result = "Пройдено" if t_p_value > 0.05 else "Не пройдено"
    variance_result = "Пройдено" if chi2_p_value > 0.05 else "Не пройдено"

    return {
        "mean_test": {"t_statistic": t_statistic, "p_value": t_p_value, "result": mean_result},
        "variance_test": {"chi2_statistic": chi2_statistic, "p_value": chi2_p_value, "result": variance_result}
    }

# Крок 4: Візуалізація послідовностей і тестів

def plot_sequence(data, title, filename):
    """Побудова графіка для візуалізації послідовності."""
    fgr = plt.figure(filename, figsize=(10, 5))
    plt.plot(data, marker='o', linestyle='')
    plt.title(title)
    plt.xlabel('Індекс')
    plt.ylabel('Значення')
    plt.grid()
    plt.show()
    fgr.savefig(output_data_folder / filename)

def plot_statistics(results):
    """Побудова графіків для оцінки середнього, дисперсії та результатів тестів"""
    results = np.array(results)
    fgr = plt.figure()
    plt.plot(results[:, 0], results[:, 1], label="Середнє")
    plt.plot(results[:, 0], results[:, 2], label="Середньоквадратичне")
    plt.title("Середнє та дисперсія для різних N")
    plt.xlabel('N')
    plt.ylabel('Значення')
    plt.legend()
    plt.grid()
    plt.show()
    fgr.savefig(output_data_folder / "mean_variance.png")

    fgr = plt.figure()
    plt.plot(results[:, 0], results[:, 3], label="Аперіодичність")
    plt.plot(results[:, 0], results[:, 4], label="Збіг моментів")
    plt.plot(results[:, 0], results[:, 5], label="Коваріація")
    plt.title("Результати тестів для різних N")
    plt.xlabel('N')
    plt.ylabel('Результат тесту (1 - пройдено, 0 - не пройдено)')
    plt.legend()
    plt.grid()
    plt.show()
    fgr.savefig(output_data_folder / "tests_results.png")

def plot_normal_fit(data, theoretical_mean, theoretical_std, filename):
    """Графік порівняння емпіричних даних із теоретичною щільністю нормального розподілу"""
    x = np.linspace(min(data), max(data), 1000)
    pdf = norm.pdf(x, theoretical_mean, theoretical_std)

    fgr = plt.figure(filename, figsize=(10, 6))
    plt.hist(data, bins=30, density=True, alpha=0.6, color='skyblue', label='Емпіричні дані')
    plt.plot(x, pdf, 'r', lw=2, label='Теоретичний нормальний розподіл')
    plt.title("Емпіричний vs Теоретичний нормальний розподіл")
    plt.xlabel("Значення")
    plt.ylabel("Щільність")
    plt.legend()
    plt.grid()
    plt.show()
    fgr.savefig(output_data_folder / filename)


def plot_cdf_comparison(data, theoretical_mean, theoretical_std, filename):
    """Порівняння емпіричної та теоретичної функцій розподілу (CDF)."""
    ecdf = ECDF(data)
    x = np.linspace(min(data), max(data), 1000)
    cdf = norm.cdf(x, loc=theoretical_mean, scale=theoretical_std)

    fgr = plt.figure(filename, figsize=(10, 6))
    plt.plot(x, ecdf(x), label="Емпірична CDF", color='blue')
    plt.plot(x, cdf, label="Теоретична CDF", linestyle="--", color='red')
    plt.title("Порівняння емпіричної і теоретичної CDF")
    plt.xlabel("Значення")
    plt.ylabel("Ймовірність")
    plt.legend()
    plt.grid()
    fgr.savefig(output_data_folder /filename)
    plt.show()

# Крок 5: Оцінка обсягу N для заданої точності

def evaluate_sample_size():
    """Оцінка якості при різних значеннях вибірки N"""
    results = []
    for n in range(50, 1050, 50):
        uniform_numbers = generate_uniform_random_numbers_with_formula(n)
        mean = np.mean(uniform_numbers)
        variance = np.var(uniform_numbers, ddof=1)
        aperiodicity = 1 if aperiodicity_test(uniform_numbers) else 0
        moments = 1 if moments_test(uniform_numbers) else 0
        covariance = 1 if covariance_test(uniform_numbers) else 0
        results.append((n, mean, variance, aperiodicity, moments, covariance))
    return results

# Генерація послідовностей і тестування
N = 1000
uniform_numbers = generate_uniform_random_numbers_with_formula(N)
normal_numbers = generate_normal_random_numbers(N)

# Візуалізація
plot_sequence(uniform_numbers, "Рівномірно розподілені випадкові числа (задана формула)", "uniform_numbers.png")
plot_sequence(normal_numbers, "Нормально розподілені випадкові числа", "normal_numbers.png")

# Перевірка якості рівномірно розподілених чисел
is_aperiodic = aperiodicity_test(uniform_numbers)
is_moments_ok = moments_test(uniform_numbers)
is_covariance_ok = covariance_test(uniform_numbers)
ks_statistic, ks_p_value = kolmogorov_smirnov_test(normal_numbers)

print(f"Результати тестів для рівномірного розподілу:")
print(f"Аперіодичність: {'Пройдено' if is_aperiodic else 'Не пройдено'}")
print(f"Збіг моментів: {'Пройдено' if is_moments_ok else 'Не пройдено'}")
print(f"Коваріація: {'Пройдено' if is_covariance_ok else 'Не пройдено'}")
print(f"Результати тесту Колмогорова-Смірнова для нормального розподілу:")
print(f"Статистика: {ks_statistic}, p-значення: {ks_p_value}")

# Перевірка гіпотези для нормального розподілу
theoretical_mean = 0
theoretical_std = 1
normal_test_results = hypothesis_test_normal(normal_numbers, theoretical_mean, theoretical_std)

print("\nПеревірка гіпотези про збіг параметрів моделювання:")
print(f"Середнє значення: t-статистика = {normal_test_results['mean_test']['t_statistic']}, "
      f"p-значення = {normal_test_results['mean_test']['p_value']}, Результат = {normal_test_results['mean_test']['result']}")
print(f"Дисперсія: хі-квадрат статистика = {normal_test_results['variance_test']['chi2_statistic']}, "
      f"p-значення = {normal_test_results['variance_test']['p_value']}, Результат = {normal_test_results['variance_test']['result']}")

# Оцінка обсягу N для заданої точності
sample_size_results = evaluate_sample_size()
plot_statistics(sample_size_results)

# Побудова графіка порівняння нормального розподілу
plot_normal_fit(normal_numbers, theoretical_mean, theoretical_std, "normal_fit.png")

# Побудова графіка порівняння CDF
plot_cdf_comparison(normal_numbers, theoretical_mean, theoretical_std, "cdf_comparison.png")

