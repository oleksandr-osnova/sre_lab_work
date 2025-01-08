import numpy as np
import matplotlib.pyplot as plt
import pathlib

# Створення теки для збереження графіків
data_folder = pathlib.Path('./../../data/lab_2')
output_data_folder = data_folder / 'output'

# Параметри інтенсивності відмов
lmbd = {
    "1": 0.006, "2": 0.004, "3": 0.01, "4": 0.006, "5": 0.004
}

# Функція надійності для експоненційного розподілу
def reliability(lmbd, t):
    return np.exp(-lmbd * t)

# Функція надійності для послідовного з’єднання
def series_reliability(lmbd_list, t):
    return np.prod([np.exp(-l * t) for l in lmbd_list], axis=0)

# Функція надійності для паралельного з’єднання
def parallel_reliability(reliabilities, t):
    return 1 - np.prod([1 - r for r in reliabilities], axis=0)

# Час для аналізу
t = np.linspace(0, 1000, 100)

# Розрахунок надійності системи
R_12 = series_reliability([lmbd["1"], lmbd["2"]], t)
R_45 = series_reliability([lmbd["4"], lmbd["5"]], t)
R_3 = reliability(lmbd["3"], t)
R_system = parallel_reliability([R_12, R_3, R_45], t)

# Побудова графіка
fgr = plt.figure(figsize=(10, 6))
plt.plot(t, R_12, label="1 & 2 (послідовно)")
plt.plot(t, R_45, label="4 & 5 (послідовно)")
plt.plot(t, R_system, label="Система (паралельно)")
plt.title("Функція надійності системи")
plt.xlabel("Час t")
plt.ylabel("R(t)")
plt.legend()
plt.grid()
plt.show()
fgr.savefig(output_data_folder / "system_reliability_function.png")

# Імітаційне моделювання
def simulate_failure(lmbd, num_simulations=10000):
    times = []
    for _ in range(num_simulations):
        t1 = np.random.exponential(1/lmbd["1"])
        t2 = np.random.exponential(1/lmbd["2"])
        t3 = np.random.exponential(1/lmbd["3"])
        t4 = np.random.exponential(1/lmbd["4"])
        t5 = np.random.exponential(1/lmbd["5"])
        t_12 = min(t1, t2)  # послідовно
        t_45 = min(t4, t5)  # послідовно
        t_system = max(t_12, t3, t_45)  # паралельно
        times.append(t_system)
    return times

# Запуск імітаційного моделювання
simulated_times = simulate_failure(lmbd)
fgr = plt.figure()
plt.hist(simulated_times, bins=50, alpha=0.7, color='blue', density=True)
plt.title("Імітація часу відмови системи")
plt.xlabel("Час")
plt.ylabel("Ймовірність")
plt.show()
fgr.savefig(output_data_folder / "system_failure_time_simulation.png")
