import random
import csv
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools

# ПАРАМЕТРИ ГЕНЕТИЧНОГО АЛГОРИТМУ
LOWER_BOUND = -5.0
UPPER_BOUND = 5.0

POPULATION_SIZE = 100
GENERATIONS = 250
CROSSOVER_PROBABILITY = 0.9
MUTATION_PROBABILITY = 0.2
GENE_MUTATION_PROBABILITY = 0.2

BLX_ALPHA = 0.5
SBX_ETA = 15
MUTATION_SIGMA = 0.15

ELITE_SIZE = 2
RUNS = 20
SUCCESS_THRESHOLD = 0.001

# ЦІЛЬОВА ФУНКЦІЯ: SPHERE
# f(x, y) = x^2 + y^2
# Глобальний мінімум: f(0, 0) = 0
def sphere(individual):
    x, y = individual
    return (x**2 + y**2,)

# СТВОРЕННЯ ТИПІВ DEAP
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register( "attribute", random.uniform, LOWER_BOUND, UPPER_BOUND)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=2)
toolbox.register("population", tools.initRepeat,list, toolbox.individual)
toolbox.register("evaluate", sphere)
toolbox.register("select", tools.selTournament, tournsize=3)

# BLX-ALPHA CROSSOVER
def blend_bounded(ind1, ind2):
    tools.cxBlend(
        ind1,
        ind2,
        alpha=BLX_ALPHA
    )

    for individual in (ind1, ind2):
        for i in range(len(individual)):
            individual[i] = np.clip(
                individual[i],
                LOWER_BOUND,
                UPPER_BOUND
            )

    return ind1, ind2

# SIMULATED BINARY CROSSOVER (SBX)
def simulated_binary_bounded(ind1, ind2):
    return tools.cxSimulatedBinaryBounded(
        ind1,
        ind2,
        eta=SBX_ETA,
        low=LOWER_BOUND,
        up=UPPER_BOUND
    )

# МУТАЦІЯ ГАУССА
def bounded_mutation(individual):
    tools.mutGaussian(
        individual,
        mu=0,
        sigma=MUTATION_SIGMA,
        indpb=GENE_MUTATION_PROBABILITY
    )

    for i in range(len(individual)):
        individual[i] = np.clip(individual[i], LOWER_BOUND, UPPER_BOUND)
    return (individual,)

# ГЕНЕТИЧНИЙ АЛГОРИТМ
def genetic_algorithm(crossover_name, seed):
    random.seed(seed)
    np.random.seed(seed)

    # Вибір оператора схрещування
    if crossover_name == "BLX-alpha":
        toolbox.register("mate", blend_bounded)
    else:
        toolbox.register("mate", simulated_binary_bounded)

    toolbox.register("mutate", bounded_mutation)

    # Створення початкової популяції
    population = toolbox.population(n=POPULATION_SIZE)

    # Оцінювання початкової популяції
    for individual in population:
        individual.fitness.values = toolbox.evaluate(individual)

    # Збереження найкращого знайденого рішення
    hall_of_fame = tools.HallOfFame(1)
    hall_of_fame.update(population)
    history = [hall_of_fame[0].fitness.values[0]]

    # Еволюційний процес
    for generation in range(GENERATIONS):
        # Відбір батьків
        offspring = toolbox.select(population, POPULATION_SIZE - ELITE_SIZE)
        offspring = [toolbox.clone(ind) for ind in offspring]

        # Схрещування
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CROSSOVER_PROBABILITY:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Мутація
        for individual in offspring:
            if random.random() < MUTATION_PROBABILITY:
                toolbox.mutate(individual)
                del individual.fitness.values

        # Оцінювання змінених особин
        invalid_individuals = [
            ind for ind in offspring
            if not ind.fitness.valid
        ]

        for individual in invalid_individuals:
            individual.fitness.values = toolbox.evaluate(
                individual
            )

        # Елітизм: зберігаємо двох найкращих особин
        elites = tools.selBest(population, ELITE_SIZE)
        population = offspring + [
            toolbox.clone(ind)
            for ind in elites
        ]

        # Оновлення найкращого результату
        hall_of_fame.update(population)
        history.append(hall_of_fame[0].fitness.values[0])

    # Найкраще рішення запуску
    best = hall_of_fame[0]
    solution = np.array(best, dtype=float)
    fitness = float(best.fitness.values[0])

    # Видалення тимчасових операторів
    del toolbox.mate
    del toolbox.mutate
    return solution, fitness, history


# ПРОВЕДЕННЯ ЕКСПЕРИМЕНТІВ
results = {}
for crossover_name in ["BLX-alpha", "SBX"]:

    results[crossover_name] = {
        "solutions": [],
        "fitness": [],
        "histories": []
    }
    print(f"ОПЕРАТОР СХРЕЩУВАННЯ: {crossover_name}")
    for run in range(RUNS):
        solution, fitness, history = genetic_algorithm(crossover_name, seed=run + 1)
        results[crossover_name]["solutions"].append(solution)
        results[crossover_name]["fitness"].append(fitness)
        results[crossover_name]["histories"].append(history)

        # Науковий формат показує дуже малі значення
        print(
            f"Запуск {run + 1:02d}: "
            f"x = {solution[0]:.12e}, "
            f"y = {solution[1]:.12e}, "
            f"f(x,y) = {fitness:.12e}"
        )


# СТАТИСТИЧНИЙ АНАЛІЗ
summary = {}
print("ПІДСУМКИ ЕКСПЕРИМЕНТІВ")
for name, data in results.items():
    values = np.array(data["fitness"], dtype=float)
    best_value = float(np.min(values))
    mean_value = float(np.mean(values))
    median_value = float(np.median(values))
    std_value = float(np.std(values))

    successful_runs = int(np.sum(values < SUCCESS_THRESHOLD))
    exact_zeros = int(np.sum(values == 0.0))

    summary[name] = {
        "best": best_value,
        "mean": mean_value,
        "median": median_value,
        "std": std_value,
        "success": successful_runs,
        "exact_zeros": exact_zeros
    }

    print(f"\nОператор: {name}")
    print(f"Найкраще значення: {best_value:.12e}")
    print(f"Середнє значення: {mean_value:.12e}")
    print(f"Медіана: {median_value:.12e}")
    print(f"Стандартне відхилення: {std_value:.12e}")

    print(
        f"Успішні запуски (f < {SUCCESS_THRESHOLD}): "
        f"{successful_runs}/{RUNS}"
    )

    print(
        f"Точних нульових значень: "
        f"{exact_zeros}/{RUNS}"
    )

# ПОРІВНЯННЯ ОПЕРАТОРІВ

blx = summary["BLX-alpha"]
sbx = summary["SBX"]
print("ПОРІВНЯННЯ BLX-ALPHA ТА SBX")

if np.isclose(blx["mean"], sbx["mean"], atol=1e-15):
    print("Середні результати практично однакові.")
elif blx["mean"] < sbx["mean"]:
    print("BLX-alpha показав менше середнє значення функції.")
else:
    print("SBX показав менше середнє значення функції.")

if blx["success"] > sbx["success"]:
    print("BLX-alpha має більше успішних запусків.")
elif sbx["success"] > blx["success"]:
    print("SBX має більше успішних запусків.")
else:
    print("Кількість успішних запусків однакова.")

if blx["std"] < sbx["std"]:
    print("BLX-alpha має менший розкид результатів.")
elif sbx["std"] < blx["std"]:
    print("SBX має менший розкид результатів.")
else:
    print("Стандартні відхилення однакові.")

# ГРАФІК ЗБІЖНОСТІ
plt.figure(figsize=(10, 6))

for name, data in results.items():
    histories = np.array(data["histories"])
    mean_history = histories.mean(axis=0)
    plt.plot(range(GENERATIONS + 1), mean_history, label=name)

plt.xlabel("Покоління")
plt.ylabel("Найкраще значення f(x, y)")
plt.title("Порівняння збіжності генетичного алгоритму")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("convergence.png", dpi=150)
plt.show()


# BOXPLOT РЕЗУЛЬТАТІВ

plt.figure(figsize=(9, 6))
plt.boxplot([results["BLX-alpha"]["fitness"], results["SBX"]["fitness"]], tick_labels=["BLX-alpha", "SBX"])
plt.ylabel("Найкраще значення f(x, y)")
plt.title("Розподіл результатів 20 запусків")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("comparison.png", dpi=150)
plt.show()

with open("results.csv", "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Operator",
        "Run",
        "x",
        "y",
        "Fitness"
    ])

    for name, data in results.items():

        for i, (solution, fitness) in enumerate(zip(data["solutions"], data["fitness"]), start=1):
            writer.writerow([
                name,
                i,
                solution[0],
                solution[1],
                fitness
            ])


with open("summary.csv", "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Operator",
        "Best",
        "Mean",
        "Median",
        "Standard deviation",
        "Successful runs",
        "Exact zero values"
    ])

    for name, stats in summary.items():
        writer.writerow([
            name,
            stats["best"],
            stats["mean"],
            stats["median"],
            stats["std"],
            f'{stats["success"]}/{RUNS}',
            stats["exact_zeros"]
        ])

print("\nФайли успішно збережено:")
print("1. convergence.png")
print("2. comparison.png")
print("3. results.csv")
print("4. summary.csv")
