import math
import numpy as np
import random
from sklearn.linear_model import LinearRegression

def euclidean_distance(city1, city2):
    x1, y1 = city1
    x2, y2 = city2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    
def calculate_total_distance(cities, solution):
    total_distance = 0
    for i in range(len(solution) - 1):
        total_distance += euclidean_distance(cities[solution[i]], cities[solution[i+1]])
    total_distance += euclidean_distance(cities[solution[-1]], cities[solution[0]])
    return total_distance

def perform_cuckoo_search(cities, num_cuckoos=25, num_iterations=1000, pa=0.25):
    # Set parameters
    n = len(cities)
    pop_size = 10
    max_iter = 1000
    pa = 0.25

    # Initialize population
    population = [random.sample(range(n), n) for _ in range(pop_size)]
    population.sort(key=lambda sol: total_distance(cities, sol))

    # Start iterations
    for iter in range(max_iter):
        # Generate new solution
        new_solution = population[0][:]
        i, j = random.sample(range(n), 2)
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]

        # Calculate fitness
        new_distance = total_distance(cities, new_solution)

        # Perform cuckoo search
        worst_index = random.randint(0, pop_size - 1)
        if new_distance < total_distance(cities, population[worst_index]):
            population[worst_index] = new_solution[:]
            population.sort(key=lambda sol: total_distance(cities, sol))

        # Perform abandon
        for i in range(pop_size):
            if random.random() < pa:
                population[i] = random.sample(range(n), n)
                population.sort(key=lambda sol: total_distance(cities, sol))

    # Return best solution
    best_solution = population[0]
    best_distance = total_distance(cities, best_solution)
    return best_solution, best_distance

def two_opt(solution, cities):
    best_distance = calculate_total_distance(cities, solution)
    improved = True

    while improved:
        improved = False
        for i in range(len(solution) - 2):
            for j in range(i + 2, len(solution)):
                new_solution = solution.copy()
                new_solution[i+1:j+1] = reversed(new_solution[i+1:j+1])
                new_distance = calculate_total_distance(cities, new_solution)
                if new_distance < best_distance:
                    solution = new_solution
                    best_distance = new_distance
                    improved = True

    return solution

def predict_pa(num_cities):
    X = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]).reshape(-1, 1)
    y = np.array([0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
    model = LinearRegression().fit(X, y)
    return model.predict(np.array([[num_cities]]))[0]

if __name__ == "__main__":
    cities = [
        (0, 0),
        (1, 5),
        (5, 5),
        (6, 3),
        (8, 1),
        (3, 2)
    ]

    num_cities = len(cities)
    pa = predict_pa(num_cities)
    best_solution, best_distance = perform_cuckoo_search(cities, pa=pa)
    refined_solution = two_opt(best_solution, cities)
    refined_distance = calculate_total_distance(cities, refined_solution)

    print("Best solution before 2-opt:", best_solution)
    print("Best distance before 2-opt:", best_distance)
    print("Best solution after 2-opt:", refined_solution)
    print("Best distance after 2-opt:", refined_distance)
