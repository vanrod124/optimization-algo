import numpy as np
import random

def euclidean_distance(city1, city2):
    return np.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

def calculate_total_distance(cities, solution):
    total_distance = 0
    for i in range(len(solution) - 1):
        total_distance += euclidean_distance(cities[solution[i]], cities[solution[i+1]])
    total_distance += euclidean_distance(cities[solution[-1]], cities[solution[0]])
    return total_distance

def generate_random_solution(num_cities):
    solution = list(range(num_cities))
    random.shuffle(solution)
    return solution

def perform_cuckoo_search(cities, num_cuckoos=25, num_iterations=1000, pa=0.25):
    num_cities = len(cities)

    # Generate initial cuckoo solutions
    cuckoos = [generate_random_solution(num_cities) for _ in range(num_cuckoos)]

    # Calculate the total distances of the initial solutions
    distances = [calculate_total_distance(cities, cuckoo) for cuckoo in cuckoos]

    # Main loop
    for _ in range(num_iterations):
        for i in range(num_cuckoos):
            # Generate a new solution
            new_solution = list(cuckoos[i])
            idx1, idx2 = random.sample(range(num_cities), 2)
            new_solution[idx1], new_solution[idx2] = new_solution[idx2], new_solution[idx1]
            new_distance = calculate_total_distance(cities, new_solution)

            # Choose a random cuckoo to compare
            j = i
            while j == i:
                j = random.randint(0, num_cuckoos - 1)

            # Replace the worse solution with the new solution
            if new_distance < distances[j]:
                cuckoos[j] = new_solution
                distances[j] = new_distance

            # Abandon some solutions with probability pa and replace them with new random solutions
            if random.random() < pa:
                new_solution = generate_random_solution(num_cities)
                new_distance = calculate_total_distance(cities, new_solution)
                idx = distances.index(max(distances))
                cuckoos[idx] = new_solution
                distances[idx] = new_distance

    # Return the best solution found
    best_cuckoo = cuckoos[np.argmin(distances)]
    best_distance = min(distances)
    return best_cuckoo, best_distance

if __name__ == "__main__":
    cities = [
        (0, 0),
        (1, 5),
        (5, 5),
        (6, 3),
        (8, 1),
        (3, 2)
    ]

    best_solution, best_distance = perform_cuckoo_search(cities)
    print("Best solution:", best_solution)
    print("Best distance:", best_distance)
