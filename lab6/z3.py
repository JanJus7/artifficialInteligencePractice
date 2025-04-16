import pygad
import time
import numpy

maze = [
    [0, 0, 1, 1, 1, 0, 0, 0, 1, 0],
    [1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1, 1, 1, 0, 1, 0],
    [1, 1, 1, 1, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 1, 1, 1],
    [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
]

start = (0, 0)
end = (9, 9)

moves = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

chromosome_length = 30
gene_space = [0, 1, 2, 3]

def fitness_func(instance, solution, solution_idx):
    x, y = start
    visited = set()
    steps = 0
    for move in solution:
        dx, dy = moves[move]
        new_x, new_y = x + dx, y + dy

        if 0 <= new_x < 10 and 0 <= new_y < 10 and maze[new_x][new_y] == 0:
            x, y = new_x, new_y
            visited.add((x, y))
            steps += 1
        else:
            break

        if (x, y) == end:
            return 1000 - steps

    dist = abs(x - end[0]) + abs(y - end[1])
    return 1 / (dist + 1)

execution_times = []

for i in range(10):

    start_time = time.time()
    ga_instance = pygad.GA(
        num_generations=100,
        num_parents_mating=10,
        fitness_func=fitness_func,
        sol_per_pop=20,
        num_genes=chromosome_length,
        gene_space=gene_space,
        parent_selection_type="sss",
        keep_parents=2,
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=10
    )

    ga_instance.run()
    end_time = time.time()
    execution_times.append(end_time - start_time)

avg_time = numpy.mean(execution_times) if execution_times else 0

solution, solution_fitness, _ = ga_instance.best_solution()
print(f"Najlepsze rozwiązanie: {solution}")
print(f"Fitness: {solution_fitness}")
print(f"Średni czas wykonania: {avg_time:.4f} sekund")



