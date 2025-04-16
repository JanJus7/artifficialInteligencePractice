import pygad
import numpy
import time

S = [1, 2, 3, 6, 10, 17, 25, 29, 30, 41, 51, 60, 70, 79, 80]

#definiujemy parametry chromosomu
#geny to liczby: 0 lub 1
gene_space = [0, 1]

#definiujemy funkcję fitness
def fitness_func(ga_instance, solution, solution_idx):
    sum1 = numpy.sum(solution * S)
    solution_invert = 1 - solution
    sum2 = numpy.sum(solution_invert * S)
    # fitness = -numpy.abs(sum1-sum2)
    fitness = 1.0 / (1.0 + numpy.abs(sum1-sum2))
    return fitness

fitness_function = fitness_func

#ile chromsomów w populacji
#ile genow ma chromosom
sol_per_pop = 10
num_genes = len(S)

#ile wylaniamy rodzicow do "rozmanazania" (okolo 50% populacji)
#ile pokolen
#ilu rodzicow zachowac (kilka procent)
num_parents_mating = 5
num_generations = 100
keep_parents = 2

#jaki typ selekcji rodzicow?
#sss = steady, rws=roulette, rank = rankingowa, tournament = turniejowa
parent_selection_type = "sss"

#w il =u punktach robic krzyzowanie?
crossover_type = "single_point"

#mutacja ma dzialac na ilu procent genow?
#trzeba pamietac ile genow ma chromosom
mutation_type = "random"
mutation_percent_genes = 8

successRun = 0
executionTimes = []

for i in range(10):

    start = time.time()
    #inicjacja algorytmu z powyzszymi parametrami wpisanymi w atrybuty
    ga_instance = pygad.GA(gene_space=gene_space,
                        num_generations=num_generations,
                        num_parents_mating=num_parents_mating,
                        fitness_func=fitness_function,
                        sol_per_pop=sol_per_pop,
                        num_genes=num_genes,
                        parent_selection_type=parent_selection_type,
                        keep_parents=keep_parents,
                        crossover_type=crossover_type,
                        mutation_type=mutation_type,
                        mutation_percent_genes=mutation_percent_genes)

    #uruchomienie algorytmu
    ga_instance.run()
    end = time.time()

    duration = end - start

    if ga_instance.best_solution()[1] == 1.0:
        successRun += 1
        executionTimes.append(duration)

    print("Run duration: {duration}".format(duration=duration))

success_rate = (successRun / 10) * 100
avg_time = numpy.mean(executionTimes) if executionTimes else 0

#podsumowanie: najlepsze znalezione rozwiazanie (chromosom+ocena)
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

#tutaj dodatkowo wyswietlamy sume wskazana przez jedynki
prediction = numpy.sum(S*solution)
print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

print(f"\nSkuteczność: {success_rate:.0f}%")
print(f"Średni czas działania (dla trafień): {avg_time:.4f} sec")

#wyswietlenie wykresu: jak zmieniala sie ocena na przestrzeni pokolen
ga_instance.plot_fitness()

# A) Za pomocą 2 list
# B) funkcja powinna znaleźć podział zbioru żeby 2 listy były podobne wartościowo
'''
def fitness_func(solution, solution_idx):
    sum1 = np.sum(solution * S)
    sum2 = np.sum((1 - solution) * S)
    fitness = -abs(sum1 - sum2)
    return fitness
'''
'''
C) Jeśli chodzi o niewielkie problemy (np. 15 elementów):
    Wielkość populacji: 10–20.
    Liczba pokoleń: 30–100.
    Typ selekcji: "sss".
    Typ krzyżowania: "single_point".
    Mutacja:
        typ: "random",
        mutation_percent_genes: 8–10%
'''
'''
D)
Parameters of the best solution : [0. 0. 1. 1. 1. 1. 1. 1. 0. 1. 1. 0. 1. 0. 0.]
Fitness value of the best solution = 1.0
Predicted output based on the best solution : 252.0

wykres rósł od 0 do 2 i od 7 do 8
'''


