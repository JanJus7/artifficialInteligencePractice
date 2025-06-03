import matplotlib.pyplot as plt
import random

from aco import AntColony


plt.style.use("dark_background")


COORDS = (
    (20, 52),
    (43, 50),
    (20, 84),
    (70, 65),
    (29, 90),
    (87, 83),
    (73, 23),
    (50, 20),
    (90, 30),
    (60, 10),
    (10, 10),
    (30, 40),
    (80, 50),
    (40, 70),
    (70, 90),
    (90, 70),
    (20, 30),
    (50, 50),
    (60, 80),
    (80, 20),
)


def random_coord():
    r = random.randint(0, len(COORDS))
    return r


def plot_nodes(w=12, h=8):
    for x, y in COORDS:
        plt.plot(x, y, "g.", markersize=15)
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches([w, h])


def plot_all_edges():
    paths = ((a, b) for a in COORDS for b in COORDS)

    for a, b in paths:
        plt.plot((a[0], b[0]), (a[1], b[1]))


plot_nodes()

# colony = AntColony(COORDS, ant_count=300, iterations=300)
colony = AntColony(
    COORDS,
    ant_count=300,
    alpha=0.5,
    beta=1.2,
    pheromone_evaporation_rate=0.40,
    pheromone_constant=1000.0,
    iterations=1,
)


optimal_nodes = colony.get_path()

for i in range(len(optimal_nodes) - 1):
    plt.plot(
        (optimal_nodes[i][0], optimal_nodes[i + 1][0]),
        (optimal_nodes[i][1], optimal_nodes[i + 1][1]),
    )


plt.show()

# WNIOSEK:
# Po eksperymentach można zauważyć:
# - Zwiększenie liczby mrówek przyspiesza znalezienie dobrej trasy, ale zwiększa czas obliczeń.
# - Wysokie alpha (np. 1.0) powoduje, że mrówki mocniej kierują się feromonem.
# - Wysokie beta sprawia, że bardziej liczy się odległość.
# - Niskie parowanie feromonu (evaporation_rate < 0.2) powoduje, że feromony zostają długo i mogą „utknąć” w lokalnym minimum.
# - Mnijesza ilość iteracji daje wynik ale przy większej ilości punktów jest on niedokładny, za duża ilość też nie jest dobra

