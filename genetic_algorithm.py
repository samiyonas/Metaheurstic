"""Student file: implement a Genetic Algorithm here."""

from experiment import run_experiment
import numpy as np

BENCHMARKS_TO_RUN = ["sphere", "rastrigin", "rosenbrock"]
SEEDS = list(range(100, 120))  # Pilot: [0, 1, 2, 3, 4]. Final: range(100, 120).
MAX_EVALUATIONS = 20_000
PARAMETERS = {
    "pop_size": 50,
    "crossover_prob": 0.9,
    "mutation_prob": 0.1,
    "eta_c": 15.0,
    "eta_m": 20.0,
    "tournament_size": 3
}  # Add your chosen hyperparameters here.


def genetic_algorithm(
    objective,
    lower_bound,
    upper_bound,
    dimension,
    rng,
    max_evaluations,
    **parameters,
):
    # TODO(student): implement and justify all algorithm design choices.
    pop_size = parameters.get("pop_size", 50)
    crossover_prob = parameters.get("crossover_prob", 0.9)
    mutation_prob = parameters.get("mutation_prob", 0.1)
    eta_c = parameters.get("eta_c", 15.0)
    eta_m = parameters.get("eta_m", 20.0)
    tournament_size = parameters.get("tournament_size", 3)

    evaluations = 0

    def evaluate(individual):
        nonlocal evaluations
        if evaluations >= max_evaluations:
            return float("inf")
        evaluations += 1
        # Ensure domain compliance strictly before objective evaluation
        clipped_ind = np.clip(individual, lower_bound, upper_bound)
        return objective(clipped_ind)

    # P <- initialize(N)
    population = rng.uniform(
        lower_bound, upper_bound, size=(pop_size, dimension)
    )
    population = np.clip(population, lower_bound, upper_bound)

    # Initial assessment
    fitness = np.array([evaluate(ind) for ind in population])

    best_idx = np.argmin(fitness)
    best_individual = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    def select_parent():
        indices = rng.choice(pop_size, size=tournament_size, replace=False)
        best_in_tournament = indices[np.argmin(fitness[indices])]
        return population[best_in_tournament]

    def sbx_crossover(p1, p2):
        if rng.random() > crossover_prob:
            return p1.copy(), p2.copy()

        c1 = np.empty_like(p1)
        c2 = np.empty_like(p2)

        for i in range(dimension):
            if rng.random() <= 0.5:
                u = rng.random()
                if u <= 0.5:
                    beta = (2.0 * u) ** (1.0 / (eta_c + 1.0))
                else:
                    beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0))

                c1[i] = 0.5 * ((1.0 + beta) * p1[i] + (1.0 - beta) * p2[i])
                c2[i] = 0.5 * ((1.0 - beta) * p1[i] + (1.0 + beta) * p2[i])
            else:
                c1[i] = p1[i]
                c2[i] = p2[i]

        return np.clip(c1, lower_bound, upper_bound), np.clip(
            c2, lower_bound, upper_bound
        )

    def polynomial_mutate(child):
        mutated = child.copy()
        for i in range(dimension):
            if rng.random() < mutation_prob:
                u = rng.random()
                if u < 0.5:
                    delta = (2.0 * u) ** (1.0 / (eta_m + 1.0)) - 1.0
                else:
                    delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta_m + 1.0))

                mutated[i] += delta * (upper_bound - lower_bound)
        return np.clip(mutated, lower_bound, upper_bound)

    while evaluations < max_evaluations:
        children = []

        while len(children) < pop_size:
            p1 = select_parent()
            p2 = select_parent()

            c1, c2 = sbx_crossover(p1, p2)

            c1 = polynomial_mutate(c1)
            c2 = polynomial_mutate(c2)

            children.append(c1)
            if len(children) < pop_size:
                children.append(c2)

        C = np.array(children)

        # assess(C)
        c_fitness = []
        for ind in C:
            if evaluations >= max_evaluations:
                break
            c_fitness.append(evaluate(ind))

        c_fitness = np.array(c_fitness)
        if len(c_fitness) == 0:
            break

        current_best_idx = np.argmin(c_fitness)
        if c_fitness[current_best_idx] < best_fitness:
            best_fitness = c_fitness[current_best_idx]
            best_individual = C[current_best_idx].copy()

        # Generational replacement with elitism
        if len(c_fitness) == pop_size:
            worst_c_idx = np.argmax(c_fitness)
            C[worst_c_idx] = best_individual.copy()
            c_fitness[worst_c_idx] = best_fitness

        population = C
        fitness = c_fitness

    return best_individual


def main():
    for benchmark_name in BENCHMARKS_TO_RUN:
        results = run_experiment(
            genetic_algorithm,
            benchmark_name,
            list(SEEDS),
            PARAMETERS,
            max_evaluations=MAX_EVALUATIONS,
        )
        print(f"{benchmark_name}: best objective = {results[0].best_value}")


if __name__ == "__main__":
    main()

