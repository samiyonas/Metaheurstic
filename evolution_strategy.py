"""Student file: implement an Evolution Strategy here."""

from experiment import run_experiment
import numpy as np

BENCHMARKS_TO_RUN = ["sphere", "rastrigin", "rosenbrock"]
SEEDS = list(range(100, 120))  # Pilot: [0, 1, 2, 3, 4]. Final: range(100, 120).
MAX_EVALUATIONS = 20_000
PARAMETERS = {
    "mu": 10,
    "lambda_": 50,
    "sigma": 0.1
}
  # Add your chosen hyperparameters here.


def evolution_strategy(
    objective,
    lower_bound,
    upper_bound,
    dimension,
    rng,
    max_evaluations,
    **parameters,
):
    # TODO(student): implement and justify all algorithm design choices.
    evaluations_used = 0

    def evaluate(x):
        nonlocal evaluations_used
        evaluations_used += 1
        return objective(x)

    # 1. Initialize population (P) of size lambda randomly within bounds
    population = rng.uniform(
        lower_bound, upper_bound, size=(parameters.get("lambda_", 50), dimension)
    )
    fitness = np.array([evaluate(ind) for ind in population])

    best_x = population[np.argmin(fitness)]
    best_f = np.min(fitness)

    while evaluations_used < max_evaluations:
        # 2. Select best mu parents from current population P
        best_indices = np.argsort(fitness)[:parameters.get("mu", 10)]
        parents = population[best_indices]

        # 3. Generate lambda children by mutating parents
        children = []
        for i in range(parameters.get("lambda_", 50)):
            # Uniformly select one of the mu parents
            parent = parents[rng.integers(0, parameters.get("mu", 10))]

            # Gaussian mutation with step size sigma
            child = parent + rng.normal(0, parameters.get("sigma", 0.1), size=dimension)

            # Enforce boundary constraints via clipping
            child = np.clip(child, lower_bound, upper_bound)
            children.append(child)

        children = np.array(children)

        # Evaluate children within remaining budget
        children_fitness = []
        for child in children:
            if evaluations_used >= max_evaluations:
                break
            f_val = evaluate(child)
            children_fitness.append(f_val)

            if f_val < best_f:
                best_f = f_val
                best_x = child

        children_fitness = np.array(children_fitness)

        # 4. Update population for next generation
        # (mu + lambda): Combine parents and evaluate children
        population = np.vstack(
            [parents[: len(children_fitness)], children]
        )
        fitness = np.hstack(
            [fitness[best_indices[: len(children_fitness)]], children_fitness]
        )

    return best_x, best_f


def main():
    for benchmark_name in BENCHMARKS_TO_RUN:
        results = run_experiment(
            evolution_strategy,
            benchmark_name,
            list(SEEDS),
            PARAMETERS,
            max_evaluations=MAX_EVALUATIONS,
        )
        print(f"{benchmark_name}: best objective = {results[0].best_value}")


if __name__ == "__main__":
    main()

