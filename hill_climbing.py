"""Student file: implement Random-Restart Hill Climbing here."""

from experiment import run_experiment
import numpy as np

# For a quick test, choose one benchmark and one seed, then run this file.
BENCHMARKS_TO_RUN = ["sphere", "rastrigin", "rosenbrock"]
SEEDS = list(range(100, 120))  # Pilot: [0, 1, 2, 3, 4]. Final: range(100, 120).
MAX_EVALUATIONS = 20_000

def gaussian(curr_state, dimension, lower_bound, upper_bound, rng, parameters):
    noise = rng.normal(loc=0.0, scale=parameters.get("step_size", 0.1), size=dimension)
    candidate = curr_state + noise
    return np.clip(candidate, lower_bound, upper_bound)

def bounded_uniform_step(curr_state, dimension, lower_bound, upper_bound, rng, parameters):
    step_size = parameters.get("step_size", 0.1)
    candidate = curr_state + rng.uniform(-step_size, step_size, size=dimension)
    return np.clip(candidate, lower_bound, upper_bound)


# Add your chosen hyperparameters here.
PARAMETERS = {
    "restarts": 20,
    "neighbour_fn": gaussian,
    "num_neighbours": 1,
    "step_size": 0.1
}


def hill_climb(objective, neighbour_fn, start, run_budget, dimension,
               lower_bound, upper_bound, rng, parameters):
    curr_state = start
    curr_score = objective(start)
    evals = 1

    while evals < run_budget:
        num_n = parameters.get("num_neighbours", 1)

        if evals + num_n > run_budget:
            break

        best_neigh_state = None
        best_neigh_score = float("inf")

        for _ in range(num_n):
            neigh_state = neighbour_fn(curr_state, dimension, lower_bound, upper_bound, rng, parameters)
            neigh_score = objective(neigh_state)
            evals += 1

            if neigh_score < best_neigh_score:
                best_neigh_state = neigh_state
                best_neigh_score = neigh_score

        if best_neigh_score >= curr_score:
            break

        curr_state, curr_score = best_neigh_state, best_neigh_score

    return curr_state, curr_score, evals


def hill_climbing(
    objective,
    lower_bound,
    upper_bound,
    dimension,
    rng,
    max_evaluations,
    **parameters,
):
    # TODO(student): implement and justify all algorithm design choices.
    best_state = None
    best_score = float("inf")

    restarts = parameters.get("restarts", 20)
    neighbour_fn = parameters.get("neighbour_fn", gaussian)

    budget_per_restart = max_evaluations // restarts
    evals_remaining = max_evaluations

    while evals_remaining > 0:
        run_budget = min(budget_per_restart, evals_remaining)
        start = rng.uniform(low=lower_bound, high=upper_bound, size=dimension)
        state, score, evals_used = hill_climb(objective, neighbour_fn, start, run_budget, dimension, lower_bound, upper_bound, rng, parameters)

        evals_remaining -= evals_used

        if score < best_score:
            best_state, best_score = state, score

    return best_state, best_score
        



def main():
    for benchmark_name in BENCHMARKS_TO_RUN:
        results = run_experiment(
            hill_climbing,
            benchmark_name,
            list(SEEDS),
            PARAMETERS,
            max_evaluations=MAX_EVALUATIONS,
        )
        print(f"{benchmark_name}: best objective = {results[0].best_value}")


if __name__ == "__main__":
    main()

