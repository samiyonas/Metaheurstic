"""Student file: implement Simulated Annealing here."""

from experiment import run_experiment
import numpy as np
import math

BENCHMARKS_TO_RUN = ["sphere", "rastrigin", "rosenbrock"]
SEEDS = list(range(100, 120))  # Pilot: [0, 1, 2, 3, 4]. Final: range(100, 120).
MAX_EVALUATIONS = 20_000
PARAMETERS = {
    "initial_temp": 100.0,
    "final_temp": 1e-8,
    "step_size": 0.05,
    "cooling_schedule": "exponential",
}  # Add your chosen hyperparameters here.


def simulated_annealing(
    objective,
    lower_bound,
    upper_bound,
    dimension,
    rng,
    max_evaluations,
    **parameters,
):
    # TODO(student): implement and justify all algorithm design choices.
    """Simulated Annealing algorithm for continuous global optimization."""

    initial_temp = parameters.get("initial_temp", 100.0)
    final_temp = parameters.get("final_temp", 1e-8)
    step_size_scale = parameters.get("step_size", 0.05)
    cooling_schedule = parameters.get("cooling_schedule", "exponential")

    bound_range = upper_bound - lower_bound
    std_dev = step_size_scale * bound_range

    current_x = rng.uniform(lower_bound, upper_bound, size=dimension)
    current_eval = objective(current_x)
    evaluations = 1

    best_x = np.copy(current_x)
    best_eval = current_eval

    # 3. Optimization Loop
    while evaluations < max_evaluations:
        # Calculate current temperature T(t) based on budget fraction
        progress = evaluations / max_evaluations

        if cooling_schedule == "exponential":
            # T(t) = T_init * (T_final / T_init) ^ progress
            temp = initial_temp * ((final_temp / initial_temp) ** progress)
        elif cooling_schedule == "linear":
            # T(t) = T_init - progress * (T_init - T_final)
            temp = max(final_temp, initial_temp - progress * (initial_temp - final_temp))
        else:
            temp = initial_temp / (1.0 + progress * 100.0)

        # Generate neighbor via Gaussian perturbation centered at current state
        perturbation = rng.normal(0, std_dev, size=dimension)
        candidate_x = current_x + perturbation

        # Clip candidate to domain bounds
        candidate_x = np.clip(candidate_x, lower_bound, upper_bound)

        # Evaluate candidate solution
        candidate_eval = objective(candidate_x)
        evaluations += 1

        # Calculate energy difference (minimization task)
        delta_e = candidate_eval - current_eval

        # Acceptance criterion: Metropolis-Hastings
        if delta_e < 0:
            accept = True
        else:
            # Prevent overflow or division by zero with small temperatures
            if temp <= 1e-12:
                accept = False
            else:
                prob = math.exp(-delta_e / temp)
                accept = rng.uniform(0.0, 1.0) < prob

        if accept:
            current_x = candidate_x
            current_eval = candidate_eval

            # Maintain historical best solution
            if current_eval < best_eval:
                best_x = np.copy(current_x)
                best_eval = current_eval

    return best_x, best_eval


def main():
    for benchmark_name in BENCHMARKS_TO_RUN:
        results = run_experiment(
            simulated_annealing,
            benchmark_name,
            list(SEEDS),
            PARAMETERS,
            max_evaluations=MAX_EVALUATIONS,
        )
        print(f"{benchmark_name}: best objective = {results[0].best_value}")


if __name__ == "__main__":
    main()

