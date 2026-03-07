"""Integration test: verify GA maintains diversity across generations."""

from vibe_quant.discovery.diversity import population_entropy
from vibe_quant.discovery.operators import (
    _random_chromosome,
    apply_elitism,
    crossover,
    is_valid_chromosome,
    mutate,
)


class TestDiversityPreservation:
    """Verify that entropy doesn't collapse to zero over multiple generations."""

    def test_entropy_maintained_with_interventions(self) -> None:
        """After 10 generations with mock fitness, entropy stays above 0.15."""
        pop = [_random_chromosome() for _ in range(20)]
        initial_entropy = population_entropy(pop)
        assert initial_entropy > 0.3  # Random pop should be diverse

        import random as rng

        for gen in range(10):
            scores = [rng.random() for _ in pop]

            # Evolution: simple tournament + crossover + mutation
            new_pop = apply_elitism(pop, scores, 1)
            indices = list(range(len(pop)))
            rng.shuffle(indices)
            while len(new_pop) < len(pop):
                i = rng.choice(indices)
                j = rng.choice(indices)
                if rng.random() < 0.8:
                    a, b = crossover(pop[i], pop[j])
                else:
                    a, b = pop[i].clone(), pop[j].clone()
                a = mutate(a, 0.15)
                b = mutate(b, 0.15)
                for c in (a, b):
                    if len(new_pop) < len(pop) and is_valid_chromosome(c):
                        new_pop.append(c)

            # Diversity intervention
            from vibe_quant.discovery.diversity import (
                inject_random_immigrants,
                should_inject_immigrants,
            )

            entropy = population_entropy(new_pop)
            if should_inject_immigrants(entropy):
                new_pop = inject_random_immigrants(new_pop, scores, fraction=0.1)

            pop = new_pop

        final_entropy = population_entropy(pop)
        assert final_entropy > 0.15, (
            f"Entropy collapsed: initial={initial_entropy:.3f}, final={final_entropy:.3f}"
        )
