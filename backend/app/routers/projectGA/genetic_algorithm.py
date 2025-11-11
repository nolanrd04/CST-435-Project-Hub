"""Core Genetic Algorithm implementation."""

import random
from typing import List, Optional, Tuple
from .individual import Individual


class GAConfig:
    """Configuration for the genetic algorithm."""

    def __init__(
        self,
        population_size: int = 200,
        mutation_rate: float = 0.01,
        crossover_rate: float = 0.8,
        elitism_count: int = 2,
        tournament_size: int = 5,
        charset: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
    ):
        """
        Initialize GA configuration.
        
        Args:
            population_size: Number of individuals in population
            mutation_rate: Probability of mutation (0-1)
            crossover_rate: Probability of crossover (0-1)
            elitism_count: Number of best individuals to preserve
            tournament_size: Number of individuals in tournament selection
            charset: Available characters for genes
        """
        self.population_size = max(2, population_size)
        self.mutation_rate = max(0.0, min(1.0, mutation_rate))
        self.crossover_rate = max(0.0, min(1.0, crossover_rate))
        self.elitism_count = max(0, min(population_size // 2, elitism_count))
        self.tournament_size = max(2, min(population_size, tournament_size))
        self.charset = charset

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "population_size": self.population_size,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "elitism_count": self.elitism_count,
            "tournament_size": self.tournament_size,
            "charset": self.charset
        }


class GeneticAlgorithm:
    """Genetic algorithm implementation for text evolution."""

    def __init__(self, config: GAConfig):
        """
        Initialize the genetic algorithm.
        
        Args:
            config: GAConfig object with algorithm parameters
        """
        self.config = config
        self.population: List[Individual] = []
        self.generation = 0
        self.target = ""
        self.best_ever: Optional[Individual] = None
        self.generation_history: List[dict] = []

    def initialize(self, target: str) -> None:
        """
        Initialize population with random individuals.
        
        Args:
            target: The target string to evolve towards
        """
        self.target = target
        self.generation = 0
        self.best_ever = None
        self.generation_history = []
        self.population = []

        # Create initial random population
        for _ in range(self.config.population_size):
            individual = Individual.create_random(
                len(target), 
                self.config.charset
            )
            individual.calculate_fitness(target)
            self.population.append(individual)

        # Track best
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        self.best_ever = self.population[0].copy()

    def evolve(self) -> bool:
        """
        Execute one generation of evolution.
        
        Returns:
            True if target is found (perfect fitness achieved), False otherwise
        """
        if not self.population:
            raise ValueError("Population not initialized. Call initialize() first.")

        # Calculate fitness for all individuals
        for individual in self.population:
            individual.calculate_fitness(self.target)

        # Sort by fitness (descending)
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        # Update best ever
        if self.population[0].fitness > self.best_ever.fitness:
            self.best_ever = self.population[0].copy()

        # Check for completion
        if self.best_ever.fitness >= 100.0:
            self.generation += 1
            self._record_generation_stats()
            return True

        # Create new population
        new_population: List[Individual] = []

        # Apply elitism - keep best individuals
        for i in range(self.config.elitism_count):
            if i < len(self.population):
                new_population.append(self.population[i].copy())

        # Fill rest of population through selection, crossover, mutation
        while len(new_population) < self.config.population_size:
            # Select two parents
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()

            # Crossover
            if random.random() < self.config.crossover_rate:
                child = parent1.crossover(parent2)
            else:
                child = parent1.copy()

            # Mutation
            child = child.mutate(self.config.mutation_rate, self.config.charset, self.target)

            new_population.append(child)

        self.population = new_population[:self.config.population_size]
        self.generation += 1

        # Record statistics
        self._record_generation_stats()

        return False

    def _tournament_selection(self) -> Individual:
        """
        Select an individual using tournament selection.
        
        Returns:
            The fittest individual from the tournament
        """
        tournament = random.sample(
            self.population, 
            min(self.config.tournament_size, len(self.population))
        )
        return max(tournament, key=lambda x: x.fitness).copy()

    def _record_generation_stats(self) -> None:
        """Record statistics for current generation."""
        if not self.population:
            return

        fitnesses = [ind.fitness for ind in self.population]
        avg_fitness = sum(fitnesses) / len(fitnesses)
        max_fitness = max(fitnesses)
        min_fitness = min(fitnesses)

        self.generation_history.append({
            "generation": self.generation,
            "best_fitness": max_fitness,
            "average_fitness": avg_fitness,
            "worst_fitness": min_fitness,
            "best_individual": self.best_ever.genes if self.best_ever else ""
        })

    def get_status(self) -> dict:
        """
        Get current algorithm status.
        
        Returns:
            Dictionary with current state and statistics
        """
        if not self.population:
            return {
                "initialized": False,
                "generation": 0,
                "population_size": 0,
                "target": "",
                "best_fitness": 0,
                "best_individual": "",
                "average_fitness": 0,
                "is_complete": False
            }

        fitnesses = [ind.fitness for ind in self.population]
        avg_fitness = sum(fitnesses) / len(fitnesses)

        return {
            "initialized": True,
            "generation": self.generation,
            "population_size": len(self.population),
            "target": self.target,
            "best_fitness": round(self.best_ever.fitness if self.best_ever else 0, 2),
            "best_individual": self.best_ever.genes if self.best_ever else "",
            "average_fitness": round(avg_fitness, 2),
            "is_complete": (self.best_ever and self.best_ever.fitness >= 100.0) if self.best_ever else False,
            "config": self.config.to_dict()
        }

    def get_top_population(self, count: int = 20) -> List[dict]:
        """
        Get the top N individuals from current population.
        
        Args:
            count: Number of top individuals to return
            
        Returns:
            List of dictionaries representing top individuals
        """
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        return [ind.to_dict() for ind in sorted_pop[:count]]

    def get_history(self) -> List[dict]:
        """Get generation history."""
        return self.generation_history

    def reset(self) -> None:
        """Reset the algorithm to initial state."""
        self.population = []
        self.generation = 0
        self.target = ""
        self.best_ever = None
        self.generation_history = []
