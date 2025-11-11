"""Individual class for Genetic Algorithm - represents a chromosome/solution."""

import random
from typing import Optional


class Individual:
    """Represents a single individual (chromosome) in the population."""

    def __init__(self, genes: str, fitness: float = 0.0):
        """
        Initialize an Individual.
        
        Args:
            genes: The chromosome (string of characters)
            fitness: The fitness score (0-100 percentage)
        """
        self.genes = genes
        self.fitness = fitness

    @staticmethod
    def create_random(length: int, charset: str) -> "Individual":
        """
        Create a random individual with random genes.
        
        Args:
            length: Length of the chromosome
            charset: Available characters to choose from
            
        Returns:
            A new Individual with random genes
        """
        genes = "".join(random.choice(charset) for _ in range(length))
        return Individual(genes)

    def calculate_fitness(self, target: str) -> None:
        """
        Calculate fitness as percentage of matching characters with target.
        
        Args:
            target: The target string to match
        """
        if not target:
            self.fitness = 0.0
            return

        matches = sum(1 for i, char in enumerate(self.genes) 
                     if i < len(target) and char == target[i])
        self.fitness = (matches / len(target)) * 100.0

    def mutate(self, mutation_rate: float, charset: str, target: str = "") -> "Individual":
        """
        Create a new individual with smart mutations.
        
        If target is provided, correct characters are protected and incorrect ones
        are biased toward the target character.
        
        Args:
            mutation_rate: Probability (0-1) of each character mutating
            charset: Available characters for mutation
            target: Optional target string for smart mutations
            
        Returns:
            A new mutated Individual
        """
        mutated_genes = ""
        
        # Smart mutation: aware of target string
        if target:
            for i, char in enumerate(self.genes):
                if i < len(target):
                    target_char = target[i]
                    
                    # If character is CORRECT, rarely mutate it (10% of normal rate)
                    if char == target_char:
                        if random.random() < mutation_rate * 0.1:
                            mutated_genes += random.choice(charset)
                        else:
                            mutated_genes += char  # Keep correct character
                    
                    # If character is WRONG, try to fix it
                    else:
                        if random.random() < mutation_rate:
                            # 80% chance to mutate to target, 20% chance random
                            if random.random() < 0.8:
                                mutated_genes += target_char
                            else:
                                mutated_genes += random.choice(charset)
                        else:
                            mutated_genes += char
                else:
                    mutated_genes += char
        
        # Fallback: regular mutation if no target provided
        else:
            for char in self.genes:
                if random.random() < mutation_rate:
                    mutated_genes += random.choice(charset)
                else:
                    mutated_genes += char
        
        return Individual(mutated_genes)

    def crossover(self, other: "Individual") -> "Individual":
        """
        Single-point crossover with another individual.
        
        Args:
            other: The other parent individual
            
        Returns:
            A new Individual created from crossover
        """
        if not self.genes or not other.genes:
            return Individual(self.genes)

        crossover_point = random.randint(1, len(self.genes) - 1)
        new_genes = self.genes[:crossover_point] + other.genes[crossover_point:]
        return Individual(new_genes)

    def copy(self) -> "Individual":
        """Create a copy of this individual."""
        return Individual(self.genes, self.fitness)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "genes": self.genes,
            "fitness": round(self.fitness, 2)
        }

    def __repr__(self) -> str:
        return f"Individual(genes='{self.genes}', fitness={self.fitness:.2f})"
