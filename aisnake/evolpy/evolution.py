import random
from copy import deepcopy
from .abstract_individuum import AbstractIndividuum
import numpy as np


class Evolution(object):
    """
    Playing field of Evolution
    """

    def __init__(self, obj_individuum):
        if isinstance(obj_individuum, AbstractIndividuum):
            raise ValueError("Expect input of instance AbstractIndividuum")
        self.individuum = obj_individuum
        print("Input ok.")

    def optimize(
        self,
        population_size=200,
        max_generations=1000,
        crossover_rate=0.8,
        mutation_rate=0.1,
        preservation_rate=0.0,
        max_fitness=None,
        callback=None,
        offsprings_per_recombination=2,
        pop=None,
        add_individuum=None,
    ):
        """
        Skeletons of operations to perform the evolutionary algorithm.

        .. note::
        DO NOT override this function.
        """

        # Define the mix of old and new individuums in each new generation
        new_individuums = int((1 - preservation_rate) * population_size)
        new_individuums -= new_individuums % offsprings_per_recombination
        old_individuums = population_size - new_individuums

        self.display(
            {
                "Total population": population_size,
                "From new generation": new_individuums,
                "From old generation": old_individuums,
                "mutation_rate": mutation_rate,
                "crossover_rate": crossover_rate,
                "preservation_rate": preservation_rate,
            }
        )

        # Initiate population
        if pop is None:
            pop = [self.individuum.new_random() for _ in range(population_size)]
        if add_individuum is not None:
            pop[0] = add_individuum
        fit = []
        for gen in range(max_generations + 1):
            # Evaluate fitness
            fit = [ind.get_fitness() for ind in pop]

            # Callback
            if callback is None:
                self.default_callback(pop, fit, gen)
            else:
                callback(pop, fit, gen)

            # Terminate
            if self.terminate_max_fitness(fit, max_fitness):
                print("An individuum reached its maximum fitness!")
                return pop

            # Create next generation
            pop_child = []

            for i in range(new_individuums // offsprings_per_recombination):

                # Select parents
                parents = self.selection_of_parents(pop, fit)

                # Recombination
                if random.random() < crossover_rate:
                    # Apply crossover function
                    childs = self.recombine(parents)
                else:
                    # Just take parents over to new generation
                    childs = [deepcopy(p) for p in parents]

                # Mutation
                for child in childs:
                    if random.random() < mutation_rate:
                        self.mutate(child)

                    # Store in new population
                    pop_child.append(child)

            # Take some parents over to the next population
            if old_individuums > 0:
                # Find indices of N maximum values
                argmax = list(np.argpartition(fit, -old_individuums)[-old_individuums:])
                for i in argmax:
                    pop_child.append(pop[i])

            # TODO: Check more properly
            assert (
                len(pop_child) == population_size
            ), "Hm, size of population has changed."

            # replace population by children
            pop = pop_child

        print("Reached generations limit, no perfekt individuum found.")
        return pop

    def default_callback(self, pop, fit, gen):
        """
        Callback function.

        :pop: list of population
        :fit: list of populations fitness
        :gen: int, number of generation
        """
        max_fit = max(fit)
        print("Max Fit:", max_fit, " in Gen:", gen)

    def terminate_max_fitness(self, fit, max_fitness):
        """
        Terminates if an individuum has reached the
        maximum fitness.
        The maximum fitness must be supplied to optimize()
        """
        if max_fitness is None:
            return False
        return max(fit) >= max_fitness

    def selection_of_parents(self, population, fitness):
        """
        Selection of n (default = 2) individuals who take part in mating based on
        their fitness.

        There are many ways to select an individual for reproduction.
        The standard implementation is based on the roulette algorithm,
        which selects two individual with a certain probability based on
        its fitness value.

        .. note::
        You CAN override this function.
        """
        return [
            select_one_roulette(population, fitness),
            select_one_roulette(population, fitness),
        ]

    def recombine(self, parents):
        """
        Recombination of parents produces offsprings.

        .. note::
        You CAN override this function.
        """
        return self.individuum.recombine(parents)

    def mutate(self, individuum):
        """
        Mutate an individuum.

        .. note::
        You CAN override this function.
        """
        individuum.mutate()

    @staticmethod
    def display(dict):
        # width_a = 20
        # width_b = 10
        print("-" * 30)
        for k, v in dict.items():
            if type(v) == int:
                print("{:20} : {:5d}".format(k, v))
            if type(v) == float:
                print("{:20} : {:5.2f}".format(k, v))
        print("-" * 30)


def select_one_roulette(population, fitness):
    """
    Individuum with higher fitness is selected more likely.
    """
    if len(population) != len(fitness):
        raise ValueError("Size mismatch.")
    total_fitness = sum(fitness)
    pick = random.uniform(0, total_fitness)
    current = 0
    for fit, ind in zip(fitness, population):
        current += fit
        if current > pick:
            return ind
