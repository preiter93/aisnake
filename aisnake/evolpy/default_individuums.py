from .abstract_individuum import AbstractIndividuum
from abc import abstractmethod
import random


class IndividuumWithListChromosome(AbstractIndividuum):
    """
    Inherit from this individuum if you chromosome is
    a simple list.

    This applies to most use cases
    """

    chromosome: list

    @staticmethod
    @abstractmethod
    def pick_random_gene():
        """
        Returns a randomly picked gene from the gene pool

        :return: gene

        .. note::
        You MUST override this function.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_number_of_genes():
        """
        Return the number of genes in the chromosome

        :return: int

        .. note::
        You MUST override this function.
        """
        pass

    @classmethod
    def new_from_chromosome(cls, chromosome):
        """
        Return an individuum with a predefined chromosome.

        This is used to initate offsprings.

        .. note::
        You CAN override this function.
        """
        obj = cls.__new__(cls)
        super(IndividuumWithListChromosome, obj).__init__()
        obj.num_genes = len(chromosome)
        obj.chromosome = chromosome
        return obj

    @classmethod
    def new_random(cls):
        """
        Return an individuum with a random chromosome.

        This is used to initate a random initial population.

        .. note::
        You CAN override this function.
        """
        chromosome = [cls.pick_random_gene() for _ in range(cls.get_number_of_genes())]
        return cls.new_from_chromosome(chromosome)

    @staticmethod
    def recombine(parents):
        """
        Recombination of parents produces n (default = 2) offsprings.

        Again there are many ways to recombine individuals. The default
        applies simple recombination with one splitting point of
        the chromosomes.

        .. note::
        You CAN override this function.
        """
        if len(parents) != 2:
            raise ValueError(
                "Default implementation of recombination expects 2 parents."
            )
        # Get selected parents
        p1, p2 = parents[0], parents[1]
        # Extract their chromosomes
        ch1 = p1.get_chromosome()
        ch2 = p2.get_chromosome()
        # Recombine
        chs = single_crossover(ch1, ch2)
        return [p1.new_from_chromosome(ch) for ch in chs]

    def mutate(self):
        """
        Mutate an individuum.

        Default:
        Replaces a gene at a random position with a new random gene.

        .. note::
        You CAN override this function.
        """
        chromosome = self.get_chromosome()
        num_genes = len(chromosome)
        pos = random.randint(0, num_genes - 1)
        chromosome[pos] = self.pick_random_gene()


def single_crossover(p1, p2):
    """
    Recombine chromosome with another chromosome of the same type.
    (i) Picks a random position within the chromosome and
    (ii) produces two new offsprings from a crossover combination.
    .. note::
                    Homologous with 2 parents and 1 split point
    """
    num_genes = len(p1)
    if num_genes < 2:
        raise ValueError("Number of genes should be larger than two.")
    pt = random.randint(1, num_genes - 2)
    # perform crossover
    c1 = p1[:pt] + p2[pt:]
    c2 = p2[:pt] + p1[pt:]
    return [c1, c2]
