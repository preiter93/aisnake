from abc import ABC, abstractmethod


class AbstractIndividuum(ABC):
    @classmethod
    @abstractmethod
    def new_random(cls):
        """
        Return an individuum with a random chromosome.

        This is used to initate a random initial population.

        .. note::
        You MUST override this function.
        """
        pass

    @abstractmethod
    def get_fitness(self):
        """
        Evaluate the fitness function.

        :return: int of float

        .. note::
        You MUST override this function.
        """
        pass

    # @staticmethod
    @abstractmethod
    def mutate(self):
        """
        Mutate the individuum.

         .. note::
        You MUST override this function.
        """
        pass

    @staticmethod
    @abstractmethod
    def recombine(parents):
        """
        Recombination of individuums.
        Produces offsprings.

        :parents: List of 2 or more individuums

        :return: List with 2 or more offsprings

        .. note::
        You MUST override this function.
        """
        pass

    # def get_chromosome(self):
    #     """
    #     Returns the chromosome of the Individuum.
    #     """
    #     return self.chromosome


# class AbstractIndividuum(ABC):

#     # Chromosome: List of genes
#     chromosome: list

#     @staticmethod
#     @abstractmethod
#     def get_new_gene():
#         """
#         Returns a randomly picked gene from the gene pool

#         :return: gene

#         .. note::
#         You MUST override this function.
#         """
#         pass

#     @staticmethod
#     @abstractmethod
#     def get_number_of_genes():
#         """
#         Return the number of genes in the chromosome

#         :return: int

#         .. note::
#         You MUST override this function.
#         """
#         pass

#     @staticmethod
#     @abstractmethod
#     def get_fitness():
#         """
#         Evaluate the fitness function.

#         :return: int of float

#         .. note::
#         You MUST override this function.
#         """
#         pass

#     def get_chromosome(self):
#         """
#         Returns the chromosome of a single Individuum.
#         """
#         return self.chromosome

#     @classmethod
#     def new_random(cls):
#         """
#         Return an individuum with a random chromosome.

#         This is used to initate a random initial population.

#         .. note::
#         You CAN override this function.
#         """
#         # obj = cls.__new__(cls)
#         # super(AbstractIndividuum, obj).__init__()
#         chromosome = [cls.get_new_gene() for _ in range(cls.get_number_of_genes())]
#         return cls.new_from_chromosome(chromosome)

#     @classmethod
#     def new_from_chromosome(cls, chromosome):
#         """
#         Return an individuum with a predefined chromosome.

#         This is used to initate offsprings.

#         .. note::
#         You CAN override this function.
#         """
#         obj = cls.__new__(cls)
#         super(AbstractIndividuum, obj).__init__()
#         obj.num_genes = len(chromosome)
#         obj.chromosome = chromosome
#         return obj
