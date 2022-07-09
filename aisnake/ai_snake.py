from .evolpy.abstract_individuum import AbstractIndividuum
from .ai.nnet import EvolvingNeuralNet

# from .ai_game import AIGame
from .snake.snake import Snake

# Give snake a variable vision for obstacles
VISION_RADIUS = 2


def perimeter_pixel():
    layers = 0
    for i in range(VISION_RADIUS):
        layers += (1 + 2 * i) * 4 + 4
    return layers


# Vision states + 4 food states (ahead, right, below, left)
INPUT_LAYER = perimeter_pixel() + 4
HIDDEN_LAYER = 24
HIDDEN_LAYER2 = 24
OUTPUT_LAYER = 3

# Hunger determines how many steps snake can go without eating
MAX_HUNGER = 30


class AISnake(Snake, AbstractIndividuum):
    """
    AI Snake has a neural net and knows evolutionary
    methods, such as mutation and recombination
    """

    def __init__(self, neural_net):
        super(AISnake, self).__init__()
        self.neural_net = neural_net
        self.max_hunger = MAX_HUNGER
        self.vision_radius = VISION_RADIUS

    @classmethod
    def new_random(cls):
        obj = cls.__new__(cls)
        super(AISnake, obj).__init__()
        obj.neural_net = EvolvingNeuralNet(
            [INPUT_LAYER, HIDDEN_LAYER, HIDDEN_LAYER2, OUTPUT_LAYER], bias=False
        )
        obj.max_hunger = MAX_HUNGER
        obj.vision_radius = VISION_RADIUS
        return obj

    def get_fitness(self, repetition_runs=10):
        """
        Returns fitness.

        Calculates average of n runs to decrease
        the influence of luck, i.e. good food spawns
        """
        from .ai_game import AIGame

        fitness = 0
        game = AIGame(self)
        for _ in range(repetition_runs):
            game.play_ai(visible=False)
            fitness += game.score
            game.reset()
        return fitness / repetition_runs

    def mutate(self):
        """
        Mutate 5 % of all genes
        """
        # Total number of weights
        total_num_weights = 0
        for i in range(len(self.neural_net.layer_sizes) - 1):
            total_num_weights += (
                self.neural_net.layer_sizes[i] * self.neural_net.layer_sizes[i + 1]
            )
        self.neural_net.mutate(int(total_num_weights * 0.05))

    @staticmethod
    def recombine(parents):
        offsprings = parents[0].neural_net.recombine(parents[1].neural_net)
        return [AISnake(neural_net) for neural_net in offsprings]
