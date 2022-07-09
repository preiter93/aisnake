from .evolpy.abstract_individuum import AbstractIndividuum
from .ai.nnet import EvolvingNeuralNet
from .snake.snake import Snake, transform_bord_to_snake
from .snake.game import Game
from .snake.camera import Camera
from copy import copy
from .snake.datatypes import Turn, perimeter
import pygame

VISION_RADIUS = 2


def perimeter_pixel():
    layers = 0
    for i in range(VISION_RADIUS):
        layers += (1 + 2 * i) * 4 + 4
    return layers


# Vision states + 4 food states
INPUT_LAYER = perimeter_pixel() + 4
HIDDEN_LAYER = 24
HIDDEN_LAYER2 = 24
OUTPUT_LAYER = 3

MAX_HUNGER = 20


class AIGame(Game):
    """
    Let game play itself without GUI
    """

    def __init__(self):
        super().__init__()
        self.camera = Camera(pixel_offsets=[3, 3, 3, 3])

    def play_ai(self, model, visible=False):
        """
        Play ai invisible in the background
        """
        if visible:
            import pygame
            import time

            pygame.init()
            self.display = pygame.display.set_mode(
                (self.camera.total_width(), self.camera.total_height())
            )
            pygame.display.set_caption("Game of Snake.")
            self.render()
            time.sleep(1)
            clock = pygame.time.Clock()

        game_over = False
        # TODO: Check copy
        snake_size = copy(self.snake.size())
        # Hunger adds penalty for non-eating snakes
        hunger = MAX_HUNGER
        while not game_over:
            # Action from ai
            turn = self.ai_action(model)
            # Execute one timestep
            self.step(turn)
            if visible:
                # Render
                self.render()

            # Snake has eaten
            if self.snake.size() > snake_size:
                hunger, snake_size = MAX_HUNGER, self.snake.size()
            else:
                hunger -= 1
            # Check collision and hunger
            game_over = any([self.collision(), hunger < 0])

            if visible:
                clock.tick(10)  # FPS
        if visible:
            pygame.quit()

    def get_state(self):
        """
        Return state of the game that is fed into the AI model.

        """
        state = []

        for i in range(VISION_RADIUS):
            for pos in perimeter(self.snake.head(), self.snake.direction, i + 1):
                state.append(self.collision_at(pos))

        # Food state [ahead, right, behind, left]
        dist = transform_bord_to_snake(self.food.pos, self.snake)
        state.append(dist.y > 0)
        state.append(dist.x > 0)
        state.append(dist.y < 0)
        state.append(dist.x < 0)

        return state

    def ai_action(self, model):
        """
        Get turn from ai model
        """
        import torch

        state = self.get_state()
        state = torch.tensor(state, dtype=torch.float)
        x = model.forward(state)
        values = list(x.detach().numpy())
        maxindex = values.index(max(values))
        if maxindex == 0:
            return Turn.NO
        elif maxindex == 1:
            return Turn.RIGHT
        elif maxindex == 2:
            return Turn.LEFT
        else:
            raise ValueError("maxindex outside of limits")

    def render(self):
        import pygame
        from .snake.game import render_field, render_snake, render_food, render_score
        from .snake.params import COL_BORDER

        self.display.fill(COL_BORDER)
        render_field(self.display, self.camera)
        render_snake(self.snake, self.display, self.camera)
        self.render_state()
        render_food(self.food, self.display, self.camera)
        render_score(self.score(), self.display, self.camera)

        pygame.display.update()

    def render_state(self):
        green = (153, 255, 153)
        red = (255, 153, 153)
        for i in range(VISION_RADIUS):
            for pos in perimeter(self.snake.head(), self.snake.direction, i + 1):
                color = red if self.collision_at(pos) else green
                render_pixel_alpha(pos, self.display, self.camera, color)


class AISnake(AbstractIndividuum):
    """
    AI Snake has a neural net and knows
    how to mutate and recombine
    """

    def __init__(self, neural_net):
        self.neural_net = neural_net

    @classmethod
    def new_random(cls):
        obj = cls.__new__(cls)
        super(AISnake, obj).__init__()
        obj.neural_net = EvolvingNeuralNet(
            [INPUT_LAYER, HIDDEN_LAYER, HIDDEN_LAYER2, OUTPUT_LAYER], bias=False
        )
        return obj

    def get_fitness(self, repetition_runs=10):
        """
        Returns fitness.

        Calculates average of n runs to decrease
        the influence of luck, i.e. good food spawns
        """
        fitness = 0
        game = AIGame()
        for _ in range(repetition_runs):
            game.play_ai(self.neural_net)
            fitness += game.snake.size() - 1
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


def render_pixel_alpha(pos, display, camera, color, scale=0.9):
    s = camera.size_pixel * scale
    pos_world = camera.bord_to_world(pos)
    rect = [pos_world.x, pos_world.y, s, s]
    draw_rect_alpha(display, color, rect)
    # pygame.draw.rect(display, color, rect)


def draw_rect_alpha(surface, color, rect):
    shape_surf = pygame.Surface(pygame.Rect(rect).size, pygame.SRCALPHA, 32)
    pygame.draw.rect(shape_surf, color, shape_surf.get_rect())
    surface.blit(shape_surf, rect)
