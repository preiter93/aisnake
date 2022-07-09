from .evolpy.abstract_individuum import AbstractIndividuum
from .ai.nnet import EvolvingNeuralNet
from .snake.snake import Snake
from .snake.game import Game
from .snake.camera import Camera
from copy import copy
from .snake.datatypes import Direction, Turn


INPUT_LAYER = 11
HIDDEN_LAYER = 16
OUTPUT_LAYER = 3
HIDDEN_LAYER2 = 16

MAX_HUNGER = 20


class AIGame(Game):
    """
    Let game play itself without GUI
    """

    def __init__(self):
        super().__init__()
        self.camera = Camera(pixel_offsets=[1, 20, 1, 1])

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
            turn = self.determine_turn(model)
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

        # Direction state
        state.append(self.snake.direction is Direction.UP)
        state.append(self.snake.direction is Direction.RIGHT)
        state.append(self.snake.direction is Direction.DOWN)
        state.append(self.snake.direction is Direction.LEFT)

        # Collision state
        state.append(self.collision_at_pos(self.snake.next_ahead()))
        state.append(self.collision_at_pos(self.snake.next_right()))
        state.append(self.collision_at_pos(self.snake.next_left()))

        # Food state
        dist_x = self.food.pos.x - self.snake.head().x
        dist_y = self.food.pos.y - self.snake.head().y
        state.append(dist_y > 0)
        state.append(dist_x > 0)
        state.append(dist_y < 0)
        state.append(dist_x < 0)
        return state

    def determine_turn(self, model):
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
        render_food(self.food, self.display, self.camera)
        render_score(self.score(), self.display, self.camera)
        self.render_state()
        pygame.display.update()

    def render_state(self, color=(255, 255, 255)):
        import pygame

        state = self.get_state()
        msg = [
            "UP",
            "RIGHT",
            "DOWN",
            "LEFT",
            "O-AHEAD",
            "O-RIGHT",
            "O-LEFT",
            "F-UP",
            "F-RIGHT",
            "F-DOWN",
            "F-LEFT",
        ]
        dy = self.camera.size_pixel * 0.6
        w, _ = pygame.display.get_surface().get_size()
        h = 2 * dy
        for s, m in zip(state, msg):
            h += dy
            text = "{:10}: {:4}".format(m, s)
            font_style = pygame.font.SysFont(None, 16)
            text = font_style.render(text, True, color)
            dest = text.get_rect(center=(w * 0.7, h))
            self.display.blit(text, dest)
        pygame.display.update()


class AISnake(AbstractIndividuum):
    def __init__(self, neural_net):
        # self.snake = Snake()
        self.neural_net = neural_net

    @classmethod
    def new_random(cls):
        obj = cls.__new__(cls)
        super(AISnake, obj).__init__()
        obj.snake = Snake()
        obj.neural_net = EvolvingNeuralNet(
            [INPUT_LAYER, HIDDEN_LAYER, HIDDEN_LAYER2, OUTPUT_LAYER]
        )
        return obj

    def get_fitness(self, repetition_runs=10):
        """
        TODO: ...
        """
        fitness = 0
        game = AIGame()
        for _ in range(repetition_runs):
            game.play_ai(self.neural_net)
            fitness += game.snake.size() - 1
            game.reset()
        return fitness / repetition_runs

    # @staticmethod
    def mutate(self):
        self.neural_net.mutate(25)

    @staticmethod
    def recombine(parents):
        offsprings = parents[0].neural_net.recombine(parents[1].neural_net)
        return [AISnake(neural_net) for neural_net in offsprings]
