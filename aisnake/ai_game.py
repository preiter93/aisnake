import pygame
import time
from .snake.camera import Camera
from .snake.game import Game
from .snake.datatypes import Turn, perimeter
from .snake.snake import transform_bord_to_snake
from .ai_snake import AISnake
from .snake.food import Food


class AIGame(Game):
    """
    Let game play itself without GUI
    """

    def __init__(self, snake: AISnake):
        # super().__init__()
        assert isinstance(snake, AISnake)
        self.camera = Camera(pixel_offsets=[3, 3, 3, 3])
        self.snake = snake
        self.food = Food()
        self.food.spawn(self.snake)
        self.display = None
        self.score = 0

    def reset(self):
        """
        Reset snake and food
        """
        self.snake.reset()
        self.food.spawn(self.snake)
        self.score = 0

    def play_ai(self, visible=False, save_game=False):
        """
        Play ai invisible in the background

        :visible: Render game with pygame
        :save_game: Save rendered game as sequence of screenshots
        """
        if visible:
            pygame.init()
            self.display = pygame.display.set_mode(
                (self.camera.total_width(), self.camera.total_height())
            )
            pygame.display.set_caption("Game of Snake.")
            self.render()
            time.sleep(1)
            clock = pygame.time.Clock()
            if save_game:
                frame = 0

        game_over = False
        # TODO: Check copy
        snake_size = self.snake.size()
        # Hunger adds penalty for non-eating snakes
        self.snake.hunger = self.snake.max_hunger
        while not game_over:
            # Action from ai
            turn = self.ai_action(self.snake.neural_net)

            # Execute one timestep
            self.step(turn)

            # Render
            if visible:
                assert isinstance(self.snake, AISnake)
                self.render()
                if save_game:
                    file_name = "img_{:5}.png".format(str(frame).zfill(5))
                    self.save_screenshot(file_name)
                    frame += 1

            # Snake has eaten
            if self.snake.size() > snake_size:
                self.update_score()
                self.snake.hunger, snake_size = self.snake.max_hunger, self.snake.size()
            else:
                self.snake.hunger -= 1

            # Check collision and hunger
            game_over = any([self.collision(), self.snake.hunger == 0])

            if visible:
                clock.tick(10)  # FPS
        if visible:
            pygame.quit()

    def update_score(self, alpha=0):
        """
        We define the score as a weighted sum of number
        of apples eaten, + how fast they were found.

        :alpha: float, if 0, only number of apples eaten count as score
        """
        self.score += 1 + alpha * (1 / self.snake.hunger)

    def get_state(self):
        """
        Return state of the game that is fed into the AI model.

        """
        state = []

        for i in range(self.snake.vision_radius):
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
        render_score(self.snake.size() - 1, self.display, self.camera)

        pygame.display.update()

    def render_state(self):
        green = (203, 255, 203)
        red = (255, 203, 203)
        assert isinstance(self.snake, AISnake)
        for i in range(self.snake.vision_radius):
            for pos in perimeter(self.snake.head(), self.snake.direction, i + 1):
                color = red if self.collision_at(pos) else green
                render_pixel_alpha(pos, self.display, self.camera, color)


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
