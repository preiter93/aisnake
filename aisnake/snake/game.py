import pygame
from .snake import Snake
from .food import Food
from .params import *
from .camera import Camera
import time
from .datatypes import Turn, Direction
import os


class Game(object):
    """
    Snake game
    """

    def __init__(self):
        self.camera = Camera()
        self.snake = Snake()
        self.food = Food()
        self.food.spawn(self.snake)
        self.display = None

    def reset(self):
        """
        Reset snake and food
        """
        self.snake.reset()
        self.food.spawn(self.snake)

    def score(self):
        return self.snake.size() - 1

    def step(self, turn: Turn):
        """
        Play a single step.
        """

        # Turn
        if turn is Turn.RIGHT:
            self.snake.turn_right()
        elif turn is Turn.LEFT:
            self.snake.turn_left()

        # Grow or move
        next_pos = self.snake.next_ahead()
        if next_pos == self.food.pos:
            self.snake.grow(next_pos)
            self.food.spawn(self.snake)
        else:
            self.snake.move(next_pos)

    def play_gui(self):
        """
        Play game in pygame gui
        """

        # Initialize
        pygame.init()
        self.display = pygame.display.set_mode(
            (self.camera.total_width(), self.camera.total_height())
        )
        pygame.display.set_caption("Game of Snake.")
        self.display.fill(WHITE)

        # Press enter to start
        render_text("Press enter to start.", self.display)
        wait()
        self.render()
        time.sleep(2)

        # Game starts
        game_over = False
        clock = pygame.time.Clock()
        while not game_over:
            event_list = []
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_over = True
                # Append permitted events to list
                self.filter_event(event, event_list)
            # Handles double inputs as well
            key = self.handle_event_list(event_list)
            # Determine wether snake should turn
            turn = self.determine_turn(key)
            # Execute one timestep
            self.step(turn)
            # Render
            self.render()
            # Check collision
            game_over = self.collision()
            clock.tick(GAME_FPS)

        render_text("Game over.", self.display)
        time.sleep(2)
        pygame.quit()

    def collision(self):
        """
        Check for collision
        """
        return self.collision_at(self.snake.head())

    def collision_at(self, pos):
        """
        Check for collision at a given Position
        """
        return any([collision_snake(pos, self.snake), collision_border(pos)])

    @staticmethod
    def filter_event(event, event_list):
        """
        TODO: Find a better way to handle events.
        Current method is not accurate enough.
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                event_list.append(Direction.LEFT)
            elif event.key == pygame.K_RIGHT:
                event_list.append(Direction.RIGHT)
            elif event.key == pygame.K_UP:
                event_list.append(Direction.UP)
            elif event.key == pygame.K_DOWN:
                event_list.append(Direction.DOWN)

    def handle_event_list(self, event_list):
        """
        Takes care of double inputs by remembering the second
        key and execute it next time if next time no input is given.
        This improves handling of the game.
        """
        if not hasattr(self, "noted_key"):
            self.noted_key = None
        if len(event_list) == 0 and self.noted_key is None:
            key = None
        elif len(event_list) == 0:
            key = self.noted_key
            self.noted_key = None
        elif len(event_list) == 1:
            key = event_list[0]
        elif len(event_list) > 1:
            key = event_list[0]
            self.noted_key = event_list[1]
        return key

    def determine_turn(self, key):
        """
        Translates a key input to a snake turn.

        :key: type Direction or None
        :return: Turn enum that is used to update the game
        """
        if key is None:
            return Turn.NO
        if type(key) is Direction:
            return self.snake.direction.determine_turn(key)
        else:
            raise ValueError("Not implemented key input")

    def render(self):
        self.display.fill(COL_BORDER)
        render_field(self.display, self.camera)
        render_snake(self.snake, self.display, self.camera)
        render_food(self.food, self.display, self.camera)
        render_score(self.score(), self.display, self.camera)
        pygame.display.update()

    def save_screenshot(self, file_name):
        model_folder_path = "./img"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        pygame.image.save(self.display, file_name)


def collision_snake(pos, snake):
    """
    Detect collision of snake with itself
    """
    if pos in snake.body[:-1]:
        return True
    return False


def collision_border(pos):
    """
    Detect collision of snake with border of game
    """
    if pos.x >= BORD_WIDTH or pos.x < 0 or pos.y >= BORD_HEIGHT or pos.y < 0:
        return True
    return False


def render_snake(snake, display, camera):
    """
    Render snake in pygame
    """
    # Head
    render_pixel(snake.head(), display, camera, COL_SNAKE_HEAD)

    # Body
    for pos in snake.body[:-1]:
        render_pixel(pos, display, camera, COL_SNAKE_BODY)


def render_food(food, display, camera):
    render_pixel(food.pos, display, camera, COL_FOOD)


def render_field(display, camera, color=WHITE):
    """
    Render game field
    """
    zero_world = camera.bord_to_world(camera.bord_zero())
    w = BORD_WIDTH * camera.size_pixel
    h = BORD_HEIGHT * camera.size_pixel
    rect = [zero_world.x, zero_world.y, w, h]
    pygame.draw.rect(display, color, rect)


def render_pixel(pos, display, camera, color, scale=0.9):
    s = camera.size_pixel * scale
    pos_world = camera.bord_to_world(pos)
    rect = [pos_world.x, pos_world.y, s, s]
    pygame.draw.rect(display, color, rect)


def render_text(msg, display, dest=None, color=COL_MSG_GAMEOVER, update=True):
    font = pygame.font.SysFont(None, 20)
    text = font.render(msg, True, color)
    if dest is None:
        w, h = pygame.display.get_surface().get_size()
        # Centers text
        dest = text.get_rect(center=(w / 2, h / 2))
    display.blit(text, dest=dest)
    if update:
        pygame.display.update()


def render_score(score, display, camera, color=COL_MSG_SCORE):
    # from .datatypes import Position
    text = "Score: {:4}".format(str(score))
    font_style = pygame.font.SysFont(None, 16)
    text = font_style.render(text, True, color)
    w, _ = pygame.display.get_surface().get_size()
    dest = text.get_rect(center=(w / 2, camera.size_pixel / 2))
    display.blit(text, dest)


def wait():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                return
