import random
from .params import BORD_WIDTH, BORD_HEIGHT
from .datatypes import Position


class Food(object):
    """
    Representation of food.

    Currently we assume that there is only one piece
    of food on the bord.
    """

    def __init__(self, position=None):
        self.pos = position

    def place_at(self, position):
        """
        Place food back on bord

        :position: type Position
        """
        self.pos = position

    def remove(self):
        """
        Remove apple from bord.
        """
        self.pos = None

    def spawn(self, snake):
        """
        Spawn food at a random location.

        If snake object is given, we try to spawn the food
        outside the snake.
        """
        if snake is None:
            self.pos = self._get_random_position_on_board()
        else:
            # Try until an ampty spot has been found.
            # TODO: Find a more efficient way.
            spawn_attempts = 0
            while spawn_attempts < 1000:
                self.pos = self._get_random_position_on_board()
                if self.pos not in snake:
                    return
                spawn_attempts += 1
            raise ValueError("Food spawn attempts reached maximum limit!")

    @staticmethod
    def _get_random_position_on_board():
        x_pos = random.randint(1, BORD_WIDTH - 1)
        y_pos = random.randint(1, BORD_HEIGHT - 1)
        return Position(x_pos, y_pos)
