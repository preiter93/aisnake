from .params import BORD_WIDTH, BORD_HEIGHT
from .datatypes import Position, Direction


class Snake(object):
    """
    Representation of the snake.
    """

    def __init__(self, start_pos=None):
        """
        :start_pos: type Position
        """
        # Determine starting position
        if start_pos is None:
            # Start at the center
            start_pos = Position(BORD_WIDTH // 2, BORD_HEIGHT // 2)
        else:
            if type(start_pos) != Position:
                raise ValueError("Expect start_pos to be of type Position.")

        # Determine starting direction
        self.direction = Direction.UP

        # Collect body in list
        self.body = [start_pos]

    def __contains__(self, position):
        """
        Returns whether this snake is on s certain field
        """
        return position in self.body

    def reset(self):
        """
        Reset Snake
        """
        start_pos = Position(BORD_WIDTH // 2, BORD_HEIGHT // 2)
        self.body = [start_pos]
        self.direction = Direction.UP

    def size(self):
        """
        Returns size of snake (includes head)
        """
        return len(self.body)

    def score(self):
        """
        Return score
        """
        return self.size() - 1

    def head(self):
        """
        Return the position of the head.
        """
        return self.body[-1]

    def grow(self, position):
        """
        Snake moves and grows.

        :position: type Position
        """
        self.body.append(position)

    def move(self, position):
        """
        Move snake one forward.

        :position: type Position
        """
        self.body.pop(0)
        self.body.append(position)

    def turn_right(self):
        """
        Updates directions after right turn
        """
        self.direction = self.direction.turn_right()

    def turn_left(self):
        """
        Updates directions after left turn
        """
        self.direction = self.direction.turn_left()

    def next_ahead(self):
        """
        Return the next field Snake visits if going ahead
        """
        return Position.from_point(self.head(), self.direction, 1)

    def next_right(self):
        """
        Return the next field Snake visits if going to its right
        """
        return Position.from_point(self.head(), self.direction.turn_right(), 1)

    def next_left(self):
        """
        Return the next field Snake visits if going to its left
        """
        return Position.from_point(self.head(), self.direction.turn_left(), 1)


def transform_bord_to_snake(pos: Position, snake: Snake) -> Position:
    """
    Given a position on the bord, determine its
    position relative to the snakes coordinate system,
    where the head is at zero, with the y-axis along the
    its looking direction.
    """
    head = snake.head()
    direction = snake.direction
    pos_new = Position(pos.x - head.x, pos.y - head.y)
    if direction is Direction.RIGHT:
        pos_new.x, pos_new.y = -pos_new.y, pos_new.x
    elif direction is Direction.DOWN:
        pos_new.x, pos_new.y = -pos_new.x, -pos_new.y
    elif direction is Direction.LEFT:
        pos_new.x, pos_new.y = pos_new.y, -pos_new.x
    return pos_new
