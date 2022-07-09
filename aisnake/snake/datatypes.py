from enum import Enum


class Turn(Enum):
    NO = 0
    LEFT = 1
    RIGHT = 2


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def turn_right(self):
        """
        Return direction to the right of the
        current direction.
        """
        if self is Direction.UP:
            return Direction.RIGHT
        elif self is Direction.RIGHT:
            return Direction.DOWN
        elif self is Direction.DOWN:
            return Direction.LEFT
        elif self is Direction.LEFT:
            return Direction.UP

    def turn_left(self):
        """
        Return direction to the left of the
        current direction.
        """
        if self is Direction.UP:
            return Direction.LEFT
        elif self is Direction.RIGHT:
            return Direction.UP
        elif self is Direction.DOWN:
            return Direction.RIGHT
        elif self is Direction.LEFT:
            return Direction.DOWN

    def to_vec(self):
        """
        Transforms direction into a 2-vector, i.e.
        up  -> [0, 1]
        right -> [1, 0]
        down -> [0, -1]
        left -> [-1, 0]
        """
        if self is Direction.UP:
            return [0, 1]
        elif self is Direction.RIGHT:
            return [1, 0]
        elif self is Direction.DOWN:
            return [0, -1]
        elif self is Direction.LEFT:
            return [-1, 0]

    def determine_turn(self, other):
        """
        Given two directions, determine wether we have to turn right,
        left or not to go from the current direction to the other direction.
        """
        if self.turn_right() == other:
            return Turn.RIGHT
        elif self.turn_left() == other:
            return Turn.LEFT
        return Turn.NO


class Position(object):
    """
    Position on Snake board.
    """

    def __init__(self, x: int, y: int):
        """
        :x: int, x-coordinate
        :y: int, y-coordinate
        """
        self.x = x
        self.y = y

    def __eq__(self, other):
        """
        Overrides the default implementation
        """
        if isinstance(other, Position):
            return self.x == other.x and self.y == other.y
        return NotImplemented

    def __repr__(self):
        return "P({}, {})".format(self.x, self.y)

    @classmethod
    def from_point(cls, position, direction: Direction, distance: int):
        """
        Calculate a new Position given a direction and distance
        between the current Position and the new Position.

        :position: Old Position
        :direction: Direction enum, pointing were the new Position should be
        :distance: int, Distance between old and new point, must be > 0.
        :return: New Position

        TODO: Find better method name...
        """
        if direction is Direction.UP:
            return Position(position.x, position.y + distance)
        elif direction is Direction.RIGHT:
            return Position(position.x + distance, position.y)
        elif direction is Direction.DOWN:
            return Position(position.x, position.y - distance)
        elif direction is Direction.LEFT:
            return Position(position.x - distance, position.y)

    @staticmethod
    def distance(pos_a, pos_b):
        """
        Determine distance between points
        """
        return [pos_a.x - pos_b.x, pos_a.y - pos_b.y]


def perimeter(position: Position, direction: Direction, radius: int):
    """
    Given a position and a direction, return all points that
    lay *radius* away.

    First element is the one ahead of the point, and then we
    rotate clockwise.

    Example, perimet with radius one returns 8 elements, starting
    with y:
    x x x
    y < x
    x x x
    """
    if direction is Direction.UP:
        vec_st, step = [0, radius], [1, 0]
    elif direction is Direction.RIGHT:
        vec_st, step = [radius, 0], [0, -1]
    elif direction is Direction.DOWN:
        vec_st, step = [0, -radius], [-1, 0]
    elif direction is Direction.LEFT:
        vec_st, step = [-radius, 0], [0, 1]

    # First
    rv = [Position(position.x + vec_st[0], position.y + vec_st[1])]
    vec = [vec_st[0] + step[0], vec_st[1] + step[1]]
    # Loop
    while vec != vec_st:
        # Append
        rv.append(Position(position.x + vec[0], position.y + vec[1]))
        # Check for corner
        if abs(vec[0]) == abs(vec[1]):
            # Turn 90 degree
            step[0], step[1] = step[1], -step[0]

        # Update vec
        vec[0], vec[1] = vec[0] + step[0], vec[1] + step[1]

    return rv
