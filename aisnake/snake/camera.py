from .params import SIZE_PIXEL, BORD_WIDTH, BORD_HEIGHT
from .datatypes import Position


class Camera(object):
    """
    Defines a linear transformation from bord coordinates
    to camera coordinates.

    The camera coordinates run from 0...display size,
    while the bord coordinates are defined in pixel
    and the origin is at the left corner of the wall.
    """

    def __init__(self, size_pixel=SIZE_PIXEL, pixel_offsets=[1, 1, 1, 1]):
        """
        Offset is uniform in all direction, it
        is defined in pixels

        Note that pygame defines the y-direcition from top to bottom,
        with self.y_reversed we redefine it to bottom to top.

        :size_pixel: Size of a pixel
        :pixel_offsets: Distance from display border to game border
        [up, right, down, left]
        """
        self.size_pixel = size_pixel
        self.pixel_offsets = pixel_offsets
        self.y_reversed = True

    def total_width(self):
        return self.total_width_pixel() * self.size_pixel

    def total_height(self):
        return self.total_height_pixel() * self.size_pixel

    def total_width_pixel(self):
        return BORD_WIDTH + self.pixel_offsets[1] + self.pixel_offsets[3]

    def total_height_pixel(self):
        return BORD_HEIGHT + self.pixel_offsets[0] + self.pixel_offsets[2]

    def bord_to_world(self, pos):
        """
        Get world coordinates from bord coordinates
        """
        world_x = (pos.x + self.pixel_offsets[3]) * self.size_pixel
        if self.y_reversed:
            world_y = (
                BORD_HEIGHT - pos.y - 1 + self.pixel_offsets[2]
            ) * self.size_pixel
        else:
            world_y = (pos.y + self.pixel_offsets[2]) * self.size_pixel
        return Position(world_x, world_y)

    def bord_zero(self):
        """
        Origin of board.
        """
        if self.y_reversed:
            return Position(0, BORD_HEIGHT - 1)
        else:
            return Position(0, 0)
