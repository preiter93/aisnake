def convert_index_1d_to_2d(
    index: int, max_row: int, max_col: int, order="R"
) -> (int, int):
    """
    Given an index in a 1D list, return the index of row and
    col of a 2D array.

    :index: Index of one-dimensional list
    :max_row: Maximum number of rows in 2-D array
    :max_col: Maximum number of cols in 2-D array
    :order: "R" = row major, "C" = col major
    :return: Index of row and column

    >>> convert_index_1d_to_2d(9, 3, 4)
    (2, 1)

    >>> convert_index_1d_to_2d(9, 3, 4, order = "C")
    (0, 3)
    """
    if not index < max_col * max_row:
        raise IndexError("index {} out of range".format(index))
    if order == "R":
        return (index // max_col, index % max_col)
    elif order == "C":
        return (index % max_row, index // max_row)
    else:
        raise ValueError("Expect order parameter to be 'R' or 'C'.")


def convert_index_2d_to_1d(
    row: int, col: int, max_row: int, max_col: int, order="R"
) -> int:
    """
    Given row and column index of a 2D array, return the index
    of the flattened array

    :row: Row index
    :col: Column index
    :max_row: Maximum number of rows in 2-D array
    :max_col: Maximum number of cols in 2-D array
    :order: "R" = row major, "C" = col major
    :return: Index of flattened array

    >>> convert_index_2d_to_1d(1, 2, 3, 4)
    7

    >>> convert_index_2d_to_1d(1, 2, 3, 4, order = "C")
    6
    """
    if row >= max_row or col >= max_col:
        raise IndexError("index of row {} or col {} out of range".format(row, col))
    if order == "R":
        return row + col * max_row
    elif order == "C":
        return col + row * max_col
    else:
        raise ValueError("Expect order parameter to be 'R' or 'C'.")
