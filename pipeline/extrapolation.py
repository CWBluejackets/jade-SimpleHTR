# point = (x, y)
# y = 0 = top
# x = 0 = left
# cv2 ndarray.shape = (height, width)


def extrapolate_vertical(point0, point1, image_array):
    """
    Extrapolate to the top and bottom edges. Warning: nothing yet to guard
    against negative x-values (or past the bottom edge)
    :param point0:
    :param point1:
    :param image_array:
    :return: top_point, bottom_point, horizontal_difference
    """
    y_top = 0
    y_bottom = image_array.shape[0] - 1
    x_upper, y_upper = min(point0, point1, key=lambda p: p[1])
    x_lower, y_lower = max(point0, point1, key=lambda p: p[1])
    x_top = (y_top - y_lower) * (x_upper - x_lower) / (y_upper - y_lower) + x_lower
    x_top = int(x_top)
    x_bottom = (y_bottom - y_upper) * (x_upper - x_lower) / (y_upper - y_lower) + x_upper
    x_bottom = int(x_bottom)
    return (x_top, y_top), (x_bottom, y_bottom), abs(x_top - x_bottom)


def extrapolate_horizontal(point0, point1, image_array):
    """
    Extrapolate to the left and right edges. Warning: nothing yet to guard
    against negative x-values (or past the RH edge)
    :param point0:
    :param point1:
    :param image_array:
    :return: left_point, right_point, vertical_difference
    """
    x_left = 0
    x_right = image_array.shape[1] - 1
    x_0, y_0 = min(point0, point1, key=lambda p: p[0])
    x_1, y_1 = max(point0, point1, key=lambda p: p[0])

    y_left = y_1 - (x_1 - x_left) * (y_1 - y_0) / (x_1 - x_0)
    y_left = int(y_left)
    y_right = (x_right - x_0) * (y_1 - y_0) / (x_1 - x_0) + y_0
    y_right = int(y_right)
    return (x_left, y_left), (x_right, y_right), abs(y_right - y_left)

