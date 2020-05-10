# Copyright (c) Open-MMLab. All rights reserved.
from enum import Enum
import random
import numpy as np

from mmcv.utils import is_str


class OLDColor(Enum):
    """An enum that defines common colors.

    Contains red, green, blue, cyan, yellow, magenta, white and black.
    """
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    cyan = (255, 255, 0)
    yellow = (0, 255, 255)
    magenta = (255, 0, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)

class Color(Enum):
    """An enum that defines common colors.

    Contains red, green, blue, cyan, yellow, magenta, white and black.
    """
    red   = (255,0,0)
    green = (0,128,0)
    lime =  (0,255,0)
    blue  =	(0,0,255)
    cyan  = (0,255,255)
    yellow= (255,255,0)
    magenta=(255,0,255)
    white = (255,255,255)
    black = (0,0,0)
    maroon= (128,0,0)
    purple= (128,0,128)
    navy  = (0,0,128)

def get_random_color():
  # except black and white
  colors = ['red', 'green', 'blue','cyan','yellow','magenta']
  col = random.sample(colors, 1)[0]
  return Color[col].value

def _label2color(labels):
  colors = ['black', 'green', 'red', 'blue','cyan','magenta', 'yellow']
  n = len(colors)
  color_strs = [colors[ int(k%n) ] for k in labels]
  color_values = [color_val(c) for c in color_strs]
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  return color_values

def color_val(color):
    """Convert various input to color tuples.

    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs

    Returns:
        tuple[int]: A tuple of 3 integers indicating BGR channels.
    """
    if is_str(color):
        return Color[color].value
    elif isinstance(color, Color):
        return color.value
    elif isinstance(color, tuple):
        assert len(color) == 3
        for channel in color:
            assert channel >= 0 and channel <= 255
        return color
    elif isinstance(color, int):
        assert color >= 0 and color <= 255
        return color, color, color
    elif isinstance(color, np.ndarray):
        assert color.ndim == 1 and color.size == 3
        assert np.all((color >= 0) & (color <= 255))
        color = color.astype(np.uint8)
        return tuple(color)
    else:
        raise TypeError('Invalid type for color: {}'.format(type(color)))

COLOR_MAP_RGB = (
    (241, 255, 82),
    (102, 168, 226),
    (0, 255, 0),
    (113, 143, 65),
    (89, 173, 163),
    (254, 158, 137),
    (190, 123, 75),
    (100, 22, 116),
    (0, 18, 141),
    (84, 84, 84),
    (85, 116, 127),
    (255, 31, 33),
    (228, 228, 228),
    (0, 255, 0),
    (70, 145, 150),
    (237, 239, 94),
)
IGNORE_COLOR = (0, 0, 0)

def label2color(labels):
  labels = labels.astype(np.int32).tolist()
  colors= [COLOR_MAP_RGB[i] for i in labels ]
  return colors
