"""
Matplotlib won't show the z value of a plot on mouseover
Set ax.format_coord = Formatter(matrix)
To enable this behavior
Example from:
http://stackoverflow.com/questions/27704490/interactive-pixel-information-of-an-image-in-python
"""
class Formatter(object):
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={:.4f}'.format(x, y, z)
