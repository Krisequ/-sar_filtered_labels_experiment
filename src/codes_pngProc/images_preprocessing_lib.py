import numpy as np
import PIL
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = 4000000000


""" Function returning the approximate percentage of black area in the image  """
def black_image_percentage(source_image, step=10, threshold=2): #400 times faster
    hei, wid = source_image.shape
    top = (hei // step * wid // step)
    return sum(sum(pix_val <= threshold for pix_val in source_image[::step, ::step]))/top


#                                           x0, y0,              dx,  dy
def trim_image(source_image, starting_point=(0, 0), target_size=(512, 1024)):
    hei, wid = source_image.shape

    x_min = int(np.max([starting_point[0], 0]))
    y_min = int(np.max([starting_point[1], 0]))
    x_max = int(np.min([starting_point[0]+target_size[0], wid-1]))
    y_max = int(np.min([starting_point[1]+target_size[1], hei-1]))

    trimmed = Image.fromarray(np.uint8(source_image[y_min:y_max, x_min:x_max]))

    if trimmed.size == (target_size[0], target_size[1]):  # full image was obtained
        return trimmed
    else:  # needs padding
        result = Image.new(trimmed.mode, (target_size[0], target_size[1]), (0))
        # print(f" offset x = {x_min-starting_point[0]}, dy = {y_min-starting_point[1]}")
        result.paste(trimmed, (x_min-starting_point[0], y_min-starting_point[1]))
        return result


def get_subimage(source_image, starting_point = (0,0), target_size = (1024,1024), scale = 1):
    """
    Returns image with padding starting point can be subzero
    """
    # print(f"{target_size} vs {scale}")
    subimage = trim_image(source_image, starting_point, (target_size[0]*scale, target_size[1]*scale))
    if scale != 1:
        subimage = subimage.resize(target_size)
    return subimage


def cut_image_into_covering_puzzles(source_image,
                                    name_prefix='test',
                                    name_iterator=0,
                                    self_cover_rate=0.4,
                                    max_black_percentage=0.3,
                                    starting_padding=0.1,  # black rim at the start
                                    target_size=(1024, 1024),  # x, y
                                    scale=1
                                    ):
    hei, wid = source_image.shape

    wid_offset = int(starting_padding * target_size[0] * scale)
    hei_offset = int(starting_padding * target_size[1] * scale)
    complete_wid = int(wid + 2 * wid_offset)
    complete_hei = int(hei + 2 * hei_offset)

    wid_steps = int(np.round(complete_wid/(target_size[0] * (1 - self_cover_rate))))
    hei_steps = int(np.round(complete_hei/(target_size[1] * (1 - self_cover_rate))))
    # wid_step  = int(complete_wid//wid_steps)
    # hei_step  = int(complete_hei//hei_steps)
    wid_step = int(target_size[0] * scale * (1 - self_cover_rate))
    hei_step = int(target_size[1] * scale * (1 - self_cover_rate))

    for xx in range (0, wid_steps):
        for yy in range (0, hei_steps):
            subimage = get_subimage(source_image,
                                    (xx * wid_step - wid_offset, yy * hei_step - hei_offset),
                                    target_size=target_size,
                                    scale=scale)
            if black_image_percentage(np.array(subimage)) <= max_black_percentage:
                subimage.save(name_prefix+str(name_iterator)+'.png')
                name_iterator = name_iterator + 1

    return name_iterator
