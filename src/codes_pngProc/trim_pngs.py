from PIL import Image
from tqdm import tqdm
import numpy as np
import os

Image.MAX_IMAGE_PIXELS = 4000000000


def trim_png(filename, source_dir, target_dir):  # with tif format?
    """
    Trims black area off the images edges. Uses variable step for quicker processing.
    The assumption is that the trimming can cut off few pixels off the image (step-1)
    """
    print(f'Opening {source_dir + filename}', end='\t')
    source_image = np.asarray(Image.open(source_dir + filename))
    print(f'type {source_image.dtype}')
    wid, hei = source_image.shape

    min_x = wid
    min_y = hei
    max_x = 0
    max_y = 0
    step = 5

    for x in tqdm(range(0, wid, step)):
        for y in range(0, hei, step):
            if source_image[x, y] > 2:  # Is not empty/black
                if min_x > x:
                    min_x = x
                if min_y > y:
                    min_y = y
                if max_x < x:
                    max_x = x
                if max_y < y:
                    max_y = y

    if max_x >= wid - step and max_y >= hei - step and min_y <= step and min_x <= step:
        print('Skipping, left as is')
    else:
        source_image = source_image[min_x:max_x, min_y:max_y]  # trim to min max
        Image.fromarray(source_image).save(target_dir + filename)
        print('Saved')


source_dir = '../dataSrc/pngDataCapella/'
target_dir = '../dataSrc/pngDataCapella/'
if not os.path.exists(target_dir):  # Create target dir
    os.mkdir(target_dir)

sources_list = os.listdir(source_dir)
sources_list = [f for f in sources_list if os.path.isfile(os.path.join(source_dir, f))
                and (f.rsplit('.')[-1] == 'png')]
sources_list = sorted(sources_list)

for f in sources_list:
    trim_png(f, source_dir, target_dir)
