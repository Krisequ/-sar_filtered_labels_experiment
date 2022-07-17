import PIL
import matplotlib.pyplot as plt
import numpy as np
import time
PIL.Image.MAX_IMAGE_PIXELS = 4000000000

fileName2 = 'TSX_OPER_SAR_SC_GEC_20110506T011410_N26-011_W108-723_0000_v0100.tif'

"""
Extracted function to define target pictures
"""


def get_target_dir(filename, target_dir):
    return target_dir + filename[0:-4] + '.png'  # bez tif, z PNG


""" 
Function unpacking tiff files to png files.  
"""


def tif2png(filename, source_dir, target_dir, debug=False):  # z roszerzeniem tif
    print(filename, ' Start... ', end='\t')
    start_time = time.time()

    source_image = plt.imread(source_dir+filename)

    end_read = time.time()
    print('Reading: ', end_read - start_time, end='\t')
    if len(source_image.shape) > 2:
        print('ZAAAA WIEEEELKIIII : ', len(source_image.shape), end='\t')

    if debug:
        print('Input min max ', np.min(source_image), ' - ', np.max(source_image))
    # topPercentile = int(source_image.size*0.03)
    # top10Percent = heapq.nlargest(topPercentile, list(source_image.reshape(sourceImage.size)))
    # print('Min ', minVal,'\t max', maxVal, '\t top ', topPercentile, ' val is ', top10Percent[-1])

    cut_off_value = 510
    coefficient = 255/cut_off_value

    # exceedingVals = 0  # Number of values above threshold (making sure it doesnt exceed 2%
    # for pix in np.nditer(sourceImage): # Time 690 s
    #     if pix > cut_off_value:
    #         pix = cut_off_value
    #         exceedingVals = exceedingVals + 1
    #     pix = coefficient*pix
    # if exceedingVals > topPercentile:
    #     print(' UWAGA PRZEKROCZENIE - '+ exceedingVals/sourceImage.size + 'PROCENT')

    np.clip(source_image, 0, cut_off_value)
    source_image = source_image * coefficient

    end_adjust = time.time()
    print(' trimming: ', end_adjust - end_read, end='\t')

    output_img = source_image.astype('uint8')
    # del sourceImage  # zwalnianie pamięci
    end_type = time.time()
    print(' type adj: ', end_type - end_adjust, end='\t')

    if debug:
        print('Output min max ', np.min(output_img), ' - ', np.max(output_img))
        fig, ax = plt.subplots()
        #  ax.imshow(source_image)
        ax.imshow(output_img)
        plt.show()

    PIL.Image.fromarray(output_img).save(get_target_dir(filename, target_dir))
    end_save = time.time()
    print(' saving: ', end_save - end_type)
    return
