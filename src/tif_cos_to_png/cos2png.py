import heapq
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import math
import PIL
import multiprocessing
import threading
PIL.Image.MAX_IMAGE_PIXELS = 1000000000

low_value_list = ['TSX_OPER_SAR_HS_SSC_20111211T193616_S01-965_E153-619_0000_v0100',
                  'TSX_OPER_SAR_SC_SSC_20210412T142723_N84-111_W004-614_0000_v0100',# squinted
                  'TSX_OPER_SAR_SC_SSC_20210412T142701_N83-272_E005-691_0000_v0100' # squinted
                  ]

def get_target_dir(filename, target_dir):
    return target_dir + filename[0:-4] + '.png'  # bez tif, z PNG


def form_data_array(data, wid, hei):
    return np.reshape(np.frombuffer(data, dtype=np.int16), (hei, wid//2))


def amplitude(x, y):
    return np.int16(math.sqrt(float(x)*x+float(y)*y))


def image_printer(output_img):
    # print('Output min max ', np.min(output_img), ' - ', np.max(output_img))
    fig, ax = plt.subplots()
    ax.imshow(output_img)
    plt.show()


def scale_image(data_array, hei, wid):
    output_image_re = np.zeros(( (hei-4), ((wid-8)//4)), np.uint32)
    output_image_im = np.zeros(( (hei-4), ((wid-8)//4)), np.uint32)

    # # for yy in tqdm(range(0, output_image_re.shape[0])): #Slower, more understandable way of doin next 3 lines
    # for yy in range(0, output_image_re.shape[0]):
    #     for xx in range(0, output_image_re.shape[1]):
    #         output_image_re[yy][xx] = data_array[yy+2][xx*2 + 4]
    #         output_image_im[yy][xx] = data_array[yy+2][xx*2 + 5]
    for yy in range(0, output_image_re.shape[0]):
        output_image_re[yy] = data_array[yy+4][4:: 2]
        output_image_im[yy] = data_array[yy+4][5:: 2]

    output_image_im = (output_image_im**2 + output_image_re**2)**0.5

    # Check
    # print('Input min max ', np.min(output_image_im), ' - ', np.max(output_image_im))
    topPercentile = int(output_image_im.size*0.03)
    top10Percent = 40000
    # top10Percent = heapq.nlargest(topPercentile, list(output_image_im.reshape(output_image_im.size)))
    # print('\t top ', topPercentile, ' val is ', top10Percent[-1])
    output_image_im = output_image_im * (255 / top10Percent)
    np.clip(output_image_im, 0, 255)

    return output_image_im.astype(np.uint8)


def cos2png(input_filename,target_dir):
    print(input_filename)
    with open(input_filename, 'rb') as file:
        data = file.read()

    BIB  = (int.from_bytes(data[0: 4 ], byteorder='big', signed=False))  # Bytes In Burst
    RSRI = (int.from_bytes(data[4: 8 ], byteorder='big', signed=False))  # Range Sample Relative Index
    RS   = (int.from_bytes(data[8: 12], byteorder='big', signed=False))  # Range Samples
    AS   = (int.from_bytes(data[12:16], byteorder='big', signed=False))  # Azimuth Samples
    BI   = (int.from_bytes(data[16:20], byteorder='big', signed=False))  # Burst Index
    RTNB = (int.from_bytes(data[20:24], byteorder='big', signed=False))  # Rangeline Total Number of Bytes
    TNL  = (int.from_bytes(data[28:32], byteorder='big', signed=False))  # Total Number of Lines          ---????

    beginWord = 28
    CSAR = [
        chr(int.from_bytes(data[beginWord  :beginWord+1], byteorder='big', signed=False)),
        chr(int.from_bytes(data[beginWord+1:beginWord+2], byteorder='big', signed=False)),
        chr(int.from_bytes(data[beginWord+2:beginWord+3], byteorder='big', signed=False)),
        chr(int.from_bytes(data[beginWord+3:beginWord+4], byteorder='big', signed=False))
    ]
    CSAR = "".join(CSAR)

    version      = (int.from_bytes(data[beginWord+4:beginWord+8 ], byteorder='big', signed=False))
    oversampling = (int.from_bytes(data[beginWord+8:beginWord+12], byteorder='big', signed=False))
    # inverseSPECAN= (float.from_bytes(data[beginWord+12:beginWord+20], byteorder='big', signed=False))

    if len(data) != BIB or CSAR != "CSAR":
        print(f"{input_filename} problem: {CSAR} file length {len(data)} vs Bib {BIB} ", file=sys.stderr)
        return
    assert CSAR == "CSAR"
    assert len(data) == BIB

    wid = RTNB
    hei = len(data)//RTNB
    assert BIB == wid*hei

    data_array = form_data_array(data[:hei*wid], wid, hei) # be aware of two-byte size of each element
    # image_printer(output_image_im)
    zdjecie_8_bitow = scale_image(data_array, hei, wid)
    PIL.Image.fromarray(zdjecie_8_bitow).save(get_target_dir(input_filename.rsplit('/')[-1], target_dir))
    print(f"{input_filename} Done")
          # f" - Max {np.max(zdjecie_8_bitow)} Min {np.min(zdjecie_8_bitow)}")


def cos2png_thread(src_list, target_dir):
    for f in src_list:
            cos2png(source_dir+f, target_dir)



source_dir = './data/'
target_dir = 'pngData/'
# target_dir = 'G:/ESA-SARX Database/pngData/'
# filenames = ["TSX_OPER_SAR_SM_SSC_20171012T162413_N40-447_E021-736_0000_v0100.cos"]



#  Find all source .cos images
sources_list = os.listdir(source_dir)
sources_list = [f for f in sources_list if os.path.isfile(os.path.join(source_dir, f))
                                            and (f.rsplit('.')[-1] == 'cos'
                                            or f.rsplit('.')[-1] == 'COS')]  # delete directories

if not os.path.exists(target_dir):
    os.mkdir(target_dir) # create target dir if doesnt exist

target_list = os.listdir(target_dir)
target_list = [f.rsplit('.')[0] for f in target_list if (os.path.isfile(os.path.join(target_dir, f))
               and f.rsplit('.')[-1] == 'png')]# delete directories, take targets without extension (.png), leave names without .png



src_list = []
for f in sources_list:
    if f[:-4] in target_list:
        print(f + ' skipping - already transferred')
        continue
    elif f[:-4] in low_value_list:
        print(f + ' low val - already transferred')
        continue
    else:
        src_list.append(f)


thrd_count = multiprocessing.cpu_count() - 1 # use almost all cores
# thrd_count=3
print(f"Found {thrd_count} CPU cores")

thread_list = []
for cpu in range(0, thrd_count, 1):
    src = src_list[cpu*len(src_list)//thrd_count:(cpu+1)*len(src_list)//thrd_count]
    print(f"Starting a thread numero {cpu} with files {src}")
    thread_list.append(threading.Thread(target=cos2png_thread,
                                        args=(src, target_dir,)))
    thread_list[-1].start()

for thrd in thread_list:  # Doesnt have to close them in any order, they can wait
    thrd.join()
    print("Thread closed")





