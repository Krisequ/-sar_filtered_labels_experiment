import os
import sys
import threading

import PIL
from tif2png_lib import tif2png

source_dir = '../data/tifData/'
target_dir = '../dataSrc/pngData/'
# source_dir = 'H:/Capella/'
# target_dir = 'pngDataCapella/'

#  Search through source_dir for tif images
sources_list = os.listdir(source_dir)
sources_list = [f for f in sources_list if os.path.isfile(os.path.join(source_dir, f))
                                            and (f.rsplit('.')[-1] == 'tif'
                                            or f.rsplit('.')[-1] == 'TIF')]  # delete directories

if not os.path.exists(target_dir):  # Create target dir
    os.mkdir(target_dir)

target_list = os.listdir(target_dir)
target_list = [f.rsplit('.')[0] for f in target_list if (os.path.isfile(os.path.join(target_dir, f))
               and f.rsplit('.')[-1] == 'png')]
# delete directories, take targets without extension (.png)
src_list = []
for f in sources_list:
    if f[:-4] in target_list:
        print(f + 'Skipping - already transferred')
        continue
    else:
        src_list.append(f)

def tif2png_thread(src_list, target_dir):
    for f in src_list:
        try:
            tif2png(f, source_dir, target_dir, False)
        except PIL.UnidentifiedImageError:
            print(f"{f} problem: PIL.UnidentifiedImageError ", file=sys.stderr)

thrd_count = 3
thread_list = []
for cpu in range(0, thrd_count, 1):
    src = src_list[cpu*len(src_list)//thrd_count:(cpu+1)*len(src_list)//thrd_count]
    print(f"Starting a thread numero {cpu} with files {src}")
    thread_list.append(threading.Thread(target=tif2png_thread,
                                        args=(src, target_dir,)
                                        # args=(2,),
                                        # kwargs={src, target_dir})
                       )
                       )
    thread_list[-1].start()
