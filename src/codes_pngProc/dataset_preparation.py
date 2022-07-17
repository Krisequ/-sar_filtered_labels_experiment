import threading
import images_preprocessing_lib as preproc
import os
import numpy as np
import multiprocessing
from PIL import Image
Image.MAX_IMAGE_PIXELS = 4000000000


def load_and_split_whole_database_helper(local_src, target_dir, source_dir):
    """
    Function allowing for a very basic multiprocessing for the load_and_split_whole_database.
    It splits the given images into small (1024 x 1024) images.
    """
    for pic_src in local_src:
        target_path_whole = os.path.join(target_dir, pic_src.rsplit('.')[0])
        if os.path.exists(target_path_whole):
            print(target_path_whole, '- skipping, directory already found')
            continue
        os.mkdir(target_path_whole)  # create target dir if doesn't exist
        print(target_path_whole, '- created directory', end='\t')

        source_image = np.asarray(Image.open(os.path.join(source_dir, pic_src)))  # open image
        for i in range(1, 12):
            number_of_images = preproc.cut_image_into_covering_puzzles(source_image,
                                                                       name_prefix=target_path_whole + f"\\{i}_",
                                                                       name_iterator=0,
                                                                       self_cover_rate=0.0,
                                                                       max_black_percentage=0.2,
                                                                       starting_padding=0.1,
                                                                       target_size=(1024, 1024),
                                                                       scale=i
                                                                       )
            if number_of_images < 3:  # stop if only 2 or fewer images in this scale could be created
                break
        print('finished')


def load_and_split_whole_database(source_dir, target_dir):
    """
    Splits the given dataset into smaller (1024 x 1024) images using available threads.
    """
    sources_list = os.listdir(source_dir)
    sources_list = [f for f in sources_list if os.path.isfile(os.path.join(source_dir, f))
                    and (f.rsplit('.')[-1] == 'png'
                         or f.rsplit('.')[-1] == 'PNG')]  # delete directories

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)  # create target dir if doesn't exist

    target_list = os.listdir(target_dir)
    target_list = [f.rsplit('.')[0] for f in target_list if (os.path.isdir(os.path.join(target_dir, f)))]
    sources_list = [f for f in sources_list if f not in target_list]

    thrd_count = multiprocessing.cpu_count() - 1  # use almost all cores
    print(f"Found {thrd_count} CPU cores")
    thread_list = []
    for cpu in range(0, thrd_count):
        cpu_tasks = sources_list[cpu * len(sources_list) // thrd_count:(cpu + 1) * len(sources_list) // thrd_count]
        print(f"Starting a thread number {cpu} with files {cpu_tasks[1]}")
        thread_list.append(threading.Thread(target=load_and_split_whole_database_helper,
                                            args=(cpu_tasks, target_dir, source_dir,)))
        thread_list[-1].start()

    for thrd in thread_list:  # Doesn't have to close them in any order, they can wait
        thrd.join()
        print("Thread closed")


# Todo for each directory in dir
load_and_split_whole_database('..\\dataSrc\\pngDataCapella',
                              '..\\dataRdy\\train')
load_and_split_whole_database('..\\dataSrc\\pngData',
                              '..\\dataRdy\\train')
















