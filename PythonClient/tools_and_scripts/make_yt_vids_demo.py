import logging
import argparse
import cv2
import os
from pathlib import Path
import numpy as np
import struct
import csv

def get_total_screenfiles(dir):
    count = 0
    for parent_dir in os.listdir(dir):
        #filename = os.fsdecode(file)
        parent_dir_path = os.path.join(dir, parent_dir)
        if(os.path.isdir(parent_dir_path)):
            screenshot_dir_path = os.path.join(parent_dir_path, "screenshot")
            if (os.path.isdir(screenshot_dir_path)):
                for file in os.listdir(screenshot_dir_path):
                    count += 1
    return count

def make_yt_vid(input_dir, output_dir):
    screenshot_w = 3840
    screenshot_h = 2160

    repeats_count = 5

    buffers_of_interest_255rgb = ['14', '5', '6', '15', '16', '17', '18', '19', '20', '23']

    # buffers_of_interest_255rgb = ['14']

    rootdir = input_dir

    total_files = get_total_screenfiles(rootdir)
    total_files *= repeats_count
    print("total files:", total_files)
    # total_files = 2240
    files_per_buffer = total_files // len(buffers_of_interest_255rgb)

    # rootdir = os.fsencode(rootdir)
    files_iterated = 0  # 140
    current_buffer = 2  # 7
    for r in range(1, repeats_count):
        for parent_dir in os.listdir(rootdir):
            # filename = os.fsdecode(file)
            parent_dir_path = os.path.join(rootdir, parent_dir)
            if (os.path.isdir(parent_dir_path)):
                screenshot_dir_path = os.path.join(parent_dir_path, "screenshot")
                if (os.path.isdir(screenshot_dir_path)):
                    for file in os.listdir(screenshot_dir_path):
                        if file.endswith('.png'):
                            file_path = os.path.join(screenshot_dir_path, file)
                            if (buffers_of_interest_255rgb[current_buffer] == '10'):
                                file_to_pick = os.path.splitext(file)[0] + "_visualization" + ".png"
                            else:
                                file_to_pick = file
                            file_to_paste_path = os.path.join(parent_dir_path,
                                                              buffers_of_interest_255rgb[current_buffer], file_to_pick)
                            img = cv2.imread(file_path)
                            img_buff = cv2.imread(file_to_paste_path)
                            # img_buff = cv2.resize(img_buff, (screenshot_w, screenshot_h))
                            img[0:screenshot_h, screenshot_w // 2:screenshot_w] = \
                                img_buff[0:screenshot_h, screenshot_w // 2:screenshot_w]
                            os.makedirs(os.path.join(output_dir, parent_dir, str(r)), exist_ok=True)
                            output_file_path = os.path.join(output_dir, parent_dir, str(r), file)
                            cv2.imwrite(output_file_path, img)
                            files_iterated += 1
                            if (files_iterated > files_per_buffer):
                                files_iterated = 0
                                current_buffer = (current_buffer + 1) % len(buffers_of_interest_255rgb)
                    # for subdir, dirs, files in os.walk(rootdir):
    #    for dir in dirs:
    #        print(os.path.join(subdir, dir))
    # for file in files:
    # print(os.path.join(subdir, file))


def read_pfm(filename):
    with Path(filename).open('rb') as pfm_file:
        line1, line2, line3 = (pfm_file.readline().decode('latin-1').strip() for _ in range(3))
        assert line1 in ('PF', 'Pf')

        channels = 3 if "PF" in line1 else 1
        width, height = (int(s) for s in line2.split())
        scale_endianess = float(line3)
        bigendian = scale_endianess > 0
        scale = abs(scale_endianess)

        buffer = pfm_file.read()
        samples = width * height * channels
        assert len(buffer) == samples * 4

        fmt = f'{"<>"[bigendian]}{samples}f'
        decoded = struct.unpack(fmt, buffer)
        shape = (height, width, 3) if channels == 3 else (height, width)
        return np.flipud(np.reshape(decoded, shape)) * scale


def make_numpy_npz(input_dir, output_dir):
    screenshot_w = 3840
    screenshot_h = 2160

    repeats_count = 5

    buffers_of_interest_255rgb = ['1', '14', '6', '15', '16', '17', '18', '19', '20', '23'] # '5', '10'

    # buffers_of_interest_255rgb = ['14']

    rootdir = input_dir

    total_files = get_total_screenfiles(rootdir)
    print("total files:", total_files)
    # total_files = 2240
    files_per_buffer = total_files // len(buffers_of_interest_255rgb)

    # rootdir = os.fsencode(rootdir)
    for parent_dir in os.listdir(rootdir):
        # filename = os.fsdecode(file)
        parent_dir_path = os.path.join(rootdir, parent_dir)
        if (os.path.isdir(parent_dir_path)):
            screenshot_dir_path = os.path.join(parent_dir_path, "screenshot")
            if (os.path.isdir(screenshot_dir_path)):
                for file in os.listdir(screenshot_dir_path):
                    if file.endswith('.png' or '.PFM'):
                        concatenated_array = None
                        for current_buffer in buffers_of_interest_255rgb:
                            if current_buffer == '10':
                                file_to_pick = os.path.splitext(file)[0] + "_visualization" + ".png"
                            elif current_buffer == '1':
                                file_to_pick = os.path.splitext(file)[0] + ".pfm"
                            else:
                                file_to_pick = file
                            file_to_read_path = os.path.join(parent_dir_path,
                                                              current_buffer, file_to_pick)
                            if current_buffer == '1':
                                # depth
                                img_buff = read_pfm(file_to_read_path)

                                # render it like that to rgb
                                '''
                                print("img_buff shp:", img_buff.shape)
                                img_buff *= 0.001552 * 255.0
                                img_buff[img_buff > 255.0] = 255.0
                                img_buff[img_buff < 0.0] = 0.0
                                img_buff = img_buff.astype(np.uint8)
                                os.makedirs(os.path.join(output_dir, parent_dir, "tescik"), exist_ok=True)
                                output_file_path = os.path.join(output_dir, parent_dir, "tescik", file)
                                cv2.imwrite(output_file_path, img_buff)
                                '''
                            elif current_buffer == '10':
                                #@TODO
                                pass
                            else:
                                img_buff = cv2.imread(file_to_read_path)
                            # img_buff = cv2.resize(img_buff, (screenshot_w, screenshot_h))
                            # concatenate to one npz file
                            if(len(img_buff.shape) == 2):
                                img_buff = np.expand_dims(img_buff, axis=-1)

                            if concatenated_array is None:
                                concatenated_array = img_buff
                            else:
                                concatenated_array = np.concatenate((concatenated_array, img_buff), axis=-1)

                        os.makedirs(os.path.join(output_dir, parent_dir, "NPZs"), exist_ok=True)
                        filename_to_save = os.path.splitext(file)[0] + ".npz"
                        output_file_path = os.path.join(output_dir, parent_dir, "NPZs", filename_to_save)
                        # cv2.imwrite(output_file_path, img_buff)
                        np.savez(output_file_path, concatenated_array)
                        #np.savez_compressed(args.out_dir / f'{batch.path[0].stem}', data=f)

def stencil_rgb_to_gray(input_dir, output_dir):
    screenshot_w = 3840
    screenshot_h = 2160

    buffers_of_interest_255rgb = ['5']  # '5', '10'

    rootdir = input_dir

    # rootdir = os.fsencode(rootdir)
    for parent_dir in os.listdir(rootdir):
        # filename = os.fsdecode(file)
        parent_dir_path = os.path.join(rootdir, parent_dir)
        if (os.path.isdir(parent_dir_path)):
            screenshot_dir_path = os.path.join(parent_dir_path, "screenshot")
            if (os.path.isdir(screenshot_dir_path)):
                for file in os.listdir(screenshot_dir_path):
                    if file.endswith('.png' or '.PFM'):
                        for current_buffer in buffers_of_interest_255rgb:
                            file_to_pick = file
                            file_to_read_path = os.path.join(parent_dir_path,
                                                             current_buffer, file_to_pick)

                            img_buff = cv2.imread(file_to_read_path)
                            # img_buff = cv2.resize(img_buff, (screenshot_w, screenshot_h))
                            # concatenate to one npz file
                            assert (len(img_buff.shape) == 3)

                            # cars
                            img_buff[img_buff == (88, 207, 100)] = 26
                            img_buff[img_buff == (49, 89, 160)] = 27

                            # persons
                            img_buff[img_buff == (25, 175, 120)] = 24
                            img_buff[img_buff == (196, 30, 8)] = 25

                            #cars
                            #rgb, 88;207;100
                            #rgb, 49;89;160
                            #ppl
                            #rgb 25;175;120
                            #rgb 196;30;8

                            os.makedirs(os.path.join(output_dir, parent_dir, "gray_stencils"), exist_ok=True)
                            output_file_path = os.path.join(output_dir, parent_dir, "gray_stencils", file_to_pick)
                            cv2.imwrite(output_file_path, img_buff[:, :, 1])

def create_training_txt(input_dir, output_dir):
    '''
     The pipeline expects for each dataset a txt file containing paths to all images.
     Each line should contain paths to image, robust label map, gbuffer file, and ground truth label map, all separated by commas.
     Real datasets only require paths to images and robust label maps. For computing the label maps, see below.
    '''

    '''
    #!/bin/sh
    input_file=/mnt/d/Kamil/data_collected/first_vids_matrix_like_real/walking_chicago1080_qqWpy-64JjE
    model_name=mseg-3m
    model_path=mseg-3m.pth
    save_folder=/mnt/q/KAMIL/myEPE/
    config=/mnt/q/KAMIL/dependencies/mseg-semantic/mseg_semantic/config/test/1080/default_config_batched_ss.yaml
    python -u /mnt/q/KAMIL/dependencies/mseg-semantic/mseg_semantic/tool/universal_demo.py \
    --config=${config} model_name ${model_name} model_path ${model_path} input_file ${input_file} save_folder ${save_folder}
    '''

    dirs_of_interest = ["screenshot", 'gray', 'NPZs', 'gray_stencils'] # image, robust label map, gbuffer file, and ground truth label map

    rootdir = input_dir

    # rootdir = os.fsencode(rootdir)
    for parent_dir in os.listdir(rootdir):
        # filename = os.fsdecode(file)
        parent_dir_path = os.path.join(rootdir, parent_dir)
        if (os.path.isdir(parent_dir_path)):
            screenshot_dir_path = os.path.join(parent_dir_path, dirs_of_interest[0])
            gray_dir_path = os.path.join(parent_dir_path, dirs_of_interest[1])
            NPZs_dir_path = os.path.join(parent_dir_path, dirs_of_interest[2])
            gray_stencils_dir_path = os.path.join(parent_dir_path, dirs_of_interest[3])
            if (os.path.isdir(screenshot_dir_path)):
                csv_data = []
                for screen_filename in os.listdir(screenshot_dir_path):
                    if screen_filename.endswith('.png' or '.PFM'):
                        npz_file = os.path.splitext(screen_filename)[0] + ".npz"

                        screen_path = os.path.join(screenshot_dir_path, screen_filename)
                        npz_path = os.path.join(NPZs_dir_path, npz_file)
                        gray_path = os.path.join(gray_dir_path, screen_filename)
                        gray_stencils_path = os.path.join(gray_stencils_dir_path, screen_filename)

                        if(os.path.isfile(screen_path) and
                        os.path.isfile(npz_path) and
                        os.path.isfile(gray_path) and
                        os.path.isfile(gray_stencils_path)):
                            # fake data
                            csv_data.append([screen_path, npz_path, gray_path, gray_stencils_path])

                        if (os.path.isfile(screen_path) and
                                os.path.isfile(gray_path)):
                            # fake data
                            csv_data.append([screen_path, gray_path])

                output_csv_path = os.path.join(output_dir, parent_dir + ".txt")
                with open(output_csv_path, 'w', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    # write multiple rows
                    writer.writerows(csv_data)

def resize_screenshots(input_dir, output_dir, output_size = (3840,2160)):

    buffers_to_resize = ['gray'] #@TODO not yet supported ['screenshot', 'gray']

    rootdir = input_dir

    # rootdir = os.fsencode(rootdir)
    for parent_dir in os.listdir(rootdir):
        # filename = os.fsdecode(file)
        parent_dir_path = os.path.join(rootdir, parent_dir)
        if (os.path.isdir(parent_dir_path)):
            for buff in buffers_to_resize:
                dir_path = os.path.join(parent_dir_path, buff)
                if (os.path.isdir(dir_path)):
                    output_dir_name = buff + "_" + str(output_size[0]) + "x" + str(output_size[1])
                    os.makedirs(os.path.join(output_dir, parent_dir, output_dir_name), exist_ok=True)
                    for file in os.listdir(dir_path):
                        if file.endswith('.png' or '.PFM'):
                            file_to_pick = file
                            file_to_read_path = os.path.join(dir_path, file_to_pick)
                            if buff in ['gray', 'gray_stencil']:
                                img_buff = cv2.imread(file_to_read_path, 0)
                            else:
                                img_buff = cv2.imread(file_to_read_path)
                            # assert (len(img_buff.shape) == 3)
                            if buff in ['gray', 'gray_stencil']:
                                resized_image = cv2.resize(img_buff, output_size, interpolation = cv2.INTER_NEAREST)
                            else:
                                resized_image = cv2.resize(img_buff, output_size)
                            output_file_path = os.path.join(output_dir, parent_dir, output_dir_name, file_to_pick)
                            print("output_file_path")
                            cv2.imwrite(output_file_path, resized_image)

def main(args):
    logging.getLogger().setLevel(logging.INFO)

    input_dir = args["input_dir"]
    output_dir = args["output_dir"]
    yt_vid = args["yt_vid"]
    numpy_npz = args["numpy_npz"]
    stencil_to_gray = args["stencil_to_gray"]
    resize = args["resize"]
    do_create_training_txt = args["create_training_txt"]

    if(resize):
        resize_screenshots(input_dir, output_dir)
    if(yt_vid):
        make_yt_vid(input_dir, output_dir)
    if(numpy_npz):
        make_numpy_npz(input_dir, output_dir)
    if(stencil_to_gray):
        stencil_rgb_to_gray(input_dir, output_dir)
    if(do_create_training_txt):
        create_training_txt(input_dir, output_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        help="folder with separate recorded data runs",
        default="D:/Kamil/data_collected/airsim_drone/",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="save dir for produced frames",
        default="D:/Kamil/data_collected/airsim_drone_output",
    )
    parser.add_argument(
        "--yt_vid",
        type=bool,
        default=False,
        nargs="?",
        const=True,
        help="prepare showcase video for YT",
    )
    parser.add_argument(
        "--numpy_npz",
        type=bool,
        default=False,
        nargs="?",
        const=True,
        help="create NPZ numpy file for training",
    )
    parser.add_argument(
        "--stencil_to_gray",
        type=bool,
        default=False,
        nargs="?",
        const=True,
        help="save stencils as graymap for epe",
    )
    parser.add_argument(
        "--create_training_txt",
        type=bool,
        default=False,
        nargs="?",
        const=True,
        help="create training txt file",
    )
    parser.add_argument(
        "--resize",
        type=bool,
        default=False,
        nargs="?",
        const=True,
        help="resize screenshots for mseg input and save in forex. screenshot_1920x1080",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        nargs="?",
        const=True,
        help="print more info",
    )
    args = vars(parser.parse_args())
    main(args)