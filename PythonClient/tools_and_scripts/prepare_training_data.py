import logging
import argparse
import cv2
import os
from pathlib import Path
import numpy as np
import struct
import csv
from tqdm import tqdm
from datetime import datetime

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

def make_numpy_npz(input_dir, output_dir, target_type=np.float16, inspect=False, dry_run=False):

    buffers_of_interest_255rgb = ['1', '6', '14', '15', '16', '17', '18', '19', '20', '23'] # '10', '5'

    # buffers that are gray channel only except depth (but they are saved as 24 bit depth now)
    gray_buffers = ['15', '16', '18', '19', '20']

    # buffers_of_interest_255rgb = ['14']

    rootdir = input_dir

    for parent_dir in tqdm(os.listdir(rootdir)):
        # filename = os.fsdecode(file)
        parent_dir_path = os.path.join(rootdir, parent_dir)
        if (os.path.isdir(parent_dir_path)):
            screenshot_dir_path = os.path.join(parent_dir_path, "screenshot")
            if (os.path.isdir(screenshot_dir_path)):
                for file in os.listdir(screenshot_dir_path):
                    if file.endswith('.png' or '.PFM'):
                        concatenated_array = None
                        for current_buffer_id in buffers_of_interest_255rgb:
                            if current_buffer_id == '10':
                                file_to_pick = os.path.splitext(file)[0] + "_visualization" + ".png"
                            elif current_buffer_id == '1':
                                file_to_pick = os.path.splitext(file)[0] + ".pfm"
                            else:
                                file_to_pick = file
                            file_to_read_path = os.path.join(parent_dir_path,
                                                              current_buffer_id, file_to_pick)
                            if current_buffer_id == '1':
                                # depth
                                img_buff = read_pfm(file_to_read_path)

                                # normalize it so max value is close to 1
                                img_buff /= 20000

                                # TODO test alternative log scaling
                                #img_buff = np.log1p(img_buff)
                                #img_buff = img_buff / np.log1p(20000)

                                # TODO below is just inverse OP for the above log scaling
                                #scaling_factor = np.log1p(20000)  # the same scaling factor used when encoding the depth values
                                #y = buff_as_fp16.astype(np.float64)  # convert back to a higher precision type to avoid precision issues
                                #x = np.expm1(y * scaling_factor)  # reverse the scaling

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
                            elif current_buffer_id == '10':
                                #@TODO - not supported now
                                pass
                            else:
                                img_buff = cv2.imread(file_to_read_path)
                                # normalize it. All buffers here should be 0..255 before normalization
                                img_buff = img_buff.astype(target_type)
                                img_buff /= 255
                            # img_buff = cv2.resize(img_buff, (screenshot_w, screenshot_h))

                            if inspect:
                                # @todo this are after normalization stats
                                # Print the array's shape, minimum value, maximum value, mean, standard deviation, and data type
                                print("filename: ", file_to_read_path)
                                print("current_buffer_id: ", current_buffer_id)
                                print("Shape:", img_buff.shape)
                                print("Minimum value:", np.min(img_buff))
                                print("Maximum value:", np.max(img_buff))
                                print("Mean:", np.mean(img_buff))
                                print("Standard deviation:", np.std(img_buff))
                                print("Data type:", img_buff.dtype)
                                print("--------------------------------------")

                            if current_buffer_id in gray_buffers:
                                # Convert to a one-channel grayscale image
                                img_buff = img_buff[..., 0]

                            if len(img_buff.shape) == 2:
                                img_buff = np.expand_dims(img_buff, axis=-1)

                            concatenated_array = img_buff.astype(target_type) if concatenated_array is None \
                                else np.concatenate(
                                (concatenated_array, img_buff.astype(target_type)), axis=-1)

                        if not dry_run:
                            os.makedirs(os.path.join(output_dir, parent_dir, "NPZs"), exist_ok=True)
                            filename_to_save = os.path.splitext(file)[0] + ".npz"
                            output_file_path = os.path.join(output_dir, parent_dir, "NPZs", filename_to_save)
                            np.savez(output_file_path, data=concatenated_array)

def _get_color_list_for_stencils_transforms():
    # transforms from my samples (list1) to epe format (list2)

    # TODO: dictionary maybe

    list1, list2 = [], []
    # undefined (mostly street furniture)
    list1.append((255, 255, 255))
    list2.append((5, 5, 5))
    list1.append((0, 0, 0))
    list2.append((0, 0, 0))

    # buildings
    list1.append((121, 67, 28))
    list2.append((4, 4, 4))

    # sky
    list1.append((90, 162, 242))
    list2.append((23, 23, 23))

    # street furniture
    list1.append((112, 105, 191))
    list2.append((17, 17, 17))

    # cars
    list1.append((92, 31, 106))
    list2.append((26, 26, 26))

    # persons
    list1.append((196, 30, 8))
    list2.append((24, 24, 24))

    # lights and traffic lights
    list1.append((195, 237, 132))
    list2.append((19, 19, 19))

    # signs / traffic signs, bliboard
    list1.append((254, 12, 229))
    list2.append((20, 20, 20))

    # road
    list1.append((115, 176, 195))
    list2.append((6, 6, 6))

    # trees
    list1.append((19, 132, 69))
    list2.append((21, 21, 21))

    # freewway
    list1.append((135, 169, 180))
    list2.append((7, 7, 7))

    # terrain
    list1.append((220, 163, 49))
    list2.append((22, 22, 22))

    # water
    list1.append((54, 72, 205))
    list2.append((1, 1, 1))

    return list1, list2

def modify_image(img_buff, list1, list2):
    start = datetime.now()
    for i in range(len(list1)):
        img_buff = np.where(img_buff == list1[i], list2[i], img_buff)
    end = datetime.now()
    time = end - start
    print(time)

#niepotrzebne
def process_tile(img_buff, tile_index, list1, list2, tile_size, modify_image):
    i, j = tile_index
    tile = img_buff[i:i+tile_size, j:j+tile_size]
    modify_image(tile, list1, list2)
    return tile, tile_index


def stencil_rgb_to_gray(input_dir, output_dir):
    buffers_of_interest_255rgb = ['5']  # '5', '10'

    rootdir = input_dir

    list_my, list_epe = _get_color_list_for_stencils_transforms()

    # rootdir = os.fsencode(rootdir)
    for parent_dir in tqdm(os.listdir(rootdir)):
        # filename = os.fsdecode(file)
        parent_dir_path = os.path.join(rootdir, parent_dir)
        if (os.path.isdir(parent_dir_path)):
            screenshot_dir_path = os.path.join(parent_dir_path, "screenshot")
            if (os.path.isdir(screenshot_dir_path)):
                for file in tqdm(os.listdir(screenshot_dir_path)):
                    if file.endswith('.png' or '.PFM'):
                        for current_buffer in buffers_of_interest_255rgb:
                            file_to_pick = file
                            file_to_read_path = os.path.join(parent_dir_path,
                                                             current_buffer, file_to_pick)

                            output_file_path = os.path.join(output_dir, parent_dir, "gray_stencils", file_to_pick)

                            if os.path.isfile(output_file_path):
                                print(f"skipping {output_file_path}")
                            else:
                                img_buff = cv2.imread(file_to_read_path)
                                img_buff = cv2.cvtColor(img_buff, cv2.COLOR_BGR2RGB)
                                print("buff shp: ", img_buff.shape)

                                #self._get_color_list_for_stencils_transforms()
                                modify_image(img_buff, list1=list_my, list2=list_epe)

                                # img_buff = cv2.resize(img_buff, (screenshot_w, screenshot_h))
                                assert (len(img_buff.shape) == 3)

                                os.makedirs(os.path.join(output_dir, parent_dir, "gray_stencils"), exist_ok=True)
                                output_file_path = os.path.join(output_dir, parent_dir, "gray_stencils", file_to_pick)
                                cv2.imwrite(output_file_path, img_buff[:, :, 0])

def create_training_txt():
    data_dict = {
        "/mnt/d/Kamil/data_collected/airsim_drone/dataset_coffing": "fake",
        "/mnt/d/Kamil/data_collected/first_vids_matrix_like_real": "real",
        # Add more paths and types to the dictionary as needed
    }

    for path, data_type in data_dict.items():
        print(f"Processing: {path}")

        if data_type == "fake":
            dirs_of_interest = ["screenshot", "msegs/gray4k", "NPZs", "gray_stencils"]
            csv_data = []

            for subdir in os.listdir(path):
                subdir_path = os.path.join(path, subdir)

                if not os.path.isdir(subdir_path):
                    continue

                screenshot_dir = os.path.join(subdir_path, "screenshot")
                msegs_gray4k_dir = os.path.join(subdir_path, "msegs/gray4k")
                npzs_dir = os.path.join(subdir_path, "NPZs")
                gray_stencils_dir = os.path.join(subdir_path, "gray_stencils")

                screen_files = sorted(os.listdir(screenshot_dir))
                npz_files = sorted(os.listdir(npzs_dir))
                msegs_gray4k_files = sorted(os.listdir(msegs_gray4k_dir))
                gray_stencils_files = sorted(os.listdir(gray_stencils_dir))

                for screen_file, npz_file, msegs_gray4k_file, gray_stencil_file in zip(
                        screen_files, npz_files, msegs_gray4k_files, gray_stencils_files
                ):
                    screen_path = os.path.abspath(os.path.join(screenshot_dir, screen_file))
                    npz_path = os.path.abspath(os.path.join(npzs_dir, npz_file))
                    msegs_gray4k_path = os.path.abspath(os.path.join(msegs_gray4k_dir, msegs_gray4k_file))
                    gray_stencils_path = os.path.abspath(os.path.join(gray_stencils_dir, gray_stencil_file))

                    csv_data.append([screen_path, msegs_gray4k_path, npz_path, gray_stencils_path])

            output_dir = os.path.dirname(path)
            parent_dir = os.path.basename(path)
            output_csv_path = os.path.join(output_dir, parent_dir + ".txt")

            with open(output_csv_path, "w", encoding="UTF8", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(csv_data)

        elif data_type == "real":
            dirs_of_interest = ["screenshot", "msegs/gray4k"]
            csv_data = []

            for subdir in os.listdir(path):
                subdir_path = os.path.join(path, subdir)

                if not os.path.isdir(subdir_path):
                    continue

                screenshot_dir = os.path.join(subdir_path, "screenshot")
                msegs_gray4k_dir = os.path.join(subdir_path, "msegs/gray4k")

                screen_files = sorted(os.listdir(screenshot_dir))
                msegs_gray4k_files = sorted(os.listdir(msegs_gray4k_dir))

                for screen_file, msegs_gray4k_file in zip(screen_files, msegs_gray4k_files):
                    screen_path = os.path.abspath(os.path.join(screenshot_dir, screen_file))
                    msegs_gray4k_path = os.path.abspath(os.path.join(msegs_gray4k_dir, msegs_gray4k_file))

                    csv_data.append([screen_path, msegs_gray4k_path])

            output_dir = os.path.dirname(path)
            parent_dir = os.path.basename(path)

            output_csv_path = os.path.join(output_dir, parent_dir + ".txt")

            with open(output_csv_path, "w", encoding="UTF8", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(csv_data)

def convert_to_windows_paths():
    # run from windows
    # List of CSV files to process (in windows)
    csv_files = ["D:\\Kamil\\data_collected\\first_vids_matrix_like_real.txt",
                 "D:\\Kamil\\data_collected\\airsim_drone\\dataset_coffing.txt"]

    # Process each CSV file
    for csv_file in csv_files:
        modified_rows = []
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                modified_row = []
                for entry in row:
                    # Replace the Linux-like path with the Windows-friendly path
                    windows_path = entry.replace('/mnt/d/Kamil/data_collected/', 'D:\\Kamil\\data_collected\\')
                    windows_path = windows_path.replace('/', os.sep)
                    modified_row.append(windows_path)
                modified_rows.append(modified_row)

        # Create a new filename for the modified CSV file
        filename, ext = os.path.splitext(csv_file)
        new_filename = filename + '_win' + ext

        with open(new_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(modified_rows)

        print(f"Converted {csv_file} to {new_filename}")

def drop_alpha_channels_in_screenshots():
    patches = ["/mnt/c/users/Maciej/downloads/dataset_two_example_samples"]  # Replace with your list of patch paths

    for patch_path in patches:
        subdirectories = os.listdir(patch_path)
        for subdir in tqdm(subdirectories):
            subdir_path = os.path.join(patch_path, subdir)

            if not os.path.isdir(subdir_path):
                continue

            screenshot_dir = os.path.join(subdir_path, "screenshot")
            image_files = os.listdir(screenshot_dir)

            for image_file in image_files:
                if image_file.endswith(".png"):
                    image_path = os.path.join(screenshot_dir, image_file)
                    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

                    if image.shape[2] == 4:  # Check if image has 4 channels (RGBA)
                        # Convert to 3 channels (RGB) by dropping the alpha channel
                        image = image[:, :, :3]
                        # Write the modified image, overwriting the original file
                        cv2.imwrite(image_path, image)

def deprecated_create_training_txt(input_dir, output_dir):
    # @TODO this method is deprecated

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
    for parent_dir in tqdm(os.listdir(rootdir)):
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

def resize_msegs():
    patches = [
        # "/mnt/d/Kamil/data_collected/airsim_drone/dataset_coffing",
        "/mnt/c/users/Maciej/downloads/dataset_two_example_samples/",
        # Add more patch paths as needed
    ]

    for patch_path in patches:
        patch_name = os.path.basename(patch_path)
        print(f"Processing patch: {patch_name}")

        subdirectories = sorted(os.listdir(patch_path))


        for subdir in tqdm(subdirectories):
            subdir_path = os.path.join(patch_path, subdir)

            # Check if subdir is a directory
            if not os.path.isdir(subdir_path):
                continue

            screenshot_dir = os.path.join(subdir_path, "screenshot")
            msegs_gray_dir = os.path.join(subdir_path, "msegs/gray")
            msegs_gray4k_dir = os.path.join(subdir_path, "msegs/gray4k")

            # Create msegs.gray4k directory if it doesn't exist
            os.makedirs(msegs_gray4k_dir, exist_ok=True)

            # Get the list of files in screenshot_dir
            screenshot_files = sorted(os.listdir(screenshot_dir))

            for screenshot_file in screenshot_files:
                screenshot_path = os.path.join(screenshot_dir, screenshot_file)
                gray_mask_file = os.path.splitext(screenshot_file)[0] + ".png"
                gray_mask_path = os.path.join(msegs_gray_dir, gray_mask_file)
                gray_mask4k_path = os.path.join(msegs_gray4k_dir, gray_mask_file)

                # Check if gray mask exists
                if not os.path.isfile(gray_mask_path):
                    print(f"Error: Gray mask not found for subdir: {subdir}")
                    continue

                # Load the gray mask
                gray_mask = cv2.imread(gray_mask_path, cv2.IMREAD_GRAYSCALE)

                # Upscale the gray mask to 4k using INTER_NEAREST
                gray_mask4k = cv2.resize(gray_mask, (3840, 2160), interpolation=cv2.INTER_NEAREST)

                # Save the upscaled gray mask in msegs.gray4k directory
                cv2.imwrite(gray_mask4k_path, gray_mask4k)

                # print(f"Upscaled and saved gray mask for subdir: {subdir}")

        print(f"Finished processing patch: {patch_name}")

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
    do_resize_msegs = args["resize_msegs"]
    do_drop_alpha_channels = args["drop_alpha_channel"]
    do_convert_to_win_paths = args["windows_convert_paths"]

    if resize:
        resize_screenshots(input_dir, output_dir)
    if yt_vid:
        make_yt_vid(input_dir, output_dir)
    if numpy_npz:
        make_numpy_npz(input_dir, output_dir)
    if stencil_to_gray:
        stencil_rgb_to_gray(input_dir, output_dir)
    if do_create_training_txt:
        create_training_txt()
    if do_resize_msegs:
        resize_msegs()
    if do_drop_alpha_channels:
        drop_alpha_channels_in_screenshots()
    if do_convert_to_win_paths:
        convert_to_windows_paths()



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
        "--resize_msegs",
        type=bool,
        default=False,
        nargs="?",
        const=True,
        help="resize msegs to 4k. Provide patches inside python code",
    )
    parser.add_argument(
        "--drop_alpha_channel",
        type=bool,
        default=False,
        nargs="?",
        const=True,
        help="iterates all screenshot and rewriting images 4chann->3chann by dropping alpha channel",
    )
    parser.add_argument(
        "--windows_convert_paths",
        type=bool,
        default=False,
        nargs="?",
        const=True,
        help="convert paths from linux to windows system",
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