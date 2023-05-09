import airsim
import numpy as np
import quaternion  # @todo, add to dependencies: pip install --user numpy numpy-quaternion
import time
import os
import tempfile
import pprint
from airsim.types import Pose, Vector3r, Quaternionr
import csv
from typing import List
import math
from tqdm import tqdm
import cv2
import random
import re
from threading import Thread


class Capture:
    def __init__(self, verbose=False, camera_paths=None, save_dir=None, generate_camera_sample_dir=None):
        # create airsim client and establish a connection
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        # self.tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_drone")
        self.tmp_dir = 'D:/Kamil/data_collected/airsim_drone/dataset_coffing'
        self.be_verbose = verbose
        self.save_dir = save_dir
        self.camera_paths_dir = camera_paths
        self.generate_camera_sample_dir = generate_camera_sample_dir
        # random.seed(777) # non-deterministic

    def getPoseAndRotation(self):
        pose = self.client.simGetVehiclePose()
        angles = airsim.to_eularian_angles(self.client.simGetVehiclePose().orientation)
        if self.be_verbose:
            print("x={}, y={}, z={}".format(pose.position.x_val, pose.position.y_val, pose.position.z_val))
            print("pitch={}, roll={}, yaw={}".format(angles[0], angles[1], angles[2]))
        return pose, angles

    def setSegmentationIDs(self):

        # before calling this, you should also set as many stencils as possible from the map itself
        # see AirSim\PythonClient\computer_vision\UECapture\set_stencils_from_editor.py --current nad --sublevels
        # run the above once with --sublevels from editor as python command then, in editor,
        # load level chunks from worldpartition and run for each chunk (as big as your RAM, for me 1/4 BigMap worked)
        # call above with --current. After that, you need to fly around map, spot some not updated stencils,
        # revisit some packedActorLevels and reload them
        # (right click) -> Actor Options -> Level -> Update Packed Level


        # set all meshes which owner name is according to regexes
        self.client.simSetMeshNamingMethodByString('owner')

        found = self.client.simSetSegmentationObjectID("[\w]*HLOD[\w]*", 18, True)  # building
        found = self.client.simSetSegmentationObjectID("[\w]*freeway[\w]*", 9, True)  # infrastructure
        found = self.client.simSetSegmentationObjectID("[\w]*MASS[\w]*", 17,
                                                       True)  # small humans and drivers, everything MASS not captured later
        found = self.client.simSetSegmentationObjectID("[\w]*CROWD[\w]*", 17, True)
        found = self.client.simSetSegmentationObjectID("[\w]*human[\w]*", 17, True)
        found = self.client.simSetSegmentationObjectID("[\w]*person[\w]*", 17, True)
        found = self.client.simSetSegmentationObjectID("[\w]*pedestrian[\w]*", 17, True)
        # set vehicles
        found = self.client.simSetSegmentationObjectID("BP_MassTraffic[\w]*", 42, True)  # BP_MassTraffic
        found = self.client.simSetSegmentationObjectID("[\w]*veh[\w]*", 42, True)
        found = self.client.simSetSegmentationObjectID("[\w]*Van[\w]*", 42, True)
        found = self.client.simSetSegmentationObjectID("[\w]*Car[\w]*", 42, True)
        found = self.client.simSetSegmentationObjectID("[\w]*Truck[\w]*", 42, True)
        found = self.client.simSetSegmentationObjectID("[\w]*Traffic[\w]*", 42, True)

        # these are now set up in editor before map before it is packaged, see set_stencils_from_editor.py
        '''
        self.client.simSetMeshNamingMethodByString('label')
        found = self.client.simSetSegmentationObjectID("[\w]*Street_Furniture[\w]*", 2, True) # street furniture -> 12
        found = self.client.simSetSegmentationObjectID("[\w]*Sidewalk[\w]*", 2, True) # street furniture
        found = self.client.simSetSegmentationObjectID("[\w]*StartingArea[\w]*", 2, True) # street furniture
        found = self.client.simSetSegmentationObjectID("[\w]*Plaza[\w]*", 2, True) # street furniture
        found = self.client.simSetSegmentationObjectID("[\w]*Bldg[\w]*", 18, True) # building
        found = self.client.simSetSegmentationObjectID("[\w]*_bld_[\w]*", 18, True) # building
        found = self.client.simSetSegmentationObjectID("[\w]*Roof[\w]*", 18, True) # building
        found = self.client.simSetSegmentationObjectID("[\w]*Decal[\w]*", 7, True) # road -> 9
        found = self.client.simSetSegmentationObjectID("[\w]*Ground[\w]*", 7, True) # road
        found = self.client.simSetSegmentationObjectID("[\w]*Road[\w]*", 7, True) # road
        found = self.client.simSetSegmentationObjectID("[\w]*UnderPass[\w]*", 7, True) # road
        found = self.client.simSetSegmentationObjectID("[\w]*freeway[\w]*", 9, True) # infrastructure
        found = self.client.simSetSegmentationObjectID("[\w]*Billboard[\w]*", 37, True) # billboard and traffic signs
        found = self.client.simSetSegmentationObjectID("[\w]*water_plane[\w]*", 28, True) # water -> 10
        '''

        # just to make additional sure we set as much stencils as possible
        self.client.simSetMeshNamingMethodByString('mesh')
        found = self.client.simSetSegmentationObjectID("[\w]*freeway[\w]*", 9, True)  # infrastructure
        found = self.client.simSetSegmentationObjectID("[\w]*bench[\w]*", 2, True)  # street furniture
        found = self.client.simSetSegmentationObjectID("[\w]*stairs[\w]*", 2, True)  # street furniture
        found = self.client.simSetSegmentationObjectID("[\w]*lamp[\w]*", 2, True)  # street furniture
        found = self.client.simSetSegmentationObjectID("[\w]*Road[\w]*", 7, True)  # road
        found = self.client.simSetSegmentationObjectID("[\w]*curb[\w]*", 7, True)  # road
        found = self.client.simSetSegmentationObjectID("[\w]*bldg_[\w]*", 18, True) # building
        found = self.client.simSetSegmentationObjectID("[\w]*roof_[\w]*", 18, True) # building
        found = self.client.simSetSegmentationObjectID("[\w]*Sign[\w]*", 37, True) # billboard and traffic signs
        found = self.client.simSetSegmentationObjectID("[\w]*signage[\w]*", 9, True) # infrastructure
        found = self.client.simSetSegmentationObjectID("[\w]*Light[\w]*", 32, True) # lights and traffic lights
        found = self.client.simSetSegmentationObjectID("[\w]*dirt_[\w]*", 39, True)  # terrain
        found = self.client.simSetSegmentationObjectID("[\w]*rock_[\w]*", 39, True)  # terrain
        found = self.client.simSetSegmentationObjectID("[\w]*vehicle[\w]*", 42, True)
        found = self.client.simSetSegmentationObjectID("[\w]*wheel[\w]*", 42, True)
        found = self.client.simSetSegmentationObjectID("[\w]*EuropeanHorn[\w]*", 31, True) # vegetation, trees
        found = self.client.simSetSegmentationObjectID("[\w]*SM_DOME[\w]*", 35, True)  # sky

    def read_camera_paths_from_disk(self, rotation_notation='quaternion'):
        camera_patches = {'rotation_notation': rotation_notation, 'patches': {}}

        for filename in os.listdir(self.camera_paths_dir):
            f = os.path.join(self.camera_paths_dir, filename)
            if filename.endswith('txt'):
                data = []
                with open(f, newline='') as csvfile:
                    data = list(csv.reader(csvfile))
                camera_patches['patches'][os.path.basename(filename)] = data

        for patch_name, patch_val in camera_patches['patches'].items():
            for step_id, step_val in enumerate(patch_val):
                for i in range(3):
                    camera_patches['patches'][patch_name][step_id][i] = float(camera_patches['patches'][patch_name][
                                                                                step_id][i]) # / 100
                for i in range(3, len(step_val) - 1):
                    camera_patches['patches'][patch_name][step_id][i] = float(camera_patches['patches'][patch_name][
                                                                                step_id][i])
                camera_patches['patches'][patch_name][step_id][-1] = int(camera_patches['patches'][patch_name][
                                                                              step_id][-1])
        return camera_patches

    def make_camera_demo_paths(self):
        # xval, yval, zval, pitch, roll, yaw
        camera_patches = {'rotation_notation': 'PitchRollYaw', 'patches': {
            "patch1": [[24.79802894592285, -2.81339168548584, 2.290256977081299, 0.0, 0.0, 0.31573680802245074, 0],
                       [24.79802894592285, -2.803391695022583, 2.290256977081299, 0.0, 0.0, 0.31573680802245074, 16666*1],
                       [24.79802894592285, -2.793391704559326, 2.290256977081299, 0.0, 0.0, 0.31573680802245074, 16666*2],
                       [24.79802894592285, -2.7833917140960693, 2.290256977081299, 0.0, 0.0, 0.31573680802245074, 16666*3],
                       [24.79802894592285, -2.7733917236328125, 2.290256977081299, 0.0, 0.0, 0.31573680802245074, 16666*4],
                       [24.79802894592285, -2.7633917331695557, 2.290256977081299, 0.0, 0.0, 0.31573680802245074, 16666*5],
                       [24.79802894592285, -2.753391742706299, 2.290256977081299, 0.0, 0.0, 0.31573680802245074, 16666*6]],
            "patch2": [[38.13249588012695, 5.8880462646484375, -52.08500671386719, -0.9075711616403678,
                        -4.4682337973070176e-10, 0.3855499677322994, 0],
                       [38.142494201660156, 5.8880462646484375, -52.09500503540039, -0.9075711616403678,
                        -4.4682337973070176e-10, 0.3855499677322994, 16666*1],
                       [38.15249252319336, 5.8880462646484375, -52.105003356933594, -0.9075711616403678,
                        -4.4682337973070176e-10, 0.3855499677322994, 16666*2],
                       [38.16249084472656, 5.8880462646484375, -52.11499786376953, -0.9075711616403678,
                        -4.4682337973070176e-10, 0.3855499677322994, 16666*3],
                       [38.172489166259766, 5.8880462646484375, -52.12499237060547, -0.9075711616403678,
                        -4.4682337973070176e-10, 0.3855499677322994, 16666*4],
                       [38.18248748779297, 5.8880462646484375, -52.13499069213867, -0.9075711616403678,
                        -4.4682337973070176e-10, 0.3855499677322994, 16666*5],
                       [38.19248580932617, 5.8880462646484375, -52.144989013671875, -0.9075711616403678,
                        -4.4682337973070176e-10, 0.3855499677322994, 16666*6]]
        }}
        return camera_patches

    def search_nearby_times(self, nums: List[int], target: int) -> (int, int):
        '''
        searching for left and right neighboor ids for target time stamp (as int).
        Input is already sorted array.

        Args:
            nums: list of times (as ints)
            target: target time stamp (as int)

        Returns: indices for surrounding target timestamp. (left, right indices)
        '''

        def binary_search_nearby_times(nums, left, right, target):
            if right >= left:
                halfIndex = math.floor((left + right) / 2)

                #print("left, right: ", left, right)
                #print("target: ", target)
                #print("halfIndex: ", halfIndex)
                #print("nums LIST: ", nums)

                if left == right - 1 or left == right:
                    # left == right should occur only if target is equal to the last elem
                    return left, right
                elif nums[halfIndex] > target:
                    right = halfIndex
                    return binary_search_nearby_times(nums, left, right, target)
                elif nums[halfIndex] <= target:
                    left = halfIndex
                    return binary_search_nearby_times(nums, left, right, target)
            else:
                return -1, -1

        return binary_search_nearby_times(nums, 0, len(nums) - 1, target)

    def interpolate_position_and_rotation(self, passed_simulation_time_usec, camera_patch, rotation_notation,
                                          camera_time_lists):

        left, right = self.search_nearby_times(camera_time_lists, passed_simulation_time_usec)

        if self.be_verbose:
            print('-- interpolate_position_and_rotation stats --')
            print("left, right boundaries ids: ", left, right)
            print("left, right boundaries vals: ", camera_patch[left][-1], camera_patch[right][-1])
            print("target val: ", passed_simulation_time_usec)

        coords_prev = camera_patch[left]
        coords_next = camera_patch[right]

        if rotation_notation == 'quaternion':
            assert len(coords_prev) == 8 == len(coords_next)
            R1 = np.quaternion(coords_prev[6], coords_prev[3], coords_prev[4], coords_prev[5]) #constructor is w,x,y,z
            R2 = np.quaternion(coords_next[6], coords_next[3], coords_next[4], coords_next[5])
        elif rotation_notation == 'PitchRollYaw':
            assert len(coords_prev) == 7 == len(coords_next)
            o1 = airsim.to_quaternion(coords_prev[3], coords_prev[4], coords_prev[5])
            o2 = airsim.to_quaternion(coords_next[3], coords_next[4], coords_next[5])
            R1 = np.quaternion(o1.w_val, o1.x_val, o1.y_val, o1.z_val) #constructor is w,x,y,z
            R2 = np.quaternion(o2.w_val, o2.x_val, o2.y_val, o2.z_val)
        else:
            raise NotImplemented

        t1 = coords_prev[-1]
        t2 = coords_next[-1]
        t_out = passed_simulation_time_usec

        if self.be_verbose:
            print("R1, R2", R1, R2)
            print("t1, t2, tout", t1, t2, t_out)

        position_val_prev = Vector3r(x_val=coords_prev[0], y_val=coords_prev[1], z_val=coords_prev[2])
        position_val_next = Vector3r(x_val=coords_next[0], y_val=coords_next[1], z_val=coords_next[2])
        # position_interpolated = (position_val_prev + position_val_next) / 2

        if(t2 - t1 != 0):
            qi = quaternion.slerp(R1, R2, t1, t2, t_out)
            quaternion_interpolated = Quaternionr(x_val=qi.x, y_val=qi.y, z_val=qi.z, w_val=qi.w)

            part_r = (t_out-t1)/(t2 - t1)
            part_l = 1 - part_r
            position_interpolated = position_val_prev*part_l + position_val_next*part_r
        else:
            # avoid zero division
            quaternion_interpolated = Quaternionr(x_val=R1.x, y_val=R1.y, z_val=R1.z, w_val=R1.w)
            position_interpolated = Vector3r(x_val=coords_prev[0], y_val=coords_prev[1], z_val=coords_prev[2])

        if self.be_verbose:
            print("quaternion_interpolated: ", quaternion_interpolated)
            print("position_interpolated: ", position_interpolated)

        return position_interpolated, quaternion_interpolated

    def toNED(self, pos, quat, scale=0.01):
        '''
        UU to NED (unreal engine coordinates to Airsim coordinates)
        Args:
            position:
            quaternion:

        Returns:

        '''

        ned_pos = Vector3r(x_val=pos.x_val, y_val=pos.y_val, z_val=-1*pos.z_val) * scale
        ned_quat = Quaternionr(x_val=-1*quat.x_val, y_val=-1*quat.y_val, z_val=quat.z_val, w_val=quat.w_val)

        return ned_pos, ned_quat

    def get_pose_from_camera_path(self, elapsed_simulation_time_msec, camera_path, rotation_notation, camera_time_list):
        position_val, orientation_val = self.interpolate_position_and_rotation(elapsed_simulation_time_msec,
                                                                               camera_path,
                                                                               rotation_notation,
                                                                               camera_time_list)
        position_val, orientation_val = self.toNED(position_val, orientation_val)
        # position_val.z_val = position_val.z_val + 5.0  # @todo shift initial pos, orientat; add init pose support
        pose = Pose(position_val=position_val, orientation_val=orientation_val)
        return pose

    def _grab_and_save_images_for_pose(self, pose, save_path_dir, save_img_id = 0, apply_unpause = False):
        # save_path_dir = os.path.join(self.tmp_dir, path_name)
        self.client.simPause(True)

        if self.be_verbose:
            self.getPoseAndRotation()

        self.client.simPause(False)
        self.client.simSetVehiclePose(pose, True)
        time.sleep(self.get_target_simulation_time_step())
        self.client.simPause(True)

        responses = self.client.simGetImages([
            ### airsim.ImageRequest("0", airsim.ImageType.Scene), # this is replaced by screenshots. Only for preview.
            # airsim.ImageRequest("0", airsim.ImageType.Velocity, pixels_as_float=False, pixels_as_float_RGB=True, compress=False), # velocity buffer is captured earliel, as at this point it is erased because of client.simPause(True) @todo TEST NEW
            airsim.ImageRequest("0", airsim.ImageType.SurfaceNormals, pixels_as_float=False, compress=False),
            airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=False),
            airsim.ImageRequest("0", airsim.ImageType.Segmentation, pixels_as_float=False, compress=False),
            # airsim.ImageRequest("0", airsim.ImageType.ShaderID, pixels_as_float = False, compress = False),
            ### probably very limited @todo, decide do we want it
            ### airsim.ImageRequest("0", airsim.ImageType.TextureUV), # not yet supported
            ### airsim.ImageRequest("0", airsim.ImageType.SceneDepth, pixels_as_float = True, compress = False), # same as airsim DepthPlanar, but depthplannar is scaled to 0..1
            airsim.ImageRequest("0", airsim.ImageType.ReflectionVector, pixels_as_float=False, compress=False),
            airsim.ImageRequest("0", airsim.ImageType.DotSurfaceReflection, pixels_as_float=False, compress=False),
            airsim.ImageRequest("0", airsim.ImageType.Metallic, pixels_as_float=False, compress=False),
            airsim.ImageRequest("0", airsim.ImageType.Albedo, pixels_as_float=False, compress=False),
            airsim.ImageRequest("0", airsim.ImageType.Specular, pixels_as_float=False, compress=False),
            airsim.ImageRequest("0", airsim.ImageType.Opacity, pixels_as_float=False, compress=False),
            airsim.ImageRequest("0", airsim.ImageType.Roughness, pixels_as_float=False, compress=False),
            ### airsim.ImageRequest("0", airsim.ImageType.Anisotropy), # no valuable info observed
            ### airsim.ImageRequest("0", airsim.ImageType.AO), # no valuable info observed
            airsim.ImageRequest("0", airsim.ImageType.ShadingModelColor, pixels_as_float=False,
                                compress=False)])  # probably very limited @todo, decide do we want it

        # add velocity buffer captured earlier
        self.client.simPause(True)

        save_screenshot_dir = os.path.join(save_path_dir, "screenshot")
        if not os.path.exists(save_screenshot_dir):
            os.makedirs(save_screenshot_dir)
        # print(os.path.join(save_screenshot_dir.replace(os.sep, '/'), str(step_id).zfill(5) + '.png'))
        success = self.client.simTakeScreenshot(
            os.path.join(save_screenshot_dir.replace(os.sep, '/'), str(save_img_id).zfill(5) + '.png'))
        if self.be_verbose:
            print("image taken")
        if self.be_verbose:
            print("taking images")

        ### responses.extend(responses_velocity) # @todo TEST NEW

        for response in responses:
            save_dir_per_response = os.path.join(save_path_dir, str(response.image_type))
            if not os.path.exists(save_dir_per_response):
                os.makedirs(save_dir_per_response)
            if response.pixels_as_float:
                # for some reason it is flipped vertically. Reflip it.
                image_arr = airsim.get_pfm_array(response)
                image_arr = np.flip(image_arr, axis=0)  # need for depth buffer
                airsim.write_pfm(os.path.normpath(os.path.join(save_dir_per_response, str(save_img_id).zfill(5) + '.pfm')),
                                 image_arr)
            elif response.pixels_as_float_RGB:
                img1d = np.array(response.image_data_float_RGB, dtype=np.float)
                im = img1d.reshape(response.height, response.width, 3)
                im_no_alpha = im[:, :, :3]

                last_delta_usecs = self.client.simGetDeltaTime() * (1 / self.get_time_dilation()) * 1000000
                # velocity needs to support -2..2 screen space range for x and y
                # for visualization purposes:
                im_no_alpha -= 0.5
                im_no_alpha /= 0.5
                im_no_alpha = im_no_alpha * np.sqrt(self.get_targeted_time_step_usec() / (last_delta_usecs *
                                                                               self.get_time_dilation()))
                im_no_alpha *= 0.5
                im_no_alpha += 0.5

                # filename = os.path.normpath(os.path.join(save_dir, str(step_id).zfill(5) + '.npy'))
                # np.save(filename, im_no_alpha) #@todo unlock it later with switch

                im_no_alpha = im_no_alpha * 255
                im_no_alpha = np.clip(im_no_alpha, 0, 255)
                im_no_alpha = im_no_alpha.astype('uint8')
                cv2.imwrite(os.path.normpath(os.path.join(save_dir_per_response, str(save_img_id).zfill(5) + '_visualization.png')),
                            im_no_alpha)
            else:

                # Change images into numpy arrays.
                img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
                im = img1d.reshape(response.height, response.width, 3)
                im_no_alpha = im[:, :, :3]
                cv2.imwrite(os.path.normpath(os.path.join(save_dir_per_response, str(save_img_id).zfill(5) + '.png')), im_no_alpha)

        if apply_unpause:
            self.client.simPause(False)

    def iterate_all_camera_patches(self, camera_pathes):
        for path_name, camera_path in camera_pathes['patches'].items():
            self.client.simPause(True)
            camera_time_list = []
            for step_id, coords in enumerate(camera_path):
                elapsed_camera_paths_time_msec = coords[-1]
                camera_time_list.append(elapsed_camera_paths_time_msec)
            total_steps = camera_time_list[-1] // self.get_targeted_time_step_usec()
            for step_id in tqdm(range(int(total_steps))):
                elapsed_simulation_time_usec_now = int(self.get_targeted_time_step_usec() * step_id)
                pose_now = self.get_pose_from_camera_path(elapsed_simulation_time_usec_now, camera_path, camera_pathes[
                                                                                   'rotation_notation'],
                                               camera_time_list)
                if self.be_verbose:
                    self.getPoseAndRotation()

                save_path_dir = os.path.join(self.tmp_dir, path_name)
                self._grab_and_save_images_for_pose(pose=pose_now, save_path_dir=save_path_dir, save_img_id=step_id,
                                                    apply_unpause=False)
        self.client.simPause(False)  # leave demo unpaused

    def _get_next_sample_dir_name(self, pattern=r"sample\d{5}$"):

        if not os.path.isdir(os.path.join(self.tmp_dir)):
            os.makedirs(self.tmp_dir)

        # Get a list of all directories in self.tmp_dir that match the pattern
        dir_list = [d for d in os.listdir(self.tmp_dir) if
                    os.path.isdir(os.path.join(self.tmp_dir, d)) and re.match(pattern, d)]

        if dir_list:
            # Extract the number from the directory names and convert to integers
            num_list = [int(d[6:]) for d in dir_list]

            # Find the largest number in the list
            max_num = max(num_list)
        else:
            max_num = -1

        # Create the name of the new directory
        new_dir_name = f"sample{max_num + 1:05d}"

        return new_dir_name


    def save_random_movement_sample(self, camera_pathes, frame_num_to_save = 10, player_pose=None):
        '''
        randomize part of camera patch (cut small movement trace/path), move along it, and save N frames
        camera_pathes - camera_pathes allowed to random from patch
        pose - pose at which you want to start movement and save; if None, current camera/veh pose will be taken
        '''

        self.client.simPause(True)

        next_dir_name = self._get_next_sample_dir_name()

        if player_pose is None:
            player_pose = self.client.simGetVehiclePose()

        path_name, camera_path = random.choice(list(camera_pathes['patches'].items()))
        camera_time_list = []

        print("path_name", path_name)

        # populate camera time list (last coord parameter)
        for step_id, coords in enumerate(camera_path):
            elapsed_camera_paths_time_msec = coords[-1]
            camera_time_list.append(elapsed_camera_paths_time_msec)

        # total possible steps, divide camera list last time point by time step duration in simulator (all in usecs)
        total_steps = camera_time_list[-1] // self.get_targeted_time_step_usec()

        # radnomize starting point in the particular camera path
        start = random.randint(0, max(0, int(total_steps) - frame_num_to_save))

        print("start:", start)

        for step_id in tqdm(range(start, start + frame_num_to_save)):
            elapsed_simulation_time_usec_now = int(self.get_targeted_time_step_usec() * step_id)
            camera_pose = self.get_pose_from_camera_path(elapsed_simulation_time_usec_now, camera_path, camera_pathes[
                                                                               'rotation_notation'],
                                           camera_time_list)

            if step_id == start:
                new_xyz = player_pose.position
                new_orientation = player_pose.orientation
                # init cam_zero_pose, cam_zero_orientatio
                cam_zero_pose, cam_zero_orientation = camera_pose.position, camera_pose.orientation
                cam_zero_orientation_inverted = cam_zero_orientation.inverse()
            else:
                new_xyz = player_pose.position + (camera_pose.position - cam_zero_pose)
                # new_orientation = camera_pose.orientation * pose.orientation  # @todo ,non cummutative op
                q1 = cam_zero_orientation_inverted * camera_pose.orientation
                new_orientation = q1 * player_pose.orientation

            #new_xyz = pose.position + camera_pose.position
            #new_orientation = pose.orientation * camera_pose.orientation

            new_pose = Pose(position_val=new_xyz, orientation_val=new_orientation)

            save_path_dir = os.path.join(self.tmp_dir, next_dir_name)
            self._grab_and_save_images_for_pose(pose=new_pose, save_path_dir=save_path_dir, save_img_id=step_id-start,
                                                apply_unpause=False)
        self.client.simPause(False)  # leave demo unpaused

    def set_targeted_time_step_usec(self, targeted_time_step_usec):
        self._targeted_time_step_usec = targeted_time_step_usec

    def get_targeted_time_step_usec(self):
        return self._targeted_time_step_usec

    def set_time_dilation(self, time_dilation):
        self._time_dilation = time_dilation

    def get_time_dilation(self):
        return self._time_dilation

    def set_target_simulation_time_step(self, target_simulation_time_step):
        self._target_simulation_time_step = target_simulation_time_step

    def get_target_simulation_time_step(self):
        return self._target_simulation_time_step

    def record(self, camera_pathes=None, time_dilation=0.01, target_fps=60):
        '''
        performs camera recording

        Args:
            time_dilation: how much you want to slow game engine. 0.01 means 100x. Please note that according value
            (1/time_dilation) must be set in Velocity material properties in UE5.

        Returns:

        '''

        target_simulation_time_step = (1 / target_fps) * (
                    1 / time_dilation)  # how much time has to pass in slowed down game engine to hit desired fps
        # (in seconds)
        target_simulation_time_step_usec = target_simulation_time_step * 1000000  # as above, but in micro seconds (10^-6)
        targeted_time_step_usec = (1 / target_fps) * 1000000 # how much time is passing in real time (in micro seconds)

        self.set_targeted_time_step_usec(targeted_time_step_usec)
        self.set_time_dilation(time_dilation)
        self.set_target_simulation_time_step(target_simulation_time_step)

        self.client.simSetGameSpeed(time_dilation)  # dont call simSetGameSpeed often and with zero values,
        # I have observed it can break wheels in UE5 CitySample demo

        if self.be_verbose:
            print("target_time_interval: ", target_simulation_time_step)

        if self.camera_paths_dir is None:
            camera_pathes = self.make_camera_demo_paths()
        else:
            camera_pathes = self.read_camera_paths_from_disk()

        self.save_random_movement_sample(camera_pathes=camera_pathes)
        # self.iterate_all_camera_patches(camera_pathes=camera_pathes)
        self.client.simSetGameSpeed(1.0)

