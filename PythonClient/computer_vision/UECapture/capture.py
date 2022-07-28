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


class Capture:
    def __init__(self, verbose=False, camera_paths=None, save_dir=None, generate_camera_sample_dir=None):
        # create airsim client and establish a connection
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_drone")
        self.be_verbose = verbose
        self.save_dir = save_dir
        self.camera_paths_dir = camera_paths
        self.generate_camera_sample_dir = generate_camera_sample_dir

    def getPoseAndRotation(self):
        pose = self.client.simGetVehiclePose()
        angles = airsim.to_eularian_angles(self.client.simGetVehiclePose().orientation)
        if self.be_verbose:
            print("x={}, y={}, z={}".format(pose.position.x_val, pose.position.y_val, pose.position.z_val))
            print("pitch={}, roll={}, yaw={}".format(angles[0], angles[1], angles[2]))
        return pose, angles

    def setSegmentationIDs(self):
        # set Stencil Masks ids

        # setting all to black first
        found = self.client.simSetSegmentationObjectID("[\w]*", 0, True);

        # set humans (including drivers, and far away humans)
        found = self.client.simSetSegmentationObjectID("[\w]*MASS[\w]*", 17, True)  # small humans and drivers
        found = self.client.simSetSegmentationObjectID("[\w]*CROWD[\w]*", 99, True)
        found = self.client.simSetSegmentationObjectID("[\w]*human[\w]*", 99, True)
        found = self.client.simSetSegmentationObjectID("[\w]*person[\w]*", 99, True)
        found = self.client.simSetSegmentationObjectID("[\w]*pedestrian[\w]*", 99, True)

        # set vehicles
        found = self.client.simSetSegmentationObjectID("BP_MassTraffic[\w]*", 112, True)  # BP_MassTraffic
        found = self.client.simSetSegmentationObjectID("[\w]*veh[\w]*", 112, True)
        found = self.client.simSetSegmentationObjectID("[\w]*Van[\w]*", 14, True)
        found = self.client.simSetSegmentationObjectID("[\w]*Car[\w]*", 112, True)
        found = self.client.simSetSegmentationObjectID("[\w]*Truck[\w]*", 112, True)
        found = self.client.simSetSegmentationObjectID("[\w]*Traffic[\w]*", 112, True)

        # other things are not supported in stencil buffer due to nanite does not support
        # stencil buffer writing (as for UE 5.0.2)
        # meta humans works, and for every car I have placed manually low-level no-nanite mesh to be able
        # to write to stencil

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

        print("left, right: ", left, right)
        print("left_val, right_val: ", camera_patch[left][-1], camera_patch[right][-1])
        print("target: ", passed_simulation_time_usec)

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

        print("R1, R2", R1, R2)

        t1 = coords_prev[-1]
        t2 = coords_next[-1]
        t_out = passed_simulation_time_usec

        position_val_prev = Vector3r(x_val=coords_prev[0], y_val=coords_prev[1], z_val=coords_prev[2])
        position_val_next = Vector3r(x_val=coords_next[0], y_val=coords_next[1], z_val=coords_next[2])
        position_interpolated = (position_val_prev + position_val_next) / 2

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
        real_time_step_usec = (1 / target_fps) * 1000000 # how much time is passing in real time (in micro seconds)

        self.client.simSetGameSpeed(time_dilation)  # dont call simSetGameSpeed often and with zero values,
        # I have observed it can break wheels in UE5 CitySample demo

        if self.be_verbose:
            print("target_time_interval: ", target_simulation_time_step)

        if self.camera_paths_dir is None:
            camera_pathes = self.make_camera_demo_paths()
        else:
            camera_pathes = self.read_camera_paths_from_disk()

        # @todo add camera_patches supprot - for now, simple demo
        for path_name, camera_path in camera_pathes['patches'].items():
            # self.client.simPause(False)  # after moving to new location, rebuild Lumen and others for a longer time
            # time.sleep(7.0)
            self.client.simPause(True)
            camera_time_list = []  # msecs time lists from recorded camera
            for step_id, coords in enumerate(camera_path):
                elapsed_camera_paths_time_msec = coords[-1]
                camera_time_list.append(elapsed_camera_paths_time_msec)
            total_steps = camera_time_list[-1] // real_time_step_usec
            for step_id in tqdm(range(int(total_steps))):
                elapsed_simulation_time_msec = int(real_time_step_usec * step_id)
                position_val, orientation_val = self.interpolate_position_and_rotation(elapsed_simulation_time_msec,
                                                                                       camera_path,
                                                                                       camera_pathes[
                                                                                           'rotation_notation'],
                                                                                       camera_time_list)
                position_val, orientation_val = self.toNED(position_val, orientation_val)
                position_val.z_val = position_val.z_val + 5.0  # @todo shift initial pos, orientat
                pose = Pose(position_val=position_val, orientation_val=orientation_val)
                if self.be_verbose:
                    self.getPoseAndRotation()
                self.client.simPause(False)
                time.sleep(target_simulation_time_step)  # iterate simulation
                # self.client.simSetVehiclePose(pose, True)
                self.client.simSetVehiclePose(pose, True)

                # get velocity buffers before calling simPause. SimPause will erase velocity buffer.
                responses_velocity = self.client.simGetImages([
                    airsim.ImageRequest("0", airsim.ImageType.Velocity, pixels_as_float = False, compress = False)])

                self.client.simPause(True)
                time.sleep(0.1)  # @todo, does it help here (?)
                if self.be_verbose:
                    print("taking images")

                responses = self.client.simGetImages([
                    ### airsim.ImageRequest("0", airsim.ImageType.Scene), # this is replaced by screenshots. Only for preview.
                    ### airsim.ImageRequest("0", airsim.ImageType.Velocity), # velocity buffer is captured earliel, as at this point it is erased because of client.simPause(True)
                    airsim.ImageRequest("0", airsim.ImageType.SurfaceNormals, pixels_as_float = False, compress = False),
                    airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=False),
                    airsim.ImageRequest("0", airsim.ImageType.Segmentation, pixels_as_float = False, compress = False),
                    airsim.ImageRequest("0", airsim.ImageType.ShaderID, pixels_as_float = False, compress = False),
                    ### probably very limited @todo, decide do we want it
                    ### airsim.ImageRequest("0", airsim.ImageType.TextureUV), # not yet supported
                    ### airsim.ImageRequest("0", airsim.ImageType.SceneDepth, pixels_as_float = True, compress = False), # same as airsim DepthPlanar, but depthplannar is scaled to 0..1
                    airsim.ImageRequest("0", airsim.ImageType.ReflectionVector, pixels_as_float = False, compress = False),
                    airsim.ImageRequest("0", airsim.ImageType.DotSurfaceReflection, pixels_as_float = False, compress = False),
                    airsim.ImageRequest("0", airsim.ImageType.Metallic, pixels_as_float = False, compress = False),
                    airsim.ImageRequest("0", airsim.ImageType.Albedo, pixels_as_float = False, compress = False),
                    airsim.ImageRequest("0", airsim.ImageType.Specular, pixels_as_float = False, compress = False),
                    airsim.ImageRequest("0", airsim.ImageType.Opacity, pixels_as_float = False, compress = False),
                    airsim.ImageRequest("0", airsim.ImageType.Roughness, pixels_as_float = False, compress = False),
                    ### airsim.ImageRequest("0", airsim.ImageType.Anisotropy), # no valuable info observed
                    ### airsim.ImageRequest("0", airsim.ImageType.AO), # no valuable info observed
                    airsim.ImageRequest("0", airsim.ImageType.ShadingModelColor, pixels_as_float = False, compress = False)])  # probably very limited @todo, decide do we want it

                # add velocity buffer captured earlier
                responses.extend(responses_velocity)

                for response in responses:
                    save_dir = os.path.join(self.tmp_dir, path_name, str(response.image_type))
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    if response.pixels_as_float:
                        # if self.be_verbose:
                        #    print("Type %d, size %d, pos %s" % (
                        #        response.image_type, len(response.image_data_float), pprint.pformat(response.camera_position)))
                        # for some reason it is flipped vertically. Reflip it.
                        image_arr = airsim.get_pfm_array(response)
                        image_arr = np.flip(image_arr, axis=0)
                        airsim.write_pfm(os.path.normpath(os.path.join(save_dir, str(step_id).zfill(5) + '.pfm')),
                                         image_arr)
                        '''
                            # Change images into numpy arrays.
                            print("Velocity buffer now!")
                            img1d = np.array(response.image_data_float, dtype=np.float)
                            print("shape: ", img1d.shape)
                            # img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                            im = img1d.reshape(response.height, response.width, 3)
                            im_no_alpha = im[:, :, :3]
                            filename = os.path.normpath(os.path.join(save_dir, str(step_id).zfill(5) + '.npy'))
                            np.save(filename, im_no_alpha)
                            #cv2.imwrite(os.path.normpath(os.path.join(save_dir, str(step_id).zfill(5) + '.npz')),
                             #           im_no_alpha)
                        '''
                    else:
                        # if self.be_verbose:
                        #    print("Type %d, size %d, pos %s" % (
                        #        response.image_type, len(response.image_data_uint8), pprint.pformat(response.camera_position)))

                        # some glitch with colours - try opencv imwrite instead
                        #airsim.write_file(os.path.normpath(os.path.join(save_dir, str(step_id).zfill(5) + '.png')),
                        #                  response.image_data_uint8)

                        # Change images into numpy arrays.
                        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
                        # img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                        im = img1d.reshape(response.height, response.width, 3)
                        im_no_alpha = im[:,:,:3]
                        cv2.imwrite(os.path.normpath(os.path.join(save_dir, str(step_id).zfill(5) + '.png')), im_no_alpha)

                save_screenshot_dir = os.path.join(self.tmp_dir, path_name, "screenshot")
                if not os.path.exists(save_screenshot_dir):
                    os.makedirs(save_screenshot_dir)
                print(os.path.join(save_screenshot_dir.replace(os.sep, '/'), str(step_id).zfill(5) + '.png'))
                success = self.client.simTakeScreenshot(
                    os.path.join(save_screenshot_dir.replace(os.sep, '/'), str(step_id).zfill(5) + '.png'))
                if self.be_verbose:
                    print("image taken")

        self.client.simPause(False)  # leave demo unpaused
