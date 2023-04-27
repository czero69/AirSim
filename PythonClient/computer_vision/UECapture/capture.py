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
from threading import Thread


class Capture:
    def __init__(self, verbose=False, camera_paths=None, save_dir=None, generate_camera_sample_dir=None):
        # create airsim client and establish a connection
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        # self.tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_drone")
        self.tmp_dir = 'D:/Kamil/data_collected/airsim_drone/'
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

        #@todo as Label is not accessible in the packaged builds, move everything to unreal python script
        # see AirSim\PythonClient\computer_vision\UECapture\set_stencils_from_editor.py how to run it from editor
        # in editor, load level chunks from worldpartition and run for each chunk (as big as your RAM)

        #### first, set things by mesh->owner() ####
        # set all meshes which owner name is according to regexes
        self.client.simSetMeshNamingMethodByString('owner')

        # setting everything to black (unlabeled) first
        # found = self.client.simSetSegmentationObjectID("[\w]*", 0, True) # unlabelled
        # all far away buildings
        # found = self.client.simSetSegmentationObjectID("[\w]*HLOD[\w]*", 18, True) # building
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

        #### below set all mesh via owner->getActorLabel() ##########
        # these are now set up in editor before map is packaged, see set_stencils_from_editor.py
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

        #### Finally, set smaller granuality stencils by staticMesh ##########
        self.client.simSetMeshNamingMethodByString('mesh')
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

    def get_pose_from_camera_path(self, elapsed_simulation_time_msec, camera_path, camera_pathes, camera_time_list):
        position_val, orientation_val = self.interpolate_position_and_rotation(elapsed_simulation_time_msec,
                                                                               camera_path,
                                                                               camera_pathes[
                                                                                   'rotation_notation'],
                                                                               camera_time_list)
        position_val, orientation_val = self.toNED(position_val, orientation_val)
        position_val.z_val = position_val.z_val + 5.0  # @todo shift initial pos, orientat; add init pose support
        pose = Pose(position_val=position_val, orientation_val=orientation_val)
        return pose

    def move_camera(self, move_by_time_usec, elapsed_prev_simulation_time_usec, time_dilation, camera_path,
                    camera_pathes, camera_time_list, sampling_rate=1.0):
        '''
        moving camera by specified amount of time
        Args:
            move_by_time_usec:
            elapsed_prev_simulation_time_usec: time that has passed previously
            sampling_rate: how often do you want to 'jump' the camera around per real render time (fps in ue5).

        Returns:

        '''

        total_time_passed_usec = 0
        cunt = 0
        recent_u5_delta_us = self.client.simGetDeltaTime() * 1000000
        current_delta_us = recent_u5_delta_us / sampling_rate
        while(total_time_passed_usec < move_by_time_usec - current_delta_us):
            cunt += 1
            recent_u5_delta_us = self.client.simGetDeltaTime() * 1000000 # * (1 / time_dilation) # (1 / time_dilation)
            # recent_u5_delta_us_simulation = recent_u5_delta_us * (1 / time_dilation)
            current_delta_us = recent_u5_delta_us / sampling_rate
            total_time_passed_usec += current_delta_us
            elapsed_simulation_time_usec_now = elapsed_prev_simulation_time_usec + total_time_passed_usec
            if self.be_verbose:
                print("current_delta_us: ", current_delta_us)
                print("elapsed_simulation_time_usec_now:", elapsed_simulation_time_usec_now)
            pose_now = self.get_pose_from_camera_path(elapsed_simulation_time_usec_now, camera_path, camera_pathes,
                                                      camera_time_list)
            time.sleep(current_delta_us / 1000000)  # iterate simulation
            self.client.simSetVehiclePose(pose_now, True)

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
            total_steps = camera_time_list[-1] // targeted_time_step_usec
            ### estimated_safe_sleep_time = 2  # in secs # @todo SafeSleepTime
            for step_id in tqdm(range(int(total_steps))):
                # create a thread
                elapsed_simulation_time_usec_now = int(targeted_time_step_usec * step_id)
                #thread = Thread(target=self.move_camera, args=(targeted_time_step_usec,
                #                                               int(targeted_time_step_usec * (step_id - 1)), time_dilation
                #                                               , camera_path, camera_pathes, camera_time_list))
                last_delta_usecs = self.client.simGetDeltaTime() * (1/time_dilation) * 1000000
                elapsed_simulation_time_usec_prev_step = int(elapsed_simulation_time_usec_now - \
                                                             32 * (np.sqrt(last_delta_usecs * time_dilation)))  # time_dilation
                #elapsed_simulation_time_usec_prev_step = int(elapsed_simulation_time_usec_now - \
                  #                                        1.0 * last_delta_usecs * time_dilation) # time_dilation
                #elapsed_simulation_time_usec_prev_step = int(elapsed_simulation_time_usec_now - \
                  #                                        1.0 * targeted_time_step_usec) # time_dilation
                #if self.be_verbose:
                #    print("last_delta_usecs: ", last_delta_usecs)
                #    print("elapsed time now/prev:", elapsed_simulation_time_usec_now, elapsed_simulation_time_usec_prev_step)
                pose_now = self.get_pose_from_camera_path(elapsed_simulation_time_usec_now, camera_path, camera_pathes,
                                               camera_time_list)
                # so the jump from pose_prev to pose_now takes into consideration how much did we slowed things down
                # by 'time_dilation'
                pose_prev = self.get_pose_from_camera_path(elapsed_simulation_time_usec_prev_step, camera_path, camera_pathes,
                                               camera_time_list)
                if self.be_verbose:
                    self.getPoseAndRotation()

                # self.client.simSetGameSpeed(time_dilation) # @todo TEST NEW
                ### time.sleep(1)  # @todo TEST NEW
                time_before_pause = time.time()
                self.client.simPause(False)
                ### self.client.simSetVehiclePose(pose_prev, True) # pose_prev # @todo TEST NEW
                self.client.simSetVehiclePose(pose_now, True)
                ## thread.start()
                time.sleep(target_simulation_time_step)  # @todo TEST NEW iterate simulation
                ###if (estimated_safe_sleep_time > 0):
                ###    time.sleep(estimated_safe_sleep_time) # sleep some amount of time,
                                                              # so the light rebuilds after camera movement, but less than
                                                              # target_simulation_time_step, so in case now elapsed time is shorter,
                                                              # we still have some time left
                ## thread.join()
                ### self.client.simSetVehiclePose(pose_now, True) # @todo TEST NEW
                # get velocity buffers before calling simPause. SimPause will erase velocity buffer.
                '''
                responses_velocity = self.client.simGetImages([
                    airsim.ImageRequest("0", airsim.ImageType.Velocity, pixels_as_float=False, pixels_as_float_RGB=True,
                                        compress=False)]) # @todo CHECK
                end = time.time()
                print("time responses_velocity ms:", (end - start) * 1000)
                '''
                #self.client.simSetGameSpeed(0.0) # @todo TEST NEW
                ### time.sleep(1.5) # @todo TEST NEW
                self.client.simPause(True) # @todo NO PAUSE? NEW
                # time.sleep(2)  # @todo, does it help here (?)

                '''
                save_screenshot_dir = os.path.join(self.tmp_dir, path_name, "screenshot")
                if not os.path.exists(save_screenshot_dir):
                    os.makedirs(save_screenshot_dir)
                # print(os.path.join(save_screenshot_dir.replace(os.sep, '/'), str(step_id).zfill(5) + '.png'))
                success = self.client.simTakeScreenshot(
                    os.path.join(save_screenshot_dir.replace(os.sep, '/'), str(step_id).zfill(5) + '.png'))
                if self.be_verbose:
                    print("image taken")
                if self.be_verbose:
                    print("taking images")
                '''

                #responses_velocity = self.client.simGetImages([
                #    airsim.ImageRequest("0", airsim.ImageType.Velocity, pixels_as_float=False, pixels_as_float_RGB=True,
                #                        compress=False)])  # @todo CHECK

                '''
                time_after_pause = time.time()
                last_sleep_time_sec = time_after_pause - time_before_pause
                remaining_sleep_time_sec = target_simulation_time_step - last_sleep_time_sec
                estimated_safe_sleep_time = (remaining_sleep_time_sec + estimated_safe_sleep_time)*0.8
                if(remaining_sleep_time_sec > 0):
                    time.sleep(remaining_sleep_time_sec)
                else:
                    print("taking buffers takes longer time needed to run demo for desired FPS. "
                          "Consider slowing the UE5 game speed even further. "
                          "Please set time_dilation parameter for smaller value.")
                    print("time target_simulation_time_step True ms:", (target_simulation_time_step) * 1000)
                    print("time last_sleep_time_sec ms:", (last_sleep_time_sec) * 1000)
                    print("time remaining_sleep_time_sec True ms:", (remaining_sleep_time_sec) * 1000)
                    print("time estimated_safe_sleep_time True ms:", (estimated_safe_sleep_time) * 1000)
                self.client.simPause(True)
                '''

                responses = self.client.simGetImages([
                    ### airsim.ImageRequest("0", airsim.ImageType.Scene), # this is replaced by screenshots. Only for preview.
                    airsim.ImageRequest("0", airsim.ImageType.Velocity, pixels_as_float=False, pixels_as_float_RGB=True, compress=False), # velocity buffer is captured earliel, as at this point it is erased because of client.simPause(True) @todo TEST NEW
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
                self.client.simPause(True)

                save_screenshot_dir = os.path.join(self.tmp_dir, path_name, "screenshot")
                if not os.path.exists(save_screenshot_dir):
                    os.makedirs(save_screenshot_dir)
                # print(os.path.join(save_screenshot_dir.replace(os.sep, '/'), str(step_id).zfill(5) + '.png'))
                success = self.client.simTakeScreenshot(
                    os.path.join(save_screenshot_dir.replace(os.sep, '/'), str(step_id).zfill(5) + '.png'))
                if self.be_verbose:
                    print("image taken")
                if self.be_verbose:
                    print("taking images")

                ### responses.extend(responses_velocity) # @todo TEST NEW

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
                        image_arr = np.flip(image_arr, axis=0) # need for depth buffer
                        airsim.write_pfm(os.path.normpath(os.path.join(save_dir, str(step_id).zfill(5) + '.pfm')),
                                         image_arr)
                    elif response.pixels_as_float_RGB:
                        # so far this data type is only used for velocity buffer
                        # Change images into numpy arrays.
                        img1d = np.array(response.image_data_float_RGB, dtype=np.float)
                        # img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                        im = img1d.reshape(response.height, response.width, 3)
                        im_no_alpha = im[:, :, :3]
                        # standarize to selected fps number instead of what fps game engine is rendering
                        # current velocity buffer matches last_delta_usecs (what fps ue has rendered)
                        # we are targeting targeted_time_step_usec (1 / target_fps) * 10^6
                        # im_no_alpha --> last_delta_usecs

                        im_no_alpha -= 0.5
                        im_no_alpha /= 0.5
                        im_no_alpha = im_no_alpha * np.sqrt(targeted_time_step_usec / (last_delta_usecs * time_dilation)) # # @todo time_dilation added / not certain is it correct
                        # im_no_alpha *= 3 # time_dilation=0.00167 intead of 0.01 @todo remove this calc from UE shader
                        # im_no_alpha = im_no_alpha * (last_delta_usecs * time_dilation / (targeted_time_step_usec))
                        # im_no_alpha *= 1.0 # 20 # empower signal so it looks good in RGB range
                        im_no_alpha *= 0.5
                        im_no_alpha += 0.5



                        filename = os.path.normpath(os.path.join(save_dir, str(step_id).zfill(5) + '.npy'))
                        ##### np.save(filename, im_no_alpha) #@todo unlock it later with switch

                        # velocity needs to support -2..2 screen space range for x and y
                        # for visualization purposes:
                        '''
                        im_no_alpha -= 0.5
                        im_no_alpha /= 0.5
                        im_no_alpha *= 1.0
                        im_no_alpha *= 0.5
                        im_no_alpha += 0.5
                        '''

                        im_no_alpha = im_no_alpha*255
                        im_no_alpha = np.clip(im_no_alpha, 0, 255)
                        im_no_alpha = im_no_alpha.astype('uint8')
                        cv2.imwrite(os.path.normpath(os.path.join(save_dir, str(step_id).zfill(5) + '_visualization.png')),
                                   im_no_alpha)
                    else:
                        # if self.be_verbose:
                        #    print("Type %d, size %d, pos %s" % (
                        #        response.image_type, len(response.image_data_uint8), pprint.pformat(response.camera_position)))

                        # for compressed
                        #airsim.write_file(os.path.normpath(os.path.join(save_dir, str(step_id).zfill(5) + '.png')),
                        #                  response.image_data_uint8)


                        # Change images into numpy arrays.
                        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
                        # img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                        im = img1d.reshape(response.height, response.width, 3)
                        im_no_alpha = im[:,:,:3]
                        cv2.imwrite(os.path.normpath(os.path.join(save_dir, str(step_id).zfill(5) + '.png')), im_no_alpha)

                '''
                save_screenshot_dir = os.path.join(self.tmp_dir, path_name, "screenshot")
                if not os.path.exists(save_screenshot_dir):
                    os.makedirs(save_screenshot_dir)
                print(os.path.join(save_screenshot_dir.replace(os.sep, '/'), str(step_id).zfill(5) + '.png'))
                success = self.client.simTakeScreenshot(
                    os.path.join(save_screenshot_dir.replace(os.sep, '/'), str(step_id).zfill(5) + '.png'))
                if self.be_verbose:
                    print("image taken")
                '''


        self.client.simPause(False)  # leave demo unpaused
