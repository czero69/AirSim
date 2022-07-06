import airsim
import numpy as np
import time
import os
import tempfile
import pprint

# create airsim client and establish a connection
client = airsim.MultirotorClient()
client.confirmConnection()

tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_drone")
print ("Saving images to %s" % tmp_dir)
try:
    for n in range(3):
        os.makedirs(os.path.join(tmp_dir, str(n)))
except OSError:
    if not os.path.isdir(tmp_dir):
        raise

pose = client.simGetVehiclePose()
print("x={}, y={}, z={}".format(pose.position.x_val, pose.position.y_val, pose.position.z_val))
angles = airsim.to_eularian_angles(client.simGetVehiclePose().orientation)
print("pitch={}, roll={}, yaw={}".format(angles[0], angles[1], angles[2]))

# you can actually get all scene objects like that:
# data = client.simListSceneObjects(".*")
# print(data)

airsim.wait_key('Press any key to set all object IDs to 0')
found = client.simSetSegmentationObjectID("[\w]*", 0, True);
print("Done: %r" % (found))

airsim.wait_key('Press any key to change humans')
found = client.simSetSegmentationObjectID("[\w]*MASS[\w]*", 17, True) # small humans and drivers
#found = client.simSetSegmentationObjectID("[\w]*MASS[\w]*", 99, True)
found = client.simSetSegmentationObjectID("[\w]*CROWD[\w]*", 99, True)
found = client.simSetSegmentationObjectID("[\w]*human[\w]*", 99, True)
found = client.simSetSegmentationObjectID("[\w]*person[\w]*", 99, True)
found = client.simSetSegmentationObjectID("[\w]*pedestrian[\w]*", 99, True)
print("Done: %r" % (found))

#set object ID for cars and trucks
airsim.wait_key('Press any key to change vehicles')
found = client.simSetSegmentationObjectID("BP_MassTraffic[\w]*", 112, True) # BP_MassTraffic
found = client.simSetSegmentationObjectID("[\w]*_vehicle[\w]*", 112, True)
found = client.simSetSegmentationObjectID("[\w]*veh[\w]*", 112, True)
found = client.simSetSegmentationObjectID("[\w]*Van[\w]*", 14, True)
found = client.simSetSegmentationObjectID("[\w]*Car[\w]*", 112, True)
found = client.simSetSegmentationObjectID("[\w]*Truck[\w]*", 112, True)
#found = client.simSetSegmentationObjectID("BP_vehTruck_vehicle[\w]*", 14, True)
#found = client.simSetSegmentationObjectID("[\w]*Bike[\w]*", 2, True)
#found = client.simSetSegmentationObjectID("[\w]*BP_MantleBased[\w]*", 3, True)
found = client.simSetSegmentationObjectID("[\w]*Traffic[\w]*", 112, True)
#found = client.simSetSegmentationObjectID("[\w]*TRAFF[\w]*", 16, True)
#found = client.simSetSegmentationObjectID("[\w]*VEHICLE[\w]*", 18, True)
#found = client.simSetSegmentationObjectID("[\w]*CAR[\w]*", 19, True)
#found = client.simSetSegmentationObjectID("[\w]*Vehicles[\w]*", 20, True)
#found = client.simSetSegmentationObjectID("[\w]*BPC_SF_data[\w]*", 10, True)
#found = client.simSetSegmentationObjectID("[\w]*CitySample[\w]*", 12, True)
#found = client.simSetSegmentationObjectID("[\w]*Underpass[\w]*", 13, True)
print("Done: %r" % (found))

#airsim.wait_key('Press any key to change selected vehicle')
# found = client.simSetSegmentationObjectID("[\w]*MASS[\w]*", 17, True)
#found = client.simSetSegmentationObjectID("BP_vehCar_vehicle03_Sandbox_C_2147479347", 99, True)
#print("Done: %r" % (found))


#airsim.wait_key('Press any key to change one ground object ID')
#found = client.simSetSegmentationObjectID("Ground", 20);
#print("Done: %r" % (found))

#airsim.wait_key('Press any key to change one ground object ID')
#found = client.simSetSegmentationObjectID("ground[\w]*", 21, True);
#print("Done: %r" % (found))

#set object ID for sky
#airsim.wait_key('Press any key to change Sky ID')
#found = client.simSetSegmentationObjectID("SM_dome", 42, True);
#print("Done: %r" % (found))

#set object ID for street furniture
#airsim.wait_key('Press any key to change street furniture ID')
#found = client.simSetSegmentationObjectID("STREET_FURNITURE_[\w]*", 2, True);
#found = client.simSetSegmentationObjectID("BPP[\w]*", 2, True);
#found = client.simSetSegmentationObjectID("SM_Billboard[\w]*", 2, True);
#print("Done: %r" % (found))

#set object ID for trees
#airsim.wait_key('Press any key to change trees')
#found = client.simSetSegmentationObjectID("STREET_FURNITURE_D_N305", 1, True);
#print("Done: %r" % (found))

#set object ID for lamps
#airsim.wait_key('Press any key to change lamps')
#found = client.simSetSegmentationObjectID("STREET_FURNITURE_N123523", 3, True);
#found = client.simSetSegmentationObjectID("STREET_FURNITURE_N123460", 3, True);
#print("Done: %r" % (found))

#set object ID for buildings
#airsim.wait_key('Press any key to change buildings')
#found = client.simSetSegmentationObjectID("BPP_Bldg[\w]*", 4, True);
#found = client.simSetSegmentationObjectID("Okkata[\w]*", 4, True);
#found = client.simSetSegmentationObjectID("BLDG_*", 4, True);
#found = client.simSetSegmentationObjectID("SM_BLDG*", 4, True);
#print("Done: %r" % (found))


#set object ID for raffic lights
#airsim.wait_key('Press any key to change traffic lights')
#found = client.simSetSegmentationObjectID("TRAFFIC_LIGHT[\w]*", 6, True);
#print("Done: %r" % (found))

#pose.position.x_val = -721.720
#pose.position.y_val  = -387.960
#pose.position.z_val = 6.000
#client.simSetVehiclePose(pose, True)

dilation = 0.01
client.simSetGameSpeed(dilation)
client.simPause(False)

for x in range(300):
    # progress by one sec
    # client.simPause(False)
    #client.simSetGameSpeed(dilation)

    # this time does influence how much simulation has passed
    time.sleep(1.6666666)
    # grab prev-frame velocity buffer
    pose = client.simGetVehiclePose()
    #pose = client.simGetObjectPose('BP_ComputerVisionPawn')
    for _ in range(1):
        pose.position.y_val = pose.position.y_val + 0.01
        client.simSetVehiclePose(pose, True)
        # x, y, z = pose.position.x_val, pose.position.y_val, pose.position.z_val

    # stop all game objects speed completely, but not entire simulation
    # this will allow for lumen, nanite and other buffers to built in time (better screenshot image quality)
    # while at the same time stopping all objects
    responses_velocity = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Velocity)])
    # time.sleep(1.0)
    # dont call speed stop immidiately after teleport
    #client.simSetGameSpeed(0.0)
    # clear image and take snapshot


    #client.moveToPositionAsync(x,y,z, 20.0).join()
    #client.simSetObjectPose('BP_ComputerVisionPawn', pose, teleport=False)
    # time.sleep(0)  # run
    # client.simPause(True)
    # client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0)), True)
    client.simPause(True)
    time.sleep(1.0)
    # call it as early as possible, before velocity buffer gets erased (hack)
    #responses_velocity = client.simGetImages([
        #airsim.ImageRequest("0", airsim.ImageType.Velocity)])

    print("taking images")
    # time.sleep(0.5)  # run
    # take images
    responses = client.simGetImages([
        # airsim.ImageRequest("0", airsim.ImageType.Velocity), # velocity buffer is erased at this point
        airsim.ImageRequest("0", airsim.ImageType.SurfaceNormals),
        airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, pixels_as_float = True, compress = False),
        airsim.ImageRequest("0", airsim.ImageType.Segmentation),
        airsim.ImageRequest("0", airsim.ImageType.ShaderID), # probably very limited
        # airsim.ImageRequest("0", airsim.ImageType.TextureUV), # not yet supported
        airsim.ImageRequest("0", airsim.ImageType.SceneDepth, pixels_as_float = True, compress = False),
        airsim.ImageRequest("0", airsim.ImageType.ReflectionVector),
        airsim.ImageRequest("0", airsim.ImageType.DotSurfaceReflection),
        airsim.ImageRequest("0", airsim.ImageType.Metallic),
        airsim.ImageRequest("0", airsim.ImageType.Albedo),
        airsim.ImageRequest("0", airsim.ImageType.Specular),
        airsim.ImageRequest("0", airsim.ImageType.Opacity),
        airsim.ImageRequest("0", airsim.ImageType.Roughness),
        #airsim.ImageRequest("0", airsim.ImageType.Anisotropy),
        #airsim.ImageRequest("0", airsim.ImageType.AO),
        airsim.ImageRequest("0", airsim.ImageType.ShadingModelColor)])

    responses.extend(responses_velocity)

    for i, response in enumerate(responses):
        if not os.path.exists(os.path.join(tmp_dir, str(i))):
            os.makedirs(os.path.join(tmp_dir, str(i)))
        if response.pixels_as_float:
            print("Type %d, size %d, pos %s" % (
            response.image_type, len(response.image_data_float), pprint.pformat(response.camera_position)))
            airsim.write_pfm(os.path.normpath(os.path.join(tmp_dir, str(i), str(x) + "_" + str(i) + '.pfm')),
                             airsim.get_pfm_array(response))
        else:
            print("Type %d, size %d, pos %s" % (
            response.image_type, len(response.image_data_uint8), pprint.pformat(response.camera_position)))
            airsim.write_file(os.path.normpath(os.path.join(tmp_dir, str(i), str(x) + "_" + str(i) + '.png')),
                              response.image_data_uint8)

    my_dir = r"C:/Users/Wojciech/Desktop/imgui"
    success = client.simTakeScreenshot(os.path.join(my_dir, str(x) + '.png'))
    client.simPause(False)
    print("image taken")
    # client.simPause(False)

client.simPause(False)
