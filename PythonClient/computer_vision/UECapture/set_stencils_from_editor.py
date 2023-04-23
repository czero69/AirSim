import unreal
from unreal import LevelEditorSubsystem

# see below to set your local .py editor (e.g. Pycharm) for autocomplete
# https://docs.unrealengine.com/5.1/Images/setting-up-your-production-pipeline/scripting-and-automating-the-editor/Python/Autocomplete/python-stub-path.webp

# Get a reference to the editor subsystem

# Define the list of regexes, stencil values, and match types
# "match_type" on of the {"actor_name", "actor_label", "mesh_name", "mesh_static"}
'''
actor_name - dynamic in-game generated name (often generic/templated)
actor_label - actor label name accessible only in-editor (that's why better to use this script in-editor than using runtime airsim methods to set stencils)
mesh_name - in editor mesh name (like label) (this is also not accessible in airsim set stencil method)
mesh_static - mesh name same as static mesh filename (this is also what airsim set stencil method see in games as staticMesh)
'''

# the below list is a priority list (order matters). From lowest to highest prior. 
stencil_list = [
    {"regex": "*", "stencil_value": 255, "match_type": "actor_name"},  # unlabelled

    {"regex": "*HLOD*", "stencil_value": 18, "match_type": "actor_name"},  # building

    {"regex": "*MASS*", "stencil_value": 17, "match_type": "actor_name"},  # person
    {"regex": "*CROWD*", "stencil_value": 17, "match_type": "actor_name"},  # person
    {"regex": "*human*", "stencil_value": 17, "match_type": "actor_name"},  # person
    {"regex": "*person*", "stencil_value": 17, "match_type": "actor_name"},  # person
    {"regex": "*pedestrian*", "stencil_value": 17, "match_type": "actor_name"},  # person

    {"regex": "*BP_MassTraffic*", "stencil_value": 42, "match_type": "actor_name"},  # vehicle
    {"regex": "*veh*", "stencil_value": 42, "match_type": "actor_name"},  # vehicle
    {"regex": "*Van*", "stencil_value": 42, "match_type": "actor_name"},  # vehicle
    {"regex": "*Car*", "stencil_value": 42, "match_type": "actor_name"},  # vehicle
    {"regex": "*Truck*", "stencil_value": 42, "match_type": "actor_name"},  # vehicle
    {"regex": "*Traffic*", "stencil_value": 42, "match_type": "actor_name"},  # vehicle


    {"regex": "*Street_Furniture*", "stencil_value": 2, "match_type": "actor_label"},   # street furniture
    {"regex": "*Sidewalk*", "stencil_value": 2, "match_type": "actor_label"},   # street furniture
    {"regex": "*StartingArea*", "stencil_value": 2, "match_type": "actor_label"},   # street furniture
    {"regex": "*Plaza*", "stencil_value": 2, "match_type": "actor_label"},   # street furniture

    {"regex": "*Bldg*", "stencil_value": 18, "match_type": "actor_label"},   # building
    {"regex": "*_bld_*", "stencil_value": 18, "match_type": "actor_label"},   # building
    {"regex": "*Roof*", "stencil_value": 18, "match_type": "actor_label"},   # building

    {"regex": "*Decal*", "stencil_value": 7, "match_type": "actor_label"},   # road
    {"regex": "*Ground*", "stencil_value": 7, "match_type": "actor_label"},   # road
    {"regex": "*Road*", "stencil_value": 7, "match_type": "actor_label"},   # road
    {"regex": "*UnderPass*", "stencil_value": 7, "match_type": "actor_label"},   # road

    {"regex": "*freeway*", "stencil_value": 9, "match_type": "actor_label"},   # infrastructure

    {"regex": "*Billboard*", "stencil_value": 37, "match_type": "actor_label"},   # billboard and traffic signs

    {"regex": "*water_plane*", "stencil_value": 28, "match_type": "actor_label"},   # water


    {"regex": "*bench*", "stencil_value": 2, "match_type": "mesh_static"},   # street furniture
    {"regex": "*stairs*", "stencil_value": 2, "match_type": "mesh_static"},   # street furniture
    {"regex": "*lamp*", "stencil_value": 2, "match_type": "mesh_static"},   # street furniture

    {"regex": "*Road*", "stencil_value": 7, "match_type": "mesh_static"},   # road
    {"regex": "*Curb*", "stencil_value": 7, "match_type": "mesh_static"},   # road

    {"regex": "*bldg_*", "stencil_value": 18, "match_type": "mesh_static"},   # building
    {"regex": "*roof_*", "stencil_value": 18, "match_type": "mesh_static"},   # building

    {"regex": "*Sign*", "stencil_value": 37, "match_type": "mesh_static"},   # billboard and traffic signs
    
    {"regex": "*signage*", "stencil_value": 9, "match_type": "mesh_static"},   # infrastructure

    {"regex": "*Light*", "stencil_value": 32, "match_type": "mesh_static"},   # lights / traffic lights

    {"regex": "*dirt_*", "stencil_value": 39, "match_type": "mesh_static"},   # terrain
    {"regex": "*rock_*", "stencil_value": 39, "match_type": "mesh_static"},   # terrain

    {"regex": "*vehicle*", "stencil_value": 42, "match_type": "mesh_static"},   # vehicle
    {"regex": "*wheel*", "stencil_value": 42, "match_type": "mesh_static"},   # vehicle

    {"regex": "*EuropeanHorn*", "stencil_value": 31, "match_type": "mesh_static"}, # vegetation, trees

    {"regex": "*SM_DOME*", "stencil_value": 35, "match_type": "mesh_static"} # sky
]

# Get a reference to the current level
les = unreal.get_editor_subsystem(LevelEditorSubsystem)
level = les.get_current_level()

# UESUBSYS = unreal.UnrealEditorSubsystem()
# this_world = unreal.UnrealEditorSubsystem.get_editor_world()
# this_world = UESUBSYS.get_editor_world()
# all_levels = unreal.EditorLevelUtils.get_levels(this_world)

# Get all actors in the current level
level_actors = unreal.EditorLevelLibrary.get_all_level_actors()

print("len: ", len(level_actors))

#pattern = "*europeantree*"
#is_match = unreal.StringLibrary.matches_wildcard("avc_EuroPeanTreE_sss", pattern)
#print("is_match: ", is_match)
# level = unreal.EditorLevelLibrary.get_editor_world().get_current_level()

# set the directory path
if True:
    directory_path = "/Game/Environment/Courtyard/Kit_courtyard"

    asset_registry = unreal.AssetRegistryHelpers.get_asset_registry()

    path_filter = unreal.ARFilter(package_paths=[directory_path])  # class_names=["Actor", "StaticMesh"]
    # path_filter.class_names = ["Actor", "StaticMesh"]
    # path_filter.set_editor_property("package_paths", [directory_path])

    # get a list of asset data objects that match the filter
    asset_data_list = asset_registry.get_assets(path_filter)
    print("len(asset_data_list):", len(asset_data_list))

    # loop through the asset data objects and print their names
    for asset_data in asset_data_list:
        asset_name = asset_data.asset_name
        print("asset_name: ", asset_data.asset_name)
        # print("asset_class_path: ", asset_data.asset_class_path.asset_name)
        print("package_name: ", asset_data.package_name)
        print("package_path: ", asset_data.package_path)
        asset_type = asset_data.asset_class_path.asset_name
        print("asset_type: ",  asset_type)
        if asset_type == "World":
            #3#? unreal.LevelEditorSubsystem.load_level(asset_data.package_name)
            unreal.EditorLevelLibrary.load_level(asset_data.package_name)
            break
        elif asset_type == "Blueprint":
            pass
            #blueprint = unreal.EditorAssetSubsystem.load_asset(asset_data.package_name)
            #unreal.EditorLevelLibrary.load_level(asset_data.package_name)
            #break
        else:
            print("not supported type")





for actor in level_actors:
    actor_name = actor.get_name()
    actor_label_name = actor.get_actor_label()

    # Iterate through all meshes in the actor
    ### if "SIDEWALK_BIOM" in actor_label_name:
    if True:
        #for component in actor.root_component.get_children_components(include_all_descendants=True):
        if actor.get_class().get_name().startswith("BPP_"):
        #if actor.actor_has_tag("Blueprint"):
            # Load the blueprint asset
            ## print(actor.get_class().get_name())
            # blueprint_asset = actor.get_class().get_name()
            asset_path = actor.get_path_name()
            #asset_path = actor.get_path_name().split(".", 1)[0]

            print("Asset path:", asset_path)
            print("actor_name:", actor_name)
            print("actor_label_name:", actor_label_name)


            sys_path = unreal.SystemLibrary.get_path_name(actor) # unreal.SystemLibrary.get_system_path(actor)
            print("sys_path: ", sys_path)

            #pathleves = unreal.LevelEditorSubsystem.get_path_name_for_loaded_asset(actor)
            #print("pathleves:", pathleves)


            print("ac lvvl pth:", actor.get_package()) # get_full_name
            print("ac22:", actor.get_full_name())

            #path_filter = unreal.ARFilter(package_paths=[directory_path])


            # print("obj path: ", actor.object_path)

            #level_path = actor.is_level_loaded()
            #print("lvl path:", level_path)



            ##unreal.EditorLevelLibrary.load_level(asset_path)
            ##print("loaded")


            #level = actor.get_editor_property("Level")
            #print("potential level name:", level)


            '''
            blueprint = unreal.EditorAssetLibrary.load_asset(asset_path)  # its probably not this
            # blueprint_asset = actor.get_editor_property()
            # blueprint = unreal.EditorAssetSubsystem.load_asset(asset_path)
            # blueprint = unreal.SystemLibrary.load_asset_blocking(asset_path)

            ############ unreal.LevelEditorSubsystem.load_level(asset_path)
            ############ unreal.EditorLevelLibrary.load_level(asset_path)


            #blueprint = unreal.EditorLevelLibrary.get_actor_reference()
            #unreal.EditorAssetLibrary.checkout_asset(blueprint)
            #unreal.AssetEditorSubsystem.open_editor_for_assets(blueprint)

            ### print("bp label:", blueprint.get_actor_label())
            ### print("actor label:", blueprint.get_actor_label())

            # unreal.EditorAssetLibrary.sync_browser_to_objects(asset_path)

            # D:/Kamil/CitySample_5.1/CitySample/Content/__ExternalActors__/Map/Big_City_LVL/9L/LN/XMICKK9TPOOQ9NWC6LY1U.uasset
            # unreal.EditorLevelLibrary.edit_object(blueprint)
            # Iterate over all components of the blueprint
            # D:/Kamil/CitySample_5.1/CitySample/Content/__ExternalActors__/Map/Big_City_LVL/9L/LN/XMICKK9TPOOQ9NWC6LY1U.uasset


            for component in blueprint.get_components_by_class(unreal.StaticMeshComponent):
            #for component in blueprint.root_component.get_children_components(include_all_descendants=True):
                mesh_name = component.get_name()

                
                ## _static = component.get_editor_property("static_mesh")
                ## if _static is not None:
                ##    mesh_static = _static.get_name()
                ##    print("mesh_static name: ", mesh_static)
                ##else:
                ##    mesh_static = ""
                
                print("mesh name: ", mesh_name)
                # Set the custom stencil value to 77
                prop = component.get_editor_property("custom_depth_stencil_value")
                print("prop befoer:", prop)

                component.set_editor_property("render_custom_depth", True) 
                component.set_editor_property("custom_depth_stencil_value", 99)
                prop = component.get_editor_property("custom_depth_stencil_value")
                print("prop afrer:", prop)

            unreal.EditorAssetLibrary.save_loaded_asset(blueprint)
            # unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True,True)
            # editor.end_actor_edit_session(True, True)
            '''

            break

        TMP_DISABLE = True

        if TMP_DISABLE:
            for component in actor.get_components_by_class(unreal.StaticMeshComponent):
                mesh_name = component.get_name()

                _static = component.get_editor_property("static_mesh")
                if _static is not None:
                    mesh_static = _static.get_name()
                else:
                    mesh_static = ""

                stencil_value = 77  # @todo define background / non-labelled above

                # Iterate through the list of regexes, stencil values, and match types

                for stencil in reversed(stencil_list):
                    if stencil["match_type"] == "mesh_static" and unreal.StringLibrary.matches_wildcard(mesh_static, stencil["regex"]):
                        stencil_value = stencil["stencil_value"]
                        break
                    elif stencil["match_type"] == "mesh_name" and unreal.StringLibrary.matches_wildcard(mesh_name, stencil["regex"]):
                        stencil_value = stencil["stencil_value"]
                        break
                    elif stencil["match_type"] == "actor_label" and unreal.StringLibrary.matches_wildcard(actor_label_name, stencil["regex"]):
                        stencil_value = stencil["stencil_value"]
                        break
                    elif stencil["match_type"] == "actor_name" and unreal.StringLibrary.matches_wildcard(actor_name, stencil["regex"]):
                        stencil_value = stencil["stencil_value"]
                        break


                # Activate Custom Depth and Set the stencil value for the mesh
                component.set_editor_property("render_custom_depth", True)
                component.set_editor_property("custom_depth_stencil_value", stencil_value)
