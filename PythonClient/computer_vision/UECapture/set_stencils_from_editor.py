import unreal
import argparse

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

UNLABELLED_STENCIL_VAL = 255
# the below list is a priority list (order matters). From lowest to highest prior. 
stencil_list = [
    # {"regex": "*", "stencil_value": 255, "match_type": "actor_name"},  # unlabelled
    {"regex": "*HLOD*", "stencil_value": 18, "match_type": "actor_name"},  # building
    {"regex": "*bgcity*", "stencil_value": 18, "match_type": "actor_name"},  # building

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

    {"regex": "*bike*", "stencil_value": 2, "match_type": "actor_label"},   # street furniture
    {"regex": "*Barrier*", "stencil_value": 2, "match_type": "actor_label"},   # street furniture
    {"regex": "*Street_Furniture*", "stencil_value": 2, "match_type": "actor_label"},   # street furniture
    {"regex": "*Sidewalk*", "stencil_value": 7, "match_type": "actor_label"},   # road
    {"regex": "*StartingArea*", "stencil_value": 2, "match_type": "actor_label"},   # street furniture
    {"regex": "*Plaza*", "stencil_value": 2, "match_type": "actor_label"},   # street furniture

    {"regex": "*BPC_SF*", "stencil_value": 18, "match_type": "actor_label"},   # building
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

def set_stencils_for_current_level():
    '''
    setting all the stencils for the current level
    '''

    # Get all actors in the current level
    level_actors = unreal.EditorLevelLibrary.get_all_level_actors()
    print("actos in this level: ", len(level_actors))

    for actor in level_actors:
        actor_name = actor.get_name()
        actor_label_name = actor.get_actor_label()

        # Iterate through all meshes in the actor
        for component in actor.get_components_by_class(unreal.StaticMeshComponent):
            mesh_name = component.get_name()

            _static = component.get_editor_property("static_mesh")
            if _static is not None:
                mesh_static = _static.get_name()
            else:
                mesh_static = ""

            stencil_value = UNLABELLED_STENCIL_VAL  # @todo define background / non-labelled above

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


# set stencils for all sub-levels
def iterate_levels_and_set_stencils():
    # alternative way of excluding things
    '''
    exclusion_set = [unreal.TopLevelAssetPath("/Game/Building/map/Kit_Hero_Buildings"),
                     unreal.TopLevelAssetPath("/Game/Environment/Courtyard/Kit_courtyard_biomes")]
    '''

    directory_paths = ["/Game/Environment/",
                       "/Game/Building/",
                       ]

    asset_registry = unreal.AssetRegistryHelpers.get_asset_registry()

    path_filter = unreal.ARFilter(package_paths=directory_paths, recursive_paths=True)
                                  #, recursive_class_paths_exclusion_set=exclusion_set)

    # get a list of asset data objects that match the filter
    asset_data_list = asset_registry.get_assets(path_filter)
    print("len(asset_data_list):", len(asset_data_list))

    excluded_list = []
    # loop through the asset data objects and print their names
    for asset_data in asset_data_list:
        asset_type = asset_data.asset_class_path.asset_name
        asset_name = asset_data.asset_name
        if "kit_" not in str(asset_name).lower():  # exclude all kits (redundant and faster loading)
            if asset_type == "World":
                unreal.EditorLevelLibrary.load_level(asset_data.package_name)
                set_stencils_for_current_level()
                unreal.EditorLevelLibrary.save_current_level()
            elif asset_type == "Blueprint":
                pass
            else:
                pass
        else:
            excluded_list.append(asset_name)

    print("excluded levels in the process:", excluded_list)
    print("finished processing")

def main(is_current, is_sublevels):
    # set stencils for all sublevels or for current level
    if is_sublevels:
        iterate_levels_and_set_stencils()
    if is_current:
        set_stencils_for_current_level()

if __name__ == '__main__':
    # create an argument parser
    parser = argparse.ArgumentParser(description='Example script with arguments')

    # add the arguments
    parser.add_argument('--current', action='store_true', help='set all stencils for the current level. '
                                                               'Load bigCity map manually, in a '
                                                               'World Partition. then'
                                                               ' select big region e.g. ~(1/4) worked for me, then '
                                                               'call this method. '
                                                               'Repeat multiple times as long as you have covered '
                                                               'entire map. '
                                                               'Potentially you could try load entire map but '
                                                               'it is ram exhaustive. Note that only actors '
                                                               'loaded in World Partition will be subject to '
                                                               'stencil changes ')
    parser.add_argument('--sublevels', action='store_true', help='iterate all sublevels of the bigCity. '
                                                                 'Note, that setting stencils for bigCity will not '
                                                                 'adjust stencils for sublevels, (PackagedLevelActor) '
                                                                 'as they are not loaded')

    # parse the arguments
    args = parser.parse_args()

    # call the main function with the arguments
    main(is_current=args.current, is_sublevels=args.sublevels)