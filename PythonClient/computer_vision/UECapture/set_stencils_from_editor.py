import unreal
from unreal import LevelEditorSubsystem

# Define the list of regexes, stencil values, and match types
stencil_list = [
    {"regex": "*MASS*", "stencil_value": 1, "match_type": "actor_name"},
    {"regex": "*Street_Furniture*", "stencil_value": 1, "match_type": "actor_label"},
    {"regex": "*bldg*", "stencil_value": 2, "match_type": "actor_label"},
    {"regex": "*EuropeanHorn*", "stencil_value": 3, "match_type": "mesh_name"} # 
]

# Get a reference to the current level
les = unreal.get_editor_subsystem(LevelEditorSubsystem)
level = les.get_current_level()

# Get all actors in the current level
level_actors = unreal.EditorLevelLibrary.get_all_level_actors()

print("len: ", len(level_actors))

#pattern = "*europeantree*"
#is_match = unreal.StringLibrary.matches_wildcard("avc_EuroPeanTreE_sss", pattern)
#print("is_match: ", is_match)
# level = unreal.EditorLevelLibrary.get_editor_world().get_current_level()

# Iterate through all actors in the level
for actor in level_actors:
    actor_name = actor.get_name()
    actor_label_name = actor.get_actor_label()

    # Iterate through all meshes in the actor
    for component in actor.get_components_by_class(unreal.StaticMeshComponent):

        mesh_name = component.get_name()
        stencil_value = 0

        if unreal.StringLibrary.matches_wildcard(mesh_name, "*Tree_Birch*"):
            print("Tree_Birch!")
            print(component.get_editor_property("static_mesh").get_name())

        # Iterate through the list of regexes, stencil values, and match types
        for stencil in stencil_list:
            if stencil["match_type"] == "actor_name" and unreal.StringLibrary.matches_wildcard(actor_name, stencil["regex"]):
                stencil_value = stencil["stencil_value"]
                break
            elif stencil["match_type"] == "actor_label" and unreal.StringLibrary.matches_wildcard(actor_label_name, stencil["regex"]):
                stencil_value = stencil["stencil_value"]
                break
            elif stencil["match_type"] == "mesh_name" and unreal.StringLibrary.matches_wildcard(mesh_name, stencil["regex"]):
                stencil_value = stencil["stencil_value"]
                break

        # Activate Custom Depth and Set the stencil value for the mesh
        component.set_editor_property("render_custom_depth", True) 
        component.set_editor_property("custom_depth_stencil_value", stencil_value) 
