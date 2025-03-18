import os
import json
import numpy as np
import subprocess
import tempfile

class Model3DGenerator:
    def __init__(self, floor_plan_data, output_dir, wall_height=2.5, blender_path="blender"):
        """
        Initialize the 3D model generator.
        
        Args:
            floor_plan_data (dict): Data extracted from the floor plan
            output_dir (str): Directory to save the output files
            wall_height (float): Height of the walls in meters
            blender_path (str): Path to the Blender executable
        """
        self.floor_plan_data = floor_plan_data
        self.output_dir = output_dir
        self.wall_height = wall_height
        self.blender_path = blender_path
        
        # Extract required data
        self.walls = floor_plan_data['walls']
        self.doors = floor_plan_data['doors']
        self.windows = floor_plan_data['windows']
        self.rooms = floor_plan_data['rooms']
        self.dimensions = floor_plan_data['dimensions']
        
        # Scale factor (meters per pixel)
        self.scale_factor = self.dimensions['scale_factor']
        
        # Wall thickness in meters
        self.wall_thickness = 0.2  # Default wall thickness
        
    def create_basic_model(self):
     """
     Create a basic 3D model in OBJ format.
     
     Returns:
         str: Path to the created OBJ file
     """
     # Create output path
     obj_path = os.path.join(self.output_dir, 'basic_model.obj')
     mtl_path = os.path.join(self.output_dir, 'basic_model.mtl')
     
     # Create vertices and faces for the model
     vertices = []
     faces = []
     normals = []
     
     # Create materials (unchanged)
     with open(mtl_path, 'w') as mtl_file:
         mtl_file.write("# Basic materials for floor plan model\n")
         
         # Wall material
         mtl_file.write("newmtl wall\n")
         mtl_file.write("Ka 0.8 0.8 0.8\n")
         mtl_file.write("Kd 0.8 0.8 0.8\n")
         mtl_file.write("Ks 0.1 0.1 0.1\n")
         mtl_file.write("Ns 10\n")
         
         # Floor material
         mtl_file.write("newmtl floor\n")
         mtl_file.write("Ka 0.6 0.6 0.6\n")
         mtl_file.write("Kd 0.6 0.6 0.6\n")
         mtl_file.write("Ks 0.1 0.1 0.1\n")
         mtl_file.write("Ns 10\n")
         
         # Door material
         mtl_file.write("newmtl door\n")
         mtl_file.write("Ka 0.6 0.3 0.1\n")
         mtl_file.write("Kd 0.6 0.3 0.1\n")
         mtl_file.write("Ks 0.2 0.2 0.2\n")
         mtl_file.write("Ns 20\n")
         
         # Window material
         mtl_file.write("newmtl window\n")
         mtl_file.write("Ka 0.7 0.8 1.0\n")
         mtl_file.write("Kd 0.7 0.8 1.0\n")
         mtl_file.write("Ks 0.3 0.3 0.3\n")
         mtl_file.write("Ns 30\n")
         mtl_file.write("d 0.7\n")  # Transparency
     
     # Create OBJ file
     with open(obj_path, 'w') as obj_file:
         obj_file.write("# 3D Model generated from floor plan\n")
         obj_file.write(f"mtllib {os.path.basename(mtl_path)}\n")
         
         vertex_count = 1  # OBJ indices start at 1 (moved up for consistency)
         
         # Calculate offsets to center the model
         offset_x = self.dimensions['min_x']
         offset_y = self.dimensions['min_y']
         
         # Add overall building floor
         obj_file.write("g building_floor\n")
         obj_file.write("usemtl floor\n")
         
         floor_vertices = []
         # Use wall extents for precise fit
         width_meters = (self.dimensions['max_x'] - self.dimensions['min_x']) * self.scale_factor
         height_meters = (self.dimensions['max_y'] - self.dimensions['min_y']) * self.scale_factor
         v1 = (0, 0, 0)  # Bottom-left
         v2 = (width_meters, 0, 0)  # Bottom-right
         v3 = (width_meters, 0, -height_meters)  # Top-right
         v4 = (0, 0, -height_meters)  # Top-left
         
         # Add vertices to the list
         for v in [v1, v2, v3, v4]:
             vertices.append(v)
             floor_vertices.append(vertex_count)
             vertex_count += 1
         
         # Add two triangles to form the rectangular floor
         faces.append(("floor", (floor_vertices[0], floor_vertices[1], floor_vertices[2])))  # First triangle
         faces.append(("floor", (floor_vertices[0], floor_vertices[2], floor_vertices[3])))  # Second triangle
         obj_file.write("g floor\n")
         obj_file.write("usemtl floor\n")
         
         for room_idx, room in enumerate(self.rooms):
             room_vertices = []
             
             # Convert contour points to 3D vertices
             for point in room.reshape(-1, 2):
                 x, y = point
                 # Transform to meters and center the model
                 x_meters = (x - offset_x) * self.scale_factor
                 y_meters = (y - offset_y) * self.scale_factor
                 
                 # Add vertex (Y is up, Z is negated)
                 vertices.append((x_meters, 0, -y_meters))
                 room_vertices.append(vertex_count)
                 vertex_count += 1
             
             # Create a simple triangulation of the floor (assuming convex shapes)
             if len(room_vertices) > 3:
                 for i in range(1, len(room_vertices) - 1):
                     faces.append(("floor", (room_vertices[0], room_vertices[i], room_vertices[i+1])))
         
         # Add walls
         wall_id = 0
         for x1, y1, x2, y2 in self.walls:
             # Convert to meters
             x1_meters = (x1 - offset_x) * self.scale_factor
             y1_meters = (y1 - offset_y) * self.scale_factor
             x2_meters = (x2 - offset_x) * self.scale_factor
             y2_meters = (y2 - offset_y) * self.scale_factor
             
             # Calculate wall direction
             wall_vec = np.array([x2_meters - x1_meters, y2_meters - y1_meters])
             wall_length = np.sqrt(np.sum(wall_vec**2))
             
             if wall_length < 0.1:  # Skip very short walls
                 continue
             
             # Normalize
             wall_vec = wall_vec / wall_length
             
             # Calculate perpendicular direction for wall thickness
             perp_vec = np.array([-wall_vec[1], wall_vec[0]]) * (self.wall_thickness / 2)
             
             # Create wall vertices (4 corners at the bottom, 4 at the top)
             v1 = (x1_meters + perp_vec[0], 0, -(y1_meters + perp_vec[1]))  # Bottom left
             v2 = (x1_meters - perp_vec[0], 0, -(y1_meters - perp_vec[1]))  # Bottom right
             v3 = (x2_meters - perp_vec[0], 0, -(y2_meters - perp_vec[1]))  # Top right
             v4 = (x2_meters + perp_vec[0], 0, -(y2_meters + perp_vec[1]))  # Top left
             
             v5 = (v1[0], self.wall_height, v1[2])  # Top versions
             v6 = (v2[0], self.wall_height, v2[2])
             v7 = (v3[0], self.wall_height, v3[2])
             v8 = (v4[0], self.wall_height, v4[2])
             
             # Add vertices
             wall_vertices = []
             for v in [v1, v2, v3, v4, v5, v6, v7, v8]:
                 vertices.append(v)
                 wall_vertices.append(vertex_count)
                 vertex_count += 1
             
             # Add faces for the wall (6 faces for a box)
             faces.append(("wall", (wall_vertices[0], wall_vertices[1], wall_vertices[2], wall_vertices[3])))  # Bottom
             faces.append(("wall", (wall_vertices[4], wall_vertices[7], wall_vertices[6], wall_vertices[5])))  # Top
             faces.append(("wall", (wall_vertices[0], wall_vertices[3], wall_vertices[7], wall_vertices[4])))  # Front
             faces.append(("wall", (wall_vertices[1], wall_vertices[5], wall_vertices[6], wall_vertices[2])))  # Back
             faces.append(("wall", (wall_vertices[0], wall_vertices[4], wall_vertices[5], wall_vertices[1])))  # Left
             faces.append(("wall", (wall_vertices[3], wall_vertices[2], wall_vertices[6], wall_vertices[7])))  # Right
             
             wall_id += 1
         
         # Add doors
         door_id = 0
         for x, y, w, h, angle in self.doors:
             # Convert to meters
             x_meters = (x - offset_x) * self.scale_factor
             y_meters = (y - offset_y) * self.scale_factor
             w_meters = w * self.scale_factor
             h_meters = h * self.scale_factor
             
             # Door dimensions
             door_width = w_meters if angle == 0 else h_meters
             door_height = self.wall_height * 0.8  # Typical door height
             door_thickness = self.wall_thickness
             
             # Create vertices for the door
             if angle == 0:  # Horizontal door
                 v1 = (x_meters, 0, -y_meters)
                 v2 = (x_meters + w_meters, 0, -y_meters)
                 v3 = (x_meters + w_meters, 0, -(y_meters + h_meters))
                 v4 = (x_meters, 0, -(y_meters + h_meters))
             else:  # Vertical door
                 v1 = (x_meters, 0, -y_meters)
                 v2 = (x_meters + w_meters, 0, -y_meters)
                 v3 = (x_meters + w_meters, 0, -(y_meters + h_meters))
                 v4 = (x_meters, 0, -(y_meters + h_meters))
             
             v5 = (v1[0], door_height, v1[2])
             v6 = (v2[0], door_height, v2[2])
             v7 = (v3[0], door_height, v3[2])
             v8 = (v4[0], door_height, v4[2])
             
             # Add vertices
             door_vertices = []
             for v in [v1, v2, v3, v4, v5, v6, v7, v8]:
                 vertices.append(v)
                 door_vertices.append(vertex_count)
                 vertex_count += 1
             
             # Add faces for the door
             faces.append(("door", (door_vertices[0], door_vertices[1], door_vertices[2], door_vertices[3])))  # Bottom
             faces.append(("door", (door_vertices[4], door_vertices[7], door_vertices[6], door_vertices[5])))  # Top
             faces.append(("door", (door_vertices[0], door_vertices[3], door_vertices[7], door_vertices[4])))  # Front
             faces.append(("door", (door_vertices[1], door_vertices[5], door_vertices[6], door_vertices[2])))  # Back
             faces.append(("door", (door_vertices[0], door_vertices[4], door_vertices[5], door_vertices[1])))  # Left
             faces.append(("door", (door_vertices[3], door_vertices[2], door_vertices[6], door_vertices[7])))  # Right
             
             door_id += 1
         
         # Add windows
         window_id = 0
         for x, y, w, h, angle in self.windows:
             # Convert to meters
             x_meters = (x - offset_x) * self.scale_factor
             y_meters = (y - offset_y) * self.scale_factor
             w_meters = w * self.scale_factor
             h_meters = h * self.scale_factor
             
             # Window dimensions
             window_width = w_meters if angle == 0 else h_meters
             window_height = self.wall_height * 0.5  # Typical window height
             window_sill_height = self.wall_height * 0.3  # Height of window sill from floor
             window_thickness = self.wall_thickness / 2
             
             # Create vertices for the window
             if angle == 0:  # Horizontal window
                 v1 = (x_meters, window_sill_height, -y_meters)
                 v2 = (x_meters + w_meters, window_sill_height, -y_meters)
                 v3 = (x_meters + w_meters, window_sill_height, -(y_meters + h_meters))
                 v4 = (x_meters, window_sill_height, -(y_meters + h_meters))
             else:  # Vertical window
                 v1 = (x_meters, window_sill_height, -y_meters)
                 v2 = (x_meters + w_meters, window_sill_height, -y_meters)
                 v3 = (x_meters + w_meters, window_sill_height, -(y_meters + h_meters))
                 v4 = (x_meters, window_sill_height, -(y_meters + h_meters))
             
             v5 = (v1[0], window_sill_height + window_height, v1[2])
             v6 = (v2[0], window_sill_height + window_height, v2[2])
             v7 = (v3[0], window_sill_height + window_height, v3[2])
             v8 = (v4[0], window_sill_height + window_height, v4[2])
             
             # Add vertices
             window_vertices = []
             for v in [v1, v2, v3, v4, v5, v6, v7, v8]:
                 vertices.append(v)
                 window_vertices.append(vertex_count)
                 vertex_count += 1
             
             # Add faces for the window
             faces.append(("window", (window_vertices[0], window_vertices[3], window_vertices[2], window_vertices[1])))  # Bottom
             faces.append(("window", (window_vertices[4], window_vertices[5], window_vertices[6], window_vertices[7])))  # Top
             faces.append(("window", (window_vertices[0], window_vertices[1], window_vertices[5], window_vertices[4])))  # Front
             faces.append(("window", (window_vertices[3], window_vertices[7], window_vertices[6], window_vertices[2])))  # Back
             faces.append(("window", (window_vertices[0], window_vertices[4], window_vertices[7], window_vertices[3])))  # Left
             faces.append(("window", (window_vertices[1], window_vertices[2], window_vertices[6], window_vertices[5])))  # Right
             
             window_id += 1
         
         # Write vertices
         for x, y, z in vertices:
             obj_file.write(f"v {x:.4f} {y:.4f} {z:.4f}\n")
         
         # Write normals (simplified)
         obj_file.write("vn 0 1 0\n")  # Up
         obj_file.write("vn 0 -1 0\n")  # Down
         obj_file.write("vn 0 0 1\n")  # Front
         obj_file.write("vn 0 0 -1\n")  # Back
         obj_file.write("vn 1 0 0\n")  # Right
         obj_file.write("vn -1 0 0\n")  # Left
         
         # Write faces with materials
         current_material = None
         
         for material, indices in faces:
             if material != current_material:
                 obj_file.write(f"usemtl {material}\n")
                 current_material = material
             
             # Write face
             if len(indices) == 3:  # Triangle
                 obj_file.write(f"f {indices[0]} {indices[1]} {indices[2]}\n")
             elif len(indices) == 4:  # Quad
                 obj_file.write(f"f {indices[0]} {indices[1]} {indices[2]} {indices[3]}\n")
     
     return obj_path
    
    def create_detailed_model(self, basic_model_path):
        """
        Create a more detailed 3D model using Blender (if available).
        
        Args:
            basic_model_path (str): Path to the basic OBJ model
            
        Returns:
            str: Path to the created detailed model file
        """
        # Output file path
        output_path = os.path.join(self.output_dir, 'detailed_model.blend')
        
        # Create a temporary Python script for Blender
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as script_file:
            script_path = script_file.name
            
            # Write Blender Python script
            script_file.write("""
import bpy
import os
import sys

# Get arguments
argv = sys.argv
argv = argv[argv.index("--") + 1:]  # Get all args after "--"
input_path = argv[0]
output_path = argv[1]
wall_height = float(argv[2])

# Clear default scene
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import OBJ with correct split_mode parameter
bpy.ops.import_scene.obj(filepath=input_path, split_mode='ON')

# Add some basic lighting
bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
sun = bpy.context.active_object
sun.data.energy = 2.0

# Add a camera
bpy.ops.object.camera_add(location=(5, -5, 5))
cam = bpy.context.active_object
cam.rotation_euler = (0.9, 0, 0.8)
bpy.context.scene.camera = cam

# Separate objects by material and apply modifiers
objects_to_process = []
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH':
        # Select and make active
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        
        # Separate by material if it has multiple materials
        if len(obj.material_slots) > 1:
            bpy.ops.mesh.separate(type='MATERIAL')
        
        # Store for later processing
        objects_to_process.append(obj)

# Now process all mesh objects (including newly created ones)
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH':
        # Rename objects based on their material
        if len(obj.material_slots) > 0:
            material_name = obj.material_slots[0].material.name if obj.material_slots[0].material else "unknown"
            obj.name = material_name
            
            # Apply modifiers based on object type
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            
            if "wall" in material_name.lower():
                bpy.ops.object.modifier_add(type='BEVEL')
                obj.modifiers["Bevel"].width = 0.05
                obj.modifiers["Bevel"].segments = 3
            
            elif "window" in material_name.lower():
                # Make windows slightly transparent
                if obj.material_slots[0].material:
                    mat = obj.material_slots[0].material
                    if not mat.use_nodes:
                        mat.use_nodes = True
                    
                    # Set up basic transparency
                    principled = mat.node_tree.nodes.get('Principled BSDF')
                    if principled:
                        principled.inputs['Alpha'].default_value = 0.7
                    
                    mat.blend_method = 'BLEND'

# Set up rendering settings
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.device = 'CPU'
bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 1080
bpy.context.scene.render.film_transparent = True

# Save the file
bpy.ops.wm.save_as_mainfile(filepath=output_path)
""")
        
        # Try to run Blender if available
        try:
            subprocess.run([
                self.blender_path, 
                "--background", 
                "--python", script_path, 
                "--", 
                basic_model_path, 
                output_path, 
                str(self.wall_height)
            ], check=True)
            
            print(f"Detailed model created with Blender at: {output_path}")
            success = True
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            print(f"Could not create detailed model with Blender: {e}")
            success = False
        
        # Clean up temp script
        os.unlink(script_path)
        
        return output_path if success else None
    
    def save_metadata(self):
        """
        Save metadata about the 3D model.
        
        Returns:
            str: Path to the metadata file
        """
        metadata_path = os.path.join(self.output_dir, 'model_metadata.json')
        
        # Calculate room statistics
        room_data = []
        for i, room_contour in enumerate(self.rooms):
            room_info = {
                "id": i + 1,
                "area_m2": self.dimensions['rooms'][i]['area_meters'],
                "perimeter_m": self.dimensions['rooms'][i]['perimeter_meters'],
                "width_m": self.dimensions['rooms'][i]['width_meters'],
                "height_m": self.dimensions['rooms'][i]['height_meters']
            }
            room_data.append(room_info)
        
        metadata = {
            "model_info": {
                "scale_factor": self.scale_factor,
                "wall_height": self.wall_height,
                "wall_thickness": self.wall_thickness,
                "overall_width_m": self.dimensions['overall_width_meters'],
                "overall_height_m": self.dimensions['overall_height_meters']
            },
            "statistics": {
                "num_walls": len(self.walls),
                "num_doors": len(self.doors),
                "num_windows": len(self.windows),
                "num_rooms": len(self.rooms),
                "total_area_m2": sum(room['area_m2'] for room in room_data)
            },
            "rooms": room_data
        }
        
        # Save to JSON file
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        return metadata_path
    
    def process(self):
        """
        Process the floor plan data and generate the 3D model.
        
        Returns:
            dict: Dictionary with paths to the generated files
        """
        # Step 1: Create basic OBJ model
        basic_model_path = self.create_basic_model()
        
        # Step 2: Create detailed model with Blender if available
        detailed_model_path = self.create_detailed_model(basic_model_path)
        
        # Step 3: Save metadata
        metadata_path = self.save_metadata()
        
        return {
            "basic_model": basic_model_path,
            "processed_model": detailed_model_path,
            "metadata": metadata_path
        }