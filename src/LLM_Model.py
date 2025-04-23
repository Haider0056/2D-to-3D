import os
import json
import numpy as np
import subprocess
import tempfile

class Model3DGenerator:
    def __init__(self, floor_plan_data, output_dir, wall_height=2.7, blender_path="blender"):
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
        
        # Wall thickness in meters (increased for better visibility)
        self.wall_thickness = 0.15
        
        # Door and window parameters
        self.door_thickness = 0.08
        self.window_thickness = 0.05
        self.window_sill_height = 0.9  # Height of window sill from floor
        self.window_height = 1.2      # Height of window
        self.door_height = 2.0        # Height of door
        
    def create_basic_model(self):
        """
        Create a basic 3D model in OBJ format with improved geometry.
        
        Returns:
            str: Path to the created OBJ file
        """
        # Create output path
        obj_path = os.path.join(self.output_dir, 'basic_model.obj')
        mtl_path = os.path.join(self.output_dir, 'basic_model.mtl')
        
        # Create vertices and faces for the model
        vertices = []
        faces = []
        
        # Create materials with improved properties
        with open(mtl_path, 'w') as mtl_file:
            mtl_file.write("# Materials for floor plan model\n")
            
            # Wall material
            mtl_file.write("newmtl wall\n")
            mtl_file.write("Ka 0.9 0.9 0.9\n")  # Ambient color
            mtl_file.write("Kd 0.9 0.9 0.9\n")  # Diffuse color
            mtl_file.write("Ks 0.1 0.1 0.1\n")  # Specular color
            mtl_file.write("Ns 10\n")           # Specular exponent
            
            # Floor material
            mtl_file.write("newmtl floor\n")
            mtl_file.write("Ka 0.7 0.7 0.7\n")
            mtl_file.write("Kd 0.7 0.7 0.7\n")
            mtl_file.write("Ks 0.1 0.1 0.1\n")
            mtl_file.write("Ns 5\n")
            
            # Door material
            mtl_file.write("newmtl door\n")
            mtl_file.write("Ka 0.6 0.4 0.2\n")
            mtl_file.write("Kd 0.6 0.4 0.2\n")
            mtl_file.write("Ks 0.3 0.3 0.3\n")
            mtl_file.write("Ns 20\n")
            
            # Window material
            mtl_file.write("newmtl window\n")
            mtl_file.write("Ka 0.7 0.8 1.0\n")
            mtl_file.write("Kd 0.7 0.8 1.0\n")
            mtl_file.write("Ks 0.5 0.5 0.5\n")
            mtl_file.write("Ns 50\n")
            mtl_file.write("d 0.7\n")  # Transparency
            
            # Window frame material
            mtl_file.write("newmtl window_frame\n")
            mtl_file.write("Ka 0.55 0.35 0.15\n")
            mtl_file.write("Kd 0.55 0.35 0.15\n")
            mtl_file.write("Ks 0.2 0.2 0.2\n")
            mtl_file.write("Ns 20\n")
        
        # Create OBJ file
        with open(obj_path, 'w') as obj_file:
            obj_file.write("# 3D Model generated from floor plan\n")
            obj_file.write(f"mtllib {os.path.basename(mtl_path)}\n")
            
            vertex_count = 1  # OBJ indices start at 1
            
            # Calculate offsets to center the model
            offset_x = self.dimensions['min_x']
            offset_y = self.dimensions['min_y']
            
            # Calculate model dimensions
            width_meters = (self.dimensions['max_x'] - self.dimensions['min_x']) * self.scale_factor
            height_meters = (self.dimensions['max_y'] - self.dimensions['min_y']) * self.scale_factor
            
            # Add building floor with proper dimensions
            floor_vertices = []
            # Use precise dimensions
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
            faces.append(("floor", (floor_vertices[0], floor_vertices[1], floor_vertices[2])))
            faces.append(("floor", (floor_vertices[0], floor_vertices[2], floor_vertices[3])))
            
            # Add individual room floors with slight elevation to avoid z-fighting
            for room_idx, room in enumerate(self.rooms):
                room_vertices = []
                
                # Convert contour points to 3D vertices
                for point in room.reshape(-1, 2):
                    x, y = point
                    # Transform to meters and center the model
                    x_meters = (x - offset_x) * self.scale_factor
                    y_meters = (y - offset_y) * self.scale_factor
                    
                    # Add vertex (Y is up, Z is negated) with slight elevation
                    vertices.append((x_meters, 0.01, -y_meters))
                    room_vertices.append(vertex_count)
                    vertex_count += 1
                
                # Create a simple triangulation of the floor (assuming convex shapes)
                if len(room_vertices) > 3:
                    # Triangulate using fan triangulation from first vertex
                    for i in range(1, len(room_vertices) - 1):
                        faces.append(("floor", (room_vertices[0], room_vertices[i], room_vertices[i+1])))
            
            # Add walls with proper thickness
            for wall_idx, (x1, y1, x2, y2) in enumerate(self.walls):
                # Convert to meters
                x1_meters = (x1 - offset_x) * self.scale_factor
                y1_meters = (y1 - offset_y) * self.scale_factor
                x2_meters = (x2 - offset_x) * self.scale_factor
                y2_meters = (y2 - offset_y) * self.scale_factor
                
                # Calculate wall direction vector
                wall_vec = np.array([x2_meters - x1_meters, y2_meters - y1_meters])
                wall_length = np.sqrt(np.sum(wall_vec**2))
                
                if wall_length < 0.1:  # Skip very short walls
                    continue
                
                # Normalize
                wall_vec = wall_vec / wall_length
                
                # Calculate perpendicular vector for wall thickness
                perp_vec = np.array([-wall_vec[1], wall_vec[0]]) * (self.wall_thickness / 2)
                
                # Create wall vertices (4 corners at bottom, 4 at top)
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
                # Bottom
                faces.append(("wall", (wall_vertices[0], wall_vertices[1], wall_vertices[2], wall_vertices[3])))
                # Top
                faces.append(("wall", (wall_vertices[4], wall_vertices[7], wall_vertices[6], wall_vertices[5])))
                # Front
                faces.append(("wall", (wall_vertices[0], wall_vertices[3], wall_vertices[7], wall_vertices[4])))
                # Back
                faces.append(("wall", (wall_vertices[1], wall_vertices[5], wall_vertices[6], wall_vertices[2])))
                # Left
                faces.append(("wall", (wall_vertices[0], wall_vertices[4], wall_vertices[5], wall_vertices[1])))
                # Right
                faces.append(("wall", (wall_vertices[3], wall_vertices[2], wall_vertices[6], wall_vertices[7])))
            
            # Add doors with improved positioning
            for door_idx, (x, y, w, h, angle) in enumerate(self.doors):
                # Convert to meters
                x_meters = (x - offset_x) * self.scale_factor
                y_meters = (y - offset_y) * self.scale_factor
                w_meters = w * self.scale_factor
                h_meters = h * self.scale_factor
                
                # Door dimensions
                door_width = w_meters
                door_height = self.door_height
                door_thickness = self.door_thickness
                
                # Create door placement based on angle
                if angle == 0:  # Horizontal door
                    # Left side
                    v1 = (x_meters, 0, -y_meters)
                    v2 = (x_meters + door_width, 0, -y_meters)
                    v3 = (x_meters + door_width, 0, -(y_meters - door_thickness))
                    v4 = (x_meters, 0, -(y_meters - door_thickness))
                    
                    # Right side 
                    v5 = (v1[0], door_height, v1[2])
                    v6 = (v2[0], door_height, v2[2])
                    v7 = (v3[0], door_height, v3[2])
                    v8 = (v4[0], door_height, v4[2])
                else:  # Vertical door (90 degrees)
                    # Calculate proper door placement
                    v1 = (x_meters, 0, -y_meters)
                    v2 = (x_meters + door_thickness, 0, -y_meters)
                    v3 = (x_meters + door_thickness, 0, -(y_meters + h_meters))
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
                # Bottom
                faces.append(("door", (door_vertices[0], door_vertices[3], door_vertices[2], door_vertices[1])))
                # Top
                faces.append(("door", (door_vertices[4], door_vertices[5], door_vertices[6], door_vertices[7])))
                # Front
                faces.append(("door", (door_vertices[0], door_vertices[1], door_vertices[5], door_vertices[4])))
                # Back
                faces.append(("door", (door_vertices[3], door_vertices[7], door_vertices[6], door_vertices[2])))
                # Left
                faces.append(("door", (door_vertices[0], door_vertices[4], door_vertices[7], door_vertices[3])))
                # Right
                faces.append(("door", (door_vertices[1], door_vertices[2], door_vertices[6], door_vertices[5])))
            
            # Add windows with proper frame
            for window_idx, (x, y, w, h, angle) in enumerate(self.windows):
                # Convert to meters
                x_meters = (x - offset_x) * self.scale_factor
                y_meters = (y - offset_y) * self.scale_factor
                w_meters = w * self.scale_factor
                h_meters = h * self.scale_factor
                
                # Window dimensions
                window_width = w_meters
                window_height = self.window_height
                window_thickness = self.window_thickness
                sill_height = self.window_sill_height
                frame_thickness = 0.05  # Window frame thickness
                
                # Create vertices for the window based on angle
                if angle == 0:  # Horizontal window
                    # Main window glass
                    v1 = (x_meters + frame_thickness, sill_height + frame_thickness, -y_meters)
                    v2 = (x_meters + window_width - frame_thickness, sill_height + frame_thickness, -y_meters)
                    v3 = (x_meters + window_width - frame_thickness, sill_height + window_height - frame_thickness, -y_meters)
                    v4 = (x_meters + frame_thickness, sill_height + window_height - frame_thickness, -y_meters)
                    
                    # Outer window frame (front face)
                    f1 = (x_meters, sill_height, -y_meters)
                    f2 = (x_meters + window_width, sill_height, -y_meters)
                    f3 = (x_meters + window_width, sill_height + window_height, -y_meters)
                    f4 = (x_meters, sill_height + window_height, -y_meters)
                    
                    # Back face of window (with thickness)
                    v5 = (v1[0], v1[1], -(y_meters - window_thickness))
                    v6 = (v2[0], v2[1], -(y_meters - window_thickness))
                    v7 = (v3[0], v3[1], -(y_meters - window_thickness))
                    v8 = (v4[0], v4[1], -(y_meters - window_thickness))
                    
                    # Back face of frame
                    f5 = (f1[0], f1[1], -(y_meters - window_thickness))
                    f6 = (f2[0], f2[1], -(y_meters - window_thickness))
                    f7 = (f3[0], f3[1], -(y_meters - window_thickness))
                    f8 = (f4[0], f4[1], -(y_meters - window_thickness))
                else:  # Vertical window (90 degrees)
                    # Main window glass
                    v1 = (x_meters, sill_height + frame_thickness, -(y_meters + frame_thickness))
                    v2 = (x_meters, sill_height + frame_thickness, -(y_meters + h_meters - frame_thickness))
                    v3 = (x_meters, sill_height + window_height - frame_thickness, -(y_meters + h_meters - frame_thickness))
                    v4 = (x_meters, sill_height + window_height - frame_thickness, -(y_meters + frame_thickness))
                    
                    # Outer window frame
                    f1 = (x_meters, sill_height, -(y_meters))
                    f2 = (x_meters, sill_height, -(y_meters + h_meters))
                    f3 = (x_meters, sill_height + window_height, -(y_meters + h_meters))
                    f4 = (x_meters, sill_height + window_height, -(y_meters))
                    
                    # Back face of window
                    v5 = (x_meters + window_thickness, v1[1], v1[2])
                    v6 = (x_meters + window_thickness, v2[1], v2[2])
                    v7 = (x_meters + window_thickness, v3[1], v3[2])
                    v8 = (x_meters + window_thickness, v4[1], v4[2])
                    
                    # Back face of frame
                    f5 = (x_meters + window_thickness, f1[1], f1[2])
                    f6 = (x_meters + window_thickness, f2[1], f2[2])
                    f7 = (x_meters + window_thickness, f3[1], f3[2])
                    f8 = (x_meters + window_thickness, f4[1], f4[2])
                
                # Add window glass vertices
                window_vertices = []
                for v in [v1, v2, v3, v4, v5, v6, v7, v8]:
                    vertices.append(v)
                    window_vertices.append(vertex_count)
                    vertex_count += 1
                
                # Add window frame vertices
                frame_vertices = []
                for f in [f1, f2, f3, f4, f5, f6, f7, f8]:
                    vertices.append(f)
                    frame_vertices.append(vertex_count)
                    vertex_count += 1
                
                # Add faces for the window glass
                faces.append(("window", (window_vertices[0], window_vertices[3], window_vertices[2], window_vertices[1])))  # Front
                faces.append(("window", (window_vertices[4], window_vertices[5], window_vertices[6], window_vertices[7])))  # Back
                faces.append(("window", (window_vertices[0], window_vertices[1], window_vertices[5], window_vertices[4])))  # Bottom
                faces.append(("window", (window_vertices[3], window_vertices[7], window_vertices[6], window_vertices[2])))  # Top
                faces.append(("window", (window_vertices[0], window_vertices[4], window_vertices[7], window_vertices[3])))  # Left
                faces.append(("window", (window_vertices[1], window_vertices[2], window_vertices[6], window_vertices[5])))  # Right
                
                # Add faces for the window frame
                faces.append(("window_frame", (frame_vertices[0], frame_vertices[3], frame_vertices[2], frame_vertices[1])))  # Front
                faces.append(("window_frame", (frame_vertices[4], frame_vertices[5], frame_vertices[6], frame_vertices[7])))  # Back
                faces.append(("window_frame", (frame_vertices[0], frame_vertices[1], frame_vertices[5], frame_vertices[4])))  # Bottom
                faces.append(("window_frame", (frame_vertices[3], frame_vertices[7], frame_vertices[6], frame_vertices[2])))  # Top
                faces.append(("window_frame", (frame_vertices[0], frame_vertices[4], frame_vertices[7], frame_vertices[3])))  # Left
                faces.append(("window_frame", (frame_vertices[1], frame_vertices[2], frame_vertices[6], frame_vertices[5])))  # Right
            
            # Write vertices
            for x, y, z in vertices:
                obj_file.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            
            # Write normals for various faces
            obj_file.write("vn 0 1 0\n")   # Up
            obj_file.write("vn 0 -1 0\n")  # Down
            obj_file.write("vn 0 0 1\n")   # Front
            obj_file.write("vn 0 0 -1\n")  # Back
            obj_file.write("vn 1 0 0\n")   # Right
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
        Create a more detailed 3D model using Blender with improved materials and lighting.
        
        Args:a
            basic_model_path (str): Path to the basic OBJ model
            
        Returns:
            str: Path to the created detailed model file
        """
        # Output file path
        output_path = os.path.join(self.output_dir, 'detailed_model.blend')
        
        # Create a temporary Python script for Blender
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as script_file:
            script_path = script_file.name
            
            # Write improved Blender Python script
            script_file.write("""
import bpy
import os
import sys
import math

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

# Import OBJ
bpy.ops.import_scene.obj(filepath=input_path, split_mode='ON')

# Create better materials
def create_wall_material():
    mat = bpy.data.materials.new(name="wall_material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    
    # Clear default nodes
    for node in nodes:
        nodes.remove(node)
    
    # Create nodes
    output = nodes.new(type='ShaderNodeOutputMaterial')
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    noise = nodes.new(type='ShaderNodeTexNoise')
    mapping = nodes.new(type='ShaderNodeMapping')
    tex_coord = nodes.new(type='ShaderNodeTexCoord')
    color_ramp = nodes.new(type='ShaderNodeValToRGB')
    
    # Setup noise texture
    noise.inputs['Scale'].default_value = 10.0
    noise.inputs['Detail'].default_value = 2.0
    
    # Setup color ramp
    color_ramp.color_ramp.elements[0].position = 0.4
    color_ramp.color_ramp.elements[0].color = (0.85, 0.85, 0.85, 1.0)
    color_ramp.color_ramp.elements[1].position = 0.6
    color_ramp.color_ramp.elements[1].color = (0.95, 0.95, 0.95, 1.0)
    
    # Setup material
    principled.inputs['Base Color'].default_value = (0.9, 0.9, 0.9, 1.0)
    principled.inputs['Roughness'].default_value = 0.7
    principled.inputs['Specular'].default_value = 0.1
    
    # Link nodes
    links = mat.node_tree.links
    links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
    links.new(mapping.outputs['Vector'], noise.inputs['Vector'])
    links.new(noise.outputs['Fac'], color_ramp.inputs['Fac'])
    links.new(color_ramp.outputs['Color'], principled.inputs['Base Color'])
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    
    return mat

def create_floor_material():
    mat = bpy.data.materials.new(name="floor_material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    
    # Clear default nodes
    for node in nodes:
        nodes.remove(node)
    
    # Create nodes
    output = nodes.new(type='ShaderNodeOutputMaterial')
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    noise = nodes.new(type='ShaderNodeTexNoise')
    mapping = nodes.new(type='ShaderNodeMapping')
    tex_coord = nodes.new(type='ShaderNodeTexCoord')
    
    # Setup noise texture
    noise.inputs['Scale'].default_value = 20.0
    noise.inputs['Detail'].default_value = 3.0
    
    # Setup material
    principled.inputs['Base Color'].default_value = (0.7, 0.68, 0.64, 1.0)
    principled.inputs['Roughness'].default_value = 0.6
    principled.inputs['Specular'].default_value = 0.2
    
    # Link nodes
    links = mat.node_tree.links
    links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
    links.new(mapping.outputs['Vector'], noise.inputs['Vector'])
    links.new(noise.outputs['Fac'], principled.inputs['Normal'])
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    
    return mat

def create_door_material():
    mat = bpy.data.materials.new(name="door_material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    
    # Clear default nodes
    for node in nodes:
        nodes.remove(node)
    
    # Create nodes
    output = nodes.new(type='ShaderNodeOutputMaterial')
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    
    # Setup material
    principled.inputs['Base Color'].default_value = (0.6, 0.4, 0.2, 1.0)
    principled.inputs['Roughness'].default_value = 0.4
    principled.inputs['Specular'].default_value = 0.3
    
    # Link nodes
    links = mat.node_tree.links
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    
    return mat

def create_window_material():
    mat = bpy.data.materials.new(name="window_material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    
    # Clear default nodes
    for node in nodes:
        nodes.remove(node)
    
    # Create nodes
    output = nodes.new(type='ShaderNodeOutputMaterial')
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    
    # Setup material - use transparency for windows
    principled.inputs['Base Color'].default_value = (0.8, 0.9, 1.0, 1.0)
    principled.inputs['Roughness'].default_value = 0.1
    principled.inputs['Specular'].default_value = 0.9
    principled.inputs['Transmission'].default_value = 0.9
    principled.inputs['IOR'].default_value = 1.45
    principled.inputs['Alpha'].default_value = 0.7  # Add transparency
    
    # Link nodes
    links = mat.node_tree.links
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    
    # Set blend mode
    mat.blend_method = 'BLEND'
    mat.shadow_method = 'HASHED'
    
    return mat

def create_window_frame_material():
    mat = bpy.data.materials.new(name="window_frame_material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    
    # Clear default nodes
    for node in nodes:
        nodes.remove(node)
    
    # Create nodes
    output = nodes.new(type='ShaderNodeOutputMaterial')
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    
    # Setup material
    principled.inputs['Base Color'].default_value = (0.55, 0.35, 0.15, 1.0)
    principled.inputs['Roughness'].default_value = 0.5
    principled.inputs['Specular'].default_value = 0.2
    
    # Link nodes
    links = mat.node_tree.links
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    
    return mat

# Create materials
wall_mat = create_wall_material()
floor_mat = create_floor_material()
door_mat = create_door_material()
window_mat = create_window_material()
window_frame_mat = create_window_frame_material()

# Assign materials to objects
for obj in bpy.data.objects:
    if obj.type != 'MESH':
        continue
        
    # Remove existing materials
    while obj.data.materials:
        obj.data.materials.pop()
    
    # Assign appropriate material based on object name
    if 'wall' in obj.name.lower():
        obj.data.materials.append(wall_mat)
    elif 'floor' in obj.name.lower():
        obj.data.materials.append(floor_mat)
    elif 'door' in obj.name.lower():
        obj.data.materials.append(door_mat)
        # Make doors visible from both sides (optional - uncomment if needed)
        # door_mat.blend_method = 'BLEND'
        # if 'Principled BSDF' in door_mat.node_tree.nodes:
        #     door_mat.node_tree.nodes['Principled BSDF'].inputs['Alpha'].default_value = 0.95
    elif 'window_frame' in obj.name.lower():
        obj.data.materials.append(window_frame_mat)
    elif 'window' in obj.name.lower():
        obj.data.materials.append(window_mat)


# Setup camera with automatic positioning
def setup_camera():
    # Get bounds of all objects
    min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
    max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')
    
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
            
        for vertex in obj.bound_box:
            x, y, z = obj.matrix_world @ mathutils.Vector(vertex)
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            min_z = min(min_z, z)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
            max_z = max(max_z, z)
    
    # Calculate model dimensions and center
    width = max_x - min_x
    depth = max_y - min_y
    height = max_z - min_z
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_z = (min_z + max_z) / 2
    
    # Calculate camera position
    distance = max(width, depth) * 1.5
    camera_x = center_x
    camera_y = center_y - distance
    camera_z = center_z + height * 0.7
    
    # Create camera
    bpy.ops.object.camera_add(location=(camera_x, camera_y, camera_z))
    camera = bpy.context.active_object
    camera.name = 'Camera'
    
    # Point camera to model center
    direction = mathutils.Vector((center_x, center_y, center_z)) - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    
    # Set as active camera
    bpy.context.scene.camera = camera
    
    # Set camera settings for better output
    camera.data.lens = 35  # mm
    
    return camera

# Import mathutils if not already imported
import mathutils

# Setup camera
camera = setup_camera()

# Configure render settings
def configure_render_settings():
    render = bpy.context.scene.render
    render.engine = 'CYCLES'  # Use Cycles renderer for better quality
    render.resolution_x = 1920
    render.resolution_y = 1080
    render.resolution_percentage = 100
    render.film_transparent = False
    
    # Configure cycles settings for faster preview
    cycles = bpy.context.scene.cycles
    cycles.samples = 128
    cycles.preview_samples = 32
    cycles.max_bounces = 8
    cycles.tile_x = 256
    cycles.tile_y = 256
    cycles.use_denoising = True

configure_render_settings()

# World settings
def configure_world():
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']
    bg.inputs['Color'].default_value = (0.9, 0.9, 1.0, 1.0)
    bg.inputs['Strength'].default_value = 1.0

configure_world()

# Save the blend file
bpy.ops.wm.save_as_mainfile(filepath=output_path)
print(f"3D model saved to {output_path}")
""")

        # Close the file to ensure it's written to disk
        script_file.close()
        
        try:
            # Execute Blender with the script
            command = [
                self.blender_path,
                "--background",
                "--python", script_path,
                "--",  # Pass the following as arguments to the Python script
                os.path.abspath(basic_model_path),
                os.path.abspath(output_path),
                str(self.wall_height)
            ]
            
            # Run the command
            result = subprocess.run(
                command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            
            print("Blender output:", result.stdout)
            
            # Check if file was created
            if os.path.exists(output_path):
                print(f"Detailed model created successfully: {output_path}")
            else:
                print("Warning: Output file was not created.")
                
        except subprocess.CalledProcessError as e:
            print(f"Error running Blender: {e}")
            print(f"Blender stderr: {e.stderr}")
            return None
        except FileNotFoundError:
            print(f"Error: Blender executable not found at '{self.blender_path}'")
            return None
        finally:
            # Clean up temporary script file
            if os.path.exists(script_path):
                os.remove(script_path)
            
        return output_path

    def generate_model(self):
        """
        Main method to generate 3D models.
        
        Returns:
            dict: Paths to the generated model files
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create basic model
        basic_model_path = self.create_basic_model()
        
        # Create detailed model
        detailed_model_path = self.create_detailed_model(basic_model_path)
        
        return {
            'basic_model': basic_model_path,
            'detailed_model': detailed_model_path
        }