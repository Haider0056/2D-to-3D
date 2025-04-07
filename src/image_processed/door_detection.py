import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from dotenv import load_dotenv
import os
# Load environment variables from .env
load_dotenv()

# Access the variables
api_key = os.getenv("API_KEY")
def detect_doors(self, image_path, confidence_threshold=0.1, overlap_threshold=0.5):
    """
    Detect doors in the floor plan using the Roboflow API and represent them
    as closed doors attached to walls.
    
    Args:
        image_path (str): Path to the floor plan image.
        confidence_threshold (int): Minimum confidence score (0-100) for door detection.
        overlap_threshold (int): Maximum overlap percentage (0-100) allowed between doors.
        Note: overlap_threshold is kept for backward compatibility but not used.
    
    Returns:
        list: List of detected door positions [(x, y, width, height, angle), ...]
        Note: The returned coordinates represent closed doors on walls.
    """
    
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=api_key
    )
    custom_configuration = InferenceConfiguration(
        confidence_threshold=confidence_threshold,
    )
    # Use either the path or the image object, depending on what's provided
    if isinstance(image_path, str):
         with CLIENT.use_configuration(custom_configuration):
          result = CLIENT.infer(image_path, model_id="doors-vetjc/1")   #image_path or floor_plan.jpg
    else:
        # Assuming image_path is actually an image object
        # Save the image temporarily and use the path
        temp_path = "/tmp/temp_door_image.jpg"
        cv2.imwrite(temp_path, image_path)
        result = CLIENT.infer(image_path, model_id="doors-vetjc/1")

    # Parse result into (x, y, width, height, angle)
    door_candidates_list = []
    
    # Debug print
    print(f"Number of door predictions: {len(result.get('predictions', []))}")
    
    for prediction in result.get('predictions', []):
        # Filter based on confidence threshold (convert from 0-1 to 0-100)
        confidence = prediction.get('confidence', 0) * 100
        if confidence < confidence_threshold:
            continue
            
        x_center = prediction['x']
        y_center = prediction['y']
        width = prediction['width']
        height = prediction['height']
        
        # Estimate angle (assuming horizontal or vertical orientation)
        angle = 0 if width > height else 90
        
        # Convert to (x, y, width, height, angle)
        x = int(x_center - width / 2)
        y = int(y_center - height / 2)
        door_candidates_list.append((x, y, int(width), int(height), angle, confidence))

    
    # Check if door_candidates_list is empty or not properly formatted
    if not door_candidates_list:
        print("No doors found after confidence filtering")
        return []
    
    # Skip NMS entirely - use all candidates that passed confidence threshold
    filtered_by_overlap = door_candidates_list
    
    # Make sure walls are detected first
    if self.walls is None:
        # If walls are not detected yet, try to detect them
        try:
            self.detect_walls()
        except:
            # If wall detection fails, store all doors temporarily without wall filtering
            self.doors = [(x, y, w, h, a) for x, y, w, h, a, _ in filtered_by_overlap]
            return self.doors
    
    # Maximum distance a door can be from a wall to be considered attached
    max_distance = max(self.estimated_wall_thickness * 0.5, 5)  # At least 5 pixels
    
    final_doors = []
    
    # Door area uniqueness constraint
    area_size = 50  # pixels
    door_areas = {}
    
    # Sort doors by size in descending order
    filtered_by_overlap.sort(key=lambda d: d[2] * d[3], reverse=True)
    
    # Set the standard door size (based on wall thickness)
    standard_door_thickness = max(int(self.estimated_wall_thickness * 1.2), 3)  # Door thickness slightly more than wall
    standard_door_length = int(self.estimated_wall_thickness * 10)  # Approximately 10x wall thickness
    
    for door in filtered_by_overlap:
        x, y, w, h, angle, _ = door
        
        # Calculate door center
        center_x = x + w/2
        center_y = y + h/2
        
        # Check both the corners and the midpoints of the door edges
        door_points = [
            # Corners
            (x, y),             # Top-left
            (x + w, y),         # Top-right
            (x + w, y + h),     # Bottom-right
            (x, y + h),         # Bottom-left
            # Midpoints of edges
            (x + w/2, y),       # Top middle
            (x + w, y + h/2),   # Right middle
            (x + w/2, y + h),   # Bottom middle
            (x, y + h/2),       # Left middle
            # Center
            (center_x, center_y)  # Center
        ]
        
        # Find the closest wall to this door
        closest_wall = None
        closest_wall_dist = float('inf')
        closest_point = None
        
        for point_x, point_y in door_points:
            for wall_x1, wall_y1, wall_x2, wall_y2 in self.walls:
                distance = self._point_to_line_distance(point_x, point_y, wall_x1, wall_y1, wall_x2, wall_y2)
                if distance < closest_wall_dist:
                    closest_wall_dist = distance
                    closest_wall = (wall_x1, wall_y1, wall_x2, wall_y2)
                    closest_point = (point_x, point_y)
        
        # If no close wall or too far from any wall, skip this door
        if closest_wall is None or closest_wall_dist > max_distance * 3:
            continue
            
        # Calculate area coordinates for uniqueness check
        area_x = int(center_x // area_size)
        area_y = int(center_y // area_size)
        area_key = (area_x, area_y)
        
        # Skip if we already have a door in this area
        if area_key in door_areas:
            continue
            
        # Mark this area as having a door
        door_areas[area_key] = True
        
        # Get wall coordinates
        wall_x1, wall_y1, wall_x2, wall_y2 = closest_wall
        
        # Calculate wall angle
        wall_dx = wall_x2 - wall_x1
        wall_dy = wall_y2 - wall_y1
        wall_length = np.sqrt(wall_dx**2 + wall_dy**2)
        
        if wall_length < 1:  # Avoid division by zero
            continue
            
        # Calculate normalized wall direction
        wall_dir_x = wall_dx / wall_length
        wall_dir_y = wall_dy / wall_length
        
        # Determine if wall is horizontal or vertical
        is_horizontal_wall = abs(wall_dir_x) > abs(wall_dir_y)
        
        # Project door point onto wall line to find door position on wall
        door_x, door_y = closest_point
        
        # Calculate projection of door point onto wall
        t = ((door_x - wall_x1) * wall_dir_x + (door_y - wall_y1) * wall_dir_y)
        t = max(0, min(t, wall_length))  # Clamp to wall segment
        
        # Get point on wall
        wall_point_x = wall_x1 + t * wall_dir_x
        wall_point_y = wall_y1 + t * wall_dir_y
        
        # Create a door perpendicular to the wall at this point
        perp_dir_x = -wall_dir_y  # Perpendicular direction
        perp_dir_y = wall_dir_x
        
  
        if is_horizontal_wall:
            # Door will be ON horizontal wall, not perpendicular
            new_angle = 0  # Horizontal door (changed from 90)
            new_w = standard_door_length
            new_h = standard_door_thickness
            new_x = int(wall_point_x - new_w / 2)  # Center door on wall point
            new_y = int(wall_point_y - new_h / 2)  # Position door ON the wall
        else:
            # Door will be ON vertical wall, not perpendicular
            new_angle = 90  # Vertical door (changed from 0)
            new_w = standard_door_thickness
            new_h = standard_door_length
            new_x = int(wall_point_x - new_w / 2)  # Position door ON the wall
            new_y = int(wall_point_y - new_h / 2) + 10  # Center door on wall point
            if wall_dir_x < 0 or (wall_dir_x == 0 and wall_point_x > center_x):
                # If wall is to the left of door center, place door to right of wall
                new_x = int(wall_point_x)
            else:
                # If wall is to the right of door center, place door to left of wall
                new_x = int(wall_point_x - new_w)
        
        # Add the door to final list
        final_doors.append((new_x, new_y, new_w, new_h, new_angle))
    
    self.doors = final_doors
    return final_doors