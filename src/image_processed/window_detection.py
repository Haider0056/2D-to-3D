from inference_sdk import InferenceHTTPClient
import numpy as np

def detect_windows(self, image_path):
    """
    Detect windows in the floor plan using the Roboflow API and filter out
    windows that don't have any wall attached to them.
    
    Args:
        image_path (str): Path to the floor plan image.
    
    Returns:
        list: List of detected window positions [(x, y, width, height, angle), ...]
    """
    
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="Bg1c2OnSG36KxwCjUl82"
    )

    result = CLIENT.infer(image_path, model_id="window-detection-in-floor-plans/1")

    # Parse result into (x, y, width, height, angle)
    window_candidates_list = []
    for prediction in result['predictions']:
        x_center = prediction['x']
        y_center = prediction['y']
        width = prediction['width']
        height = prediction['height']
        
        # Estimate angle (assuming horizontal or vertical orientation)
        angle = 0 if width > height else 90
        
        # Convert to (x, y, width, height, angle)
        x = int(x_center - width / 2)
        y = int(y_center - height / 2)
        window_candidates_list.append((x, y, int(width), int(height), angle))
    
    # Filter out windows that don't have any wall attached
    filtered_windows = []
    
    if self.walls is None:
        # If walls are not detected yet, store all windows temporarily
        self.windows = window_candidates_list
        return window_candidates_list
    
    # Maximum distance a window can be from a wall to be considered attached
    # Using a more generous distance threshold
    max_distance = max(self.estimated_wall_thickness * 0.5, 5)  # At least 15 pixels
    
    for window in window_candidates_list:
        x, y, w, h, angle = window
        
        # Calculate window center
        center_x = x + w/2
        center_y = y + h/2
        
        # Check both the corners and the midpoints of the window edges
        window_points = [
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
        
        # Check if any point of the window is close to a wall
        is_attached_to_wall = False
        min_dist = float('inf')
        
        for point_x, point_y in window_points:
            for wall_x1, wall_y1, wall_x2, wall_y2 in self.walls:
                distance = self._point_to_line_distance(point_x, point_y, wall_x1, wall_y1, wall_x2, wall_y2)
                min_dist = min(min_dist, distance)
                if distance <= max_distance:
                    is_attached_to_wall = True
                    break
            if is_attached_to_wall:
                break
        
        # If the window is close to a wall, include it in the filtered list
        if is_attached_to_wall:
            filtered_windows.append(window)
        elif min_dist < max_distance * 2:  # Allow windows that are somewhat close but not quite at the threshold
            filtered_windows.append(window)
    
    self.windows = filtered_windows
    return filtered_windows