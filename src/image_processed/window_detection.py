from inference_sdk import InferenceHTTPClient
import numpy as np
import cv2
def detect_windows(self, image_path, confidence_threshold=50, overlap_threshold=50):
    """
    Detect windows in the floor plan using the Roboflow API and filter out
    windows that don't have any wall attached to them.
    
    Args:
        image_path (str): Path to the floor plan image.
        confidence_threshold (int): Minimum confidence score (0-100) for window detection.
        overlap_threshold (int): Maximum overlap percentage (0-100) allowed between windows.
    
    Returns:
        list: List of detected window positions [(x, y, width, height, angle), ...]
    """

    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Overlap threshold: {overlap_threshold}")
    
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="Bg1c2OnSG36KxwCjUl82"
    )

    # Use either the path or the image object, depending on what's provided
    if isinstance(image_path, str):
        result = CLIENT.infer(image_path, model_id="window-detection-in-floor-plans/1")
    else:
        # Assuming image_path is actually an image object
        # You'll need a way to convert OpenCV image to bytes or file
        # For now, this is just a placeholder
        # You may need to save the image temporarily and use the path
        temp_path = "/tmp/temp_image.jpg"
        cv2.imwrite(temp_path, image_path)
        result = CLIENT.infer(temp_path, model_id="window-detection-in-floor-plans/1")

    # Parse result into (x, y, width, height, angle)
    window_candidates_list = []
    
    # Debug print
    print(f"Number of predictions: {len(result.get('predictions', []))}")
    
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
        window_candidates_list.append((x, y, int(width), int(height), angle, confidence))
    
    # Debug print
    print(f"Number of candidate windows after confidence filtering: {len(window_candidates_list)}")
    
    # Check if window_candidates_list is empty or not properly formatted
    if not window_candidates_list:
        print("No windows found after confidence filtering")
        return []
        
    # Apply non-maximum suppression to remove overlapping windows
    try:
        filtered_by_overlap = _non_maximum_suppression(window_candidates_list, overlap_threshold/100)
        print(f"Windows after NMS: {len(filtered_by_overlap)}")
    except Exception as e:
        print(f"Error in NMS: {e}")
        filtered_by_overlap = window_candidates_list  # Fallback
    
    
    # Filter out windows that don't have any wall attached
    filtered_windows = []
    
    if self.walls is None:
        # If walls are not detected yet, store all windows temporarily
        self.windows = [(x, y, w, h, a) for x, y, w, h, a, _ in filtered_by_overlap]
        return self.windows
    
    # Maximum distance a window can be from a wall to be considered attached
    # Using a more generous distance threshold
    max_distance = max(self.estimated_wall_thickness * 0.5, 5)  # At least 15 pixels
    
    for window in filtered_by_overlap:
        x, y, w, h, angle, _ = window
        
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
            filtered_windows.append((x, y, w, h, angle))
        elif min_dist < max_distance * 2:  # Allow windows that are somewhat close but not quite at the threshold
            filtered_windows.append((x, y, w, h, angle))
    
    self.windows = filtered_windows
    return filtered_windows

# Add these methods to the FloorPlanProcessor class in base.py

# In window_detection.py
def _non_maximum_suppression(windows, iou_threshold=0.5):
    """
    Apply non-maximum suppression to remove overlapping window detections.
    
    Args:
        windows (list): List of window candidates (x, y, width, height, angle, confidence)
        iou_threshold (float): Threshold for intersection over union (0-1)
        
    Returns:
        list: Filtered windows with minimal overlap
    """
    # Check if windows is a list and not empty
    if not isinstance(windows, list) or not windows:
        print(f"Warning: Expected a list for windows, got {type(windows)}")
        return []
        
    # Sort windows by confidence (descending)
    try:
        windows.sort(key=lambda x: x[5], reverse=True)
    except (IndexError, TypeError) as e:
        print(f"Error sorting windows: {e}")
        print(f"Windows data: {windows}")
        # Fallback: return the input if sorting fails
        return windows
    
    selected_windows = []
    
    while windows:
        # Select the window with highest confidence
        current = windows.pop(0)
        selected_windows.append(current)
        
        # Remove windows that have high overlap with the selected one
        # Pass both arguments explicitly to the standalone function
        windows = [w for w in windows if _calculate_iou(current, w) < iou_threshold]
    
    return selected_windows
def _calculate_iou(window1, window2):
    """
    Calculate the Intersection over Union (IoU) between two windows.
    
    Args:
        window1 (tuple): (x, y, width, height, angle, confidence)
        window2 (tuple): (x, y, width, height, angle, confidence)
        
    Returns:
        float: IoU value (0-1)
    """
    # Extract coordinates
    x1, y1, w1, h1, _, _ = window1
    x2, y2, w2, h2, _, _ = window2
    
    # Calculate the coordinates of the intersection rectangle
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    # If there's no overlap, return 0
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    window1_area = w1 * h1
    window2_area = w2 * h2
    union_area = window1_area + window2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    
    return iou