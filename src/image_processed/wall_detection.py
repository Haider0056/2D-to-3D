import cv2
import numpy as np

def _estimate_wall_thickness(binary_image):
    """
    Estimate the wall thickness from the binary image.
    
    Args:
        binary_image (numpy.ndarray): Binary image of the floor plan
        
    Returns:
        int: Estimated wall thickness in pixels
    """
    # Create a distance transform
    dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
    
    # Find local maxima of the distance transform
    kernel = np.ones((5, 5), np.uint8)
    locmax = cv2.dilate(dist_transform, kernel)
    mask = cv2.compare(dist_transform, locmax, cv2.CMP_EQ)
    
    # Get the non-zero values (potential wall thicknesses)
    points = cv2.findNonZero(mask)
    if points is None or len(points) < 10:
        return 5  # Default thickness if detection fails
    
    # Get the distance values at these points
    thicknesses = [dist_transform[y, x] for x, y in points[:, 0]]
    
    # Use the most common thickness value
    hist, bins = np.histogram(thicknesses, bins=20)
    most_common_thickness = bins[np.argmax(hist)] * 2  # Multiply by 2 as distance transform gives half-thickness
    
    return max(3, int(most_common_thickness))

def _are_parallel(line1, line2, angle_threshold=10):
    """
    Check if two lines are parallel within a given angle threshold.
    
    Args:
        line1 (tuple): First line (x1, y1, x2, y2)
        line2 (tuple): Second line (x1, y1, x2, y2)
        angle_threshold (float): Maximum angle difference in degrees
        
    Returns:
        bool: True if lines are parallel, False otherwise
    """
    # Calculate angles
    angle1 = np.arctan2(line1[3] - line1[1], line1[2] - line1[0]) * 180 / np.pi
    angle2 = np.arctan2(line2[3] - line2[1], line2[2] - line2[0]) * 180 / np.pi
    
    # Normalize angles to 0-180 range
    angle1 = (angle1 + 180) % 180
    angle2 = (angle2 + 180) % 180
    
    # Check if angles are similar
    return abs(angle1 - angle2) < angle_threshold

def _calculate_line_distance(line1, line2):
    """
    Calculate the average distance between two parallel lines.
    
    Args:
        line1 (tuple): First line (x1, y1, x2, y2)
        line2 (tuple): Second line (x1, y1, x2, y2)
        
    Returns:
        float: Average distance between the lines
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    # For horizontal lines, calculate vertical distance
    if abs(y2 - y1) < abs(x2 - x1):
        return abs((y1 + y2) / 2 - (y3 + y4) / 2)
    # For vertical lines, calculate horizontal distance
    else:
        return abs((x1 + x2) / 2 - (x3 + x4) / 2)
 
def detect_walls(self):
    """
    Detect walls in the floor plan with improved line detection and merging of parallel lines.
    
    Returns:
        list: List of detected wall line segments [(x1, y1, x2, y2), ...]
    """
    if self.processed_image is None:
        self.preprocess()
    
    # Use Probabilistic Hough Transform with adaptive parameters
    min_line_length = max(30, self.height / 20)
    max_line_gap = self.estimated_wall_thickness * 2
    
    lines = cv2.HoughLinesP(
        self.processed_image, 
        rho=1, 
        theta=np.pi/180, 
        threshold=190, 
        minLineLength=min_line_length, 
        maxLineGap=max_line_gap
    )
    
    if lines is None:
        self.walls = []
        return []
    
    # Extract all line segments
    line_segments = [line[0] for line in lines]
    
    # Group line segments by orientation
    horizontal_lines = []
    vertical_lines = []
    
    for x1, y1, x2, y2 in line_segments:
        # Calculate angle and length
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        # Filter by angle
        if angle < 20 or angle > 160:
            horizontal_lines.append((x1, y1, x2, y2, length))
        elif angle > 70 and angle < 110:
            vertical_lines.append((x1, y1, x2, y2, length))
    
    # Sort by length (longest first)
    horizontal_lines.sort(key=lambda x: x[4], reverse=True)
    vertical_lines.sort(key=lambda x: x[4], reverse=True)
    
    # Merge collinear segments
    merged_horizontal = _merge_line_segments(self, horizontal_lines)
    merged_vertical = _merge_line_segments(self, vertical_lines)
    
    # Extract line segments without length component
    horizontal_segments = [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in merged_horizontal]
    vertical_segments = [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in merged_vertical]
    
    # Merge parallel lines that are close to each other
    distance_threshold = self.estimated_wall_thickness * 4.25  
    
    # Apply the improved parallel line merging
    merged_horizontal = _merge_parallel_lines(horizontal_segments, distance_threshold)
    merged_vertical = _merge_parallel_lines(vertical_segments, distance_threshold)
    
    # Combine all wall lines
    wall_lines = merged_horizontal + merged_vertical
    
    # Filter out very short lines that might be noise
    min_wall_length = max(35, self.estimated_wall_thickness * 3)
    filtered_walls = []
    for x1, y1, x2, y2 in wall_lines:
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length >= min_wall_length:
            filtered_walls.append((x1, y1, x2, y2))
    
    self.walls = filtered_walls
    return filtered_walls

def _merge_parallel_lines(lines, distance_threshold):
    """
    Improved method to merge parallel lines that are within a specified distance threshold.
    Avoids creating spurious walls and has better overlap detection.
    
    Args:
        lines (list): List of line segments [(x1, y1, x2, y2), ...]
        distance_threshold (float): Maximum distance between lines to merge
        
    Returns:
        list: List of merged line segments with integer coordinates
    """
    if not lines:
        return []
    
    result = []
    processed = [False] * len(lines)
    
    # Define a minimum overlap ratio for merging parallel lines
    min_overlap_ratio = 0.1
    
    for i in range(len(lines)):
        if processed[i]:
            continue
            
        processed[i] = True
        current_line = lines[i]
        merged_lines = [current_line]
        
        # Get the "direction" of the current line
        is_horizontal = abs(current_line[3] - current_line[1]) < abs(current_line[2] - current_line[0])
        
        # Find all parallel lines within distance threshold
        for j in range(i + 1, len(lines)):
            if processed[j]:
                continue
                
            other_line = lines[j]
            
            # Only process lines with the same orientation
            other_is_horizontal = abs(other_line[3] - other_line[1]) < abs(other_line[2] - other_line[0])
            if is_horizontal != other_is_horizontal:
                continue
            
            if _are_parallel(current_line, other_line):
                distance = _calculate_line_distance(current_line, other_line)
                
                # Check if lines have sufficient overlap before merging
                has_overlap = False
                if is_horizontal:
                    # For horizontal lines, check x-overlap
                    current_x_range = [min(current_line[0], current_line[2]), max(current_line[0], current_line[2])]
                    other_x_range = [min(other_line[0], other_line[2]), max(other_line[0], other_line[2])]
                    
                    # Calculate overlap
                    overlap_start = max(current_x_range[0], other_x_range[0])
                    overlap_end = min(current_x_range[1], other_x_range[1])
                    overlap_length = max(0, overlap_end - overlap_start)
                    
                    # Current line length
                    current_length = current_x_range[1] - current_x_range[0]
                    other_length = other_x_range[1] - other_x_range[0]
                    
                    # Check if overlap is sufficient
                    if overlap_length > min(current_length, other_length) * min_overlap_ratio:
                        has_overlap = True
                else:
                    # For vertical lines, check y-overlap
                    current_y_range = [min(current_line[1], current_line[3]), max(current_line[1], current_line[3])]
                    other_y_range = [min(other_line[1], other_line[3]), max(other_line[1], other_line[3])]
                    
                    # Calculate overlap
                    overlap_start = max(current_y_range[0], other_y_range[0])
                    overlap_end = min(current_y_range[1], other_y_range[1])
                    overlap_length = max(0, overlap_end - overlap_start)
                    
                    # Current line length
                    current_length = current_y_range[1] - current_y_range[0]
                    other_length = other_y_range[1] - other_y_range[0]
                    
                    # Check if overlap is sufficient
                    if overlap_length > min(current_length, other_length) * min_overlap_ratio:
                        has_overlap = True
                
                # Only merge if lines are close enough and have sufficient overlap
                if distance <= distance_threshold and has_overlap:
                    merged_lines.append(other_line)
                    processed[j] = True
        
        # If we found lines to merge
        if len(merged_lines) > 1:
            # For horizontal lines
            if is_horizontal:
                # Get all x and y coordinates
                all_x = [line[0] for line in merged_lines] + [line[2] for line in merged_lines]
                all_y = [line[1] for line in merged_lines] + [line[3] for line in merged_lines]
                
                # Calculate weighted average y-coordinate based on line lengths
                weighted_y = 0
                total_weight = 0
                
                for x1, y1, x2, y2 in merged_lines:
                    length = abs(x2 - x1)
                    weighted_y += (y1 + y2) / 2 * length
                    total_weight += length
                
                avg_y = weighted_y / total_weight if total_weight > 0 else sum(all_y) / len(all_y)
                
                # Use min and max x values
                min_x = min(all_x)
                max_x = max(all_x)
                
                # Create the merged line
                result.append((int(min_x), int(avg_y), int(max_x), int(avg_y)))
            
            # For vertical lines
            else:
                # Get all x and y coordinates
                all_x = [line[0] for line in merged_lines] + [line[2] for line in merged_lines]
                all_y = [line[1] for line in merged_lines] + [line[3] for line in merged_lines]
                
                # Calculate weighted average x-coordinate based on line lengths
                weighted_x = 0
                total_weight = 0
                
                for x1, y1, x2, y2 in merged_lines:
                    length = abs(y2 - y1)
                    weighted_x += (x1 + x2) / 2 * length
                    total_weight += length
                
                avg_x = weighted_x / total_weight if total_weight > 0 else sum(all_x) / len(all_x)
                
                # Use min and max y values
                min_y = min(all_y)
                max_y = max(all_y)
                
                # Create the merged line
                result.append((int(avg_x), int(min_y), int(avg_x), int(max_y)))
        else:
            # Just add the current line
            result.append((int(current_line[0]), int(current_line[1]), 
                          int(current_line[2]), int(current_line[3])))
    
    return result
    
def _merge_line_segments(self, line_segments):
    """
    Merge collinear line segments.
    
    Args:
        self: The FloorPlanProcessor instance
        line_segments (list): List of line segments with length [(x1, y1, x2, y2, length), ...]
        
    Returns:
        list: List of merged line segments
    """
    if not line_segments:
        return []
    
    merged_lines = []
    
    # Determine if two lines are collinear and close
    def are_collinear(line1, line2):
        x1, y1, x2, y2, _ = line1
        x3, y3, x4, y4, _ = line2
        
        # Check if lines are roughly horizontal or vertical
        is_horizontal1 = abs(y2 - y1) < abs(x2 - x1)
        is_horizontal2 = abs(y4 - y3) < abs(x4 - x3)
        
        if is_horizontal1 != is_horizontal2:
            return False
        
        # For horizontal lines, check y-proximity
        if is_horizontal1:
            if abs(y1 - y3) > self.estimated_wall_thickness or abs(y2 - y4) > self.estimated_wall_thickness:
                return False
            
            # Check if x-ranges overlap or are close
            if max(x1, x2) < min(x3, x4) - self.estimated_wall_thickness * 2:
                return False
            if min(x1, x2) > max(x3, x4) + self.estimated_wall_thickness * 2:
                return False
            
            return True
        
        # For vertical lines, check x-proximity
        else:
            if abs(x1 - x3) > self.estimated_wall_thickness or abs(x2 - x4) > self.estimated_wall_thickness:
                return False
            
            # Check if y-ranges overlap or are close
            if max(y1, y2) < min(y3, y4) - self.estimated_wall_thickness * 2:
                return False
            if min(y1, y2) > max(y3, y4) + self.estimated_wall_thickness * 2:
                return False
            
            return True
    
    # Try to merge each line
    remaining = line_segments.copy()
    
    while remaining:
        current = remaining.pop(0)
        x1, y1, x2, y2, _ = current
        
        # Find all collinear segments
        collinear_indices = []
        for i, other in enumerate(remaining):
            if are_collinear(current, other):
                collinear_indices.append(i)
        
        # Extract collinear segments
        collinear_segments = [remaining[i] for i in sorted(collinear_indices, reverse=True)]
        for i in sorted(collinear_indices, reverse=True):
            remaining.pop(i)
        
        # Merge current with all collinear segments
        points = [(x1, y1), (x2, y2)]
        for x3, y3, x4, y4, _ in collinear_segments:
            points.append((x3, y3))
            points.append((x4, y4))
        
        # If horizontal, sort by x-coordinate
        is_horizontal = abs(y2 - y1) < abs(x2 - x1)
        if is_horizontal:
            points.sort(key=lambda p: p[0])
            # Take first and last point
            new_x1, new_y1 = points[0]
            new_x2, new_y2 = points[-1]
        else:
            # If vertical, sort by y-coordinate
            points.sort(key=lambda p: p[1])
            new_x1, new_y1 = points[0]
            new_x2, new_y2 = points[-1]
        
        # Calculate new length
        new_length = np.sqrt((new_x2 - new_x1) ** 2 + (new_y2 - new_y1) ** 2)
        
        merged_lines.append((new_x1, new_y1, new_x2, new_y2, new_length))
    
    return merged_lines

def remove_grid_lines(image):
    """
    Specialized function to remove grid lines from the floor plan.
    
    Args:
        image (numpy.ndarray): Input image.
    
    Returns:
        numpy.ndarray: Image with grid lines removed.
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Step 1: Detect lines using Hough transform
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    # Step 2: Separate thick and thin lines
    thickness_threshold = 5  # Adjust based on your image
    thick_lines = []
    thin_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Create a mask for this line
            line_mask = np.zeros_like(gray)
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 1)
            
            # Check the line thickness by sampling the original image
            line_segment = cv2.bitwise_and(gray, gray, mask=line_mask)
            non_zero_pixels = np.count_nonzero(line_segment)
            
            # Calculate approximate thickness
            line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            thickness = non_zero_pixels / line_length if line_length > 0 else 0
            
            if thickness > thickness_threshold:
                thick_lines.append(line)
            else:
                thin_lines.append(line)
    
    # Step 3: Create a mask of thin lines (grid lines)
    grid_mask = np.zeros_like(gray)
    for line in thin_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(grid_mask, (x1, y1), (x2, y2), 255, 2)
    
    # Step 4: Remove grid lines from the original image
    grid_removed = cv2.inpaint(image, grid_mask, 3, cv2.INPAINT_TELEA)
    
    return grid_removed

