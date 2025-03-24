import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from dotenv import load_dotenv
import os
# Load environment variables from .env
load_dotenv()

# Access the variables
api_key = os.getenv("API_KEY")
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

# def _are_parallel(line1, line2, angle_threshold=5):
#     """
#     Check if two lines are parallel within a given angle threshold.
    
#     Args:
#         line1 (tuple): First line (x1, y1, x2, y2)
#         line2 (tuple): Second line (x1, y1, x2, y2)
#         angle_threshold (float): Maximum angle difference in degrees
        
#     Returns:
#         bool: True if lines are parallel, False otherwise
#     """
#     # Calculate directional vectors
#     dx1 = line1[2] - line1[0]
#     dy1 = line1[3] - line1[1]
#     dx2 = line2[2] - line2[0]
#     dy2 = line2[3] - line2[1]
    
#     # Handle zero-length lines
#     len1 = np.sqrt(dx1**2 + dy1**2)
#     len2 = np.sqrt(dx2**2 + dy2**2)
    
#     if len1 < 1 or len2 < 1:
#         return False
    
#     # Normalize vectors
#     dx1, dy1 = dx1/len1, dy1/len1
#     dx2, dy2 = dx2/len2, dy2/len2
    
#     # Calculate dot product and convert to angle
#     dot_product = abs(dx1*dx2 + dy1*dy2)
#     angle_diff = np.arccos(min(dot_product, 1.0)) * 180 / np.pi
    
#     # If angles are almost parallel or almost perpendicular (within threshold)
#     return angle_diff < angle_threshold or abs(angle_diff - 180) < angle_threshold

# def _calculate_line_distance(line1, line2):
#     """
#     Calculate the average distance between two parallel lines.
    
#     Args:
#         line1 (tuple): First line (x1, y1, x2, y2)
#         line2 (tuple): Second line (x1, y1, x2, y2)
        
#     Returns:
#         float: Average distance between the lines
#     """
#     x1, y1, x2, y2 = line1
#     x3, y3, x4, y4 = line2
    
#     # For horizontal lines, calculate vertical distance
#     if abs(y2 - y1) < abs(x2 - x1):
#         return abs((y1 + y2) / 2 - (y3 + y4) / 2)
#     # For vertical lines, calculate horizontal distance
#     else:
#         return abs((x1 + x2) / 2 - (x3 + x4) / 2)
 
# def detect_walls(self):
#     """
#     Detect walls in the floor plan with improved line detection and merging of parallel lines.
    
#     Returns:
#         list: List of detected wall line segments [(x1, y1, x2, y2), ...]
#     """
#     if self.processed_image is None:
#         self.preprocess()
    
#     # Use Probabilistic Hough Transform with adaptive parameters
#     min_line_length = max(15, self.height / 30)
#     max_line_gap = self.estimated_wall_thickness 
    
#     lines = cv2.HoughLinesP(
#         self.processed_image, 
#         rho=1, 
#         theta=np.pi/180, 
#         threshold=110, 
#         minLineLength=min_line_length, 
#         maxLineGap=max_line_gap
#     )
    
#     if lines is None:
#         self.walls = []
#         return []
    
#     # Extract all line segments
#     line_segments = [line[0] for line in lines]
    
#     # Group line segments by orientation
#     horizontal_lines = []
#     vertical_lines = []
    
#     for x1, y1, x2, y2 in line_segments:
#         # Calculate angle and length
#         angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
#         length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
#         # Filter by angle
#         if angle < 20 or angle > 160:
#             horizontal_lines.append((x1, y1, x2, y2, length))
#         elif angle > 70 and angle < 110:
#             vertical_lines.append((x1, y1, x2, y2, length))
    
#     # Sort by length (longest first)
#     horizontal_lines.sort(key=lambda x: x[4], reverse=True)
#     vertical_lines.sort(key=lambda x: x[4], reverse=True)
    
#     # Merge collinear segments
#     merged_horizontal = _merge_line_segments(self, horizontal_lines)
#     merged_vertical = _merge_line_segments(self, vertical_lines)
    
#     # Extract line segments without length component
#     horizontal_segments = [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in merged_horizontal]
#     vertical_segments = [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in merged_vertical]
    
#     # Merge parallel lines that are close to each other
#     distance_threshold = self.estimated_wall_thickness * 3 
    
#     # Apply the improved parallel line merging
#     merged_horizontal = _merge_parallel_lines(horizontal_segments, distance_threshold)
#     merged_vertical = _merge_parallel_lines(vertical_segments, distance_threshold)
    
#     # Combine all wall lines
#     wall_lines = merged_horizontal + merged_vertical
    
#     # Filter out very short lines that might be noise
#     min_wall_length = max(5, self.estimated_wall_thickness * 3)
#     filtered_walls = []
#     for x1, y1, x2, y2 in wall_lines:
#         length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
#         if length >= min_wall_length:
#             filtered_walls.append((x1, y1, x2, y2))
    
#     self.walls = filtered_walls

#     filtered_walls = fix_wall_junctions(filtered_walls, threshold=max(3, self.estimated_wall_thickness))
#     self.walls = filtered_walls
#     filtered_walls = remove_isolated_walls(filtered_walls, distance_threshold=self.estimated_wall_thickness * 3)
#     return filtered_walls
def detect_walls(self, image_path, confidence_threshold=0.001):
    """
    Detect walls in the floor plan using the Roboflow API.
    Sends raw image with no preprocessing and uses a 2% confidence threshold.
    
    Args:
        image_path (str or numpy.ndarray): Path to the floor plan image or image array.
        confidence_threshold (float): Fixed at 0.02 (2%) regardless of input.
    
    Returns:
        list: List of detected wall line segments [(x1, y1, x2, y2), ...]
    """
    # Override any input threshold with fixed 2%
    confidence_threshold = 0.04
    
    # Initialize the Roboflow client without confidence threshold
    from inference_sdk import InferenceHTTPClient, InferenceConfiguration
    
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=api_key
    )
    
    # Create configuration for inference
    custom_configuration = InferenceConfiguration(
        confidence_threshold=confidence_threshold,
    )
    
    # Handle different input types without preprocessing
    if isinstance(image_path, str):
        # Send the file path directly with custom configuration
        with CLIENT.use_configuration(custom_configuration):
            result = CLIENT.infer(image_path, model_id="wall-detection-xi9ox/2")
    else:
        # For image arrays, save as PNG with no compression temporarily
        # Use PNG format to preserve image quality
        temp_path = "/tmp/temp_wall_image.png"
        cv2.imwrite(temp_path, image_path, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        with CLIENT.use_configuration(custom_configuration):
            result = CLIENT.infer(temp_path, model_id="wall-detection-xi9ox/2")
    
    # Log API response for debugging
    print(f"Number of wall predictions: {len(result.get('predictions', []))}")
    
    # Parse results into wall segments
    wall_segments = []
    for prediction in result.get('predictions', []):
        # Extract bounding box coordinates
        x_center = prediction['x']
        y_center = prediction['y']
        width = prediction['width']
        height = prediction['height']
        
        # Determine if this is a horizontal or vertical wall
        is_horizontal = width > height
        
        # Convert bounding box to line segment
        if is_horizontal:
            x1 = int(x_center - width / 2)
            y1 = int(y_center)
            x2 = int(x_center + width / 2)
            y2 = int(y_center)
        else:  # vertical
            x1 = int(x_center)
            y1 = int(y_center - height / 2)
            x2 = int(x_center)
            y2 = int(y_center + height / 2)
        
        wall_segments.append((x1, y1, x2, y2))
    
    print(f"Wall segments after filtering by confidence: {len(wall_segments)}")
    
    if not wall_segments:
        print("No walls found after confidence filtering")
    
    return wall_segments
# def _merge_parallel_lines(lines, distance_threshold):
#     """
#     Improved method to merge parallel lines that are within a specified distance threshold.
    
#     Args:
#         lines (list): List of line segments [(x1, y1, x2, y2), ...]
#         distance_threshold (float): Maximum distance between lines to merge
        
#     Returns:
#         list: List of merged line segments with integer coordinates
#     """
#     if not lines:
#         return []
    
#     result = []
#     processed = [False] * len(lines)
    
#     # Increase overlap ratio threshold to be more restrictive
#     min_overlap_ratio = 0.2  # Increased from 0.1
    
#     for i in range(len(lines)):
#         if processed[i]:
#             continue
            
#         processed[i] = True
#         current_line = lines[i]
#         merged_lines = [current_line]
        
#         # Get the "direction" of the current line
#         is_horizontal = abs(current_line[3] - current_line[1]) < abs(current_line[2] - current_line[0])
        
#         # Find all parallel lines within distance threshold
#         for j in range(i + 1, len(lines)):
#             if processed[j]:
#                 continue
                
#             other_line = lines[j]
            
#             # Only process lines with the same orientation
#             other_is_horizontal = abs(other_line[3] - other_line[1]) < abs(other_line[2] - other_line[0])
#             if is_horizontal != other_is_horizontal:
#                 continue
            
#             # More strict parallelism check - reduce angle threshold
#             if not _are_parallel(current_line, other_line, angle_threshold=5):  # Reduced from 10
#                 continue
                
#             distance = _calculate_line_distance(current_line, other_line)
            
#             # Check if lines have sufficient overlap before merging
#             has_overlap = False
#             if is_horizontal:
#                 # For horizontal lines, check x-overlap
#                 current_x_range = [min(current_line[0], current_line[2]), max(current_line[0], current_line[2])]
#                 other_x_range = [min(other_line[0], other_line[2]), max(other_line[0], other_line[2])]
                
#                 # Calculate overlap
#                 overlap_start = max(current_x_range[0], other_x_range[0])
#                 overlap_end = min(current_x_range[1], other_x_range[1])
#                 overlap_length = max(0, overlap_end - overlap_start)
                
#                 # Current line length
#                 current_length = current_x_range[1] - current_x_range[0]
#                 other_length = other_x_range[1] - other_x_range[0]
                
#                 # Check if overlap is sufficient - more restrictive condition
#                 if overlap_length > min(current_length, other_length) * min_overlap_ratio:
#                     has_overlap = True
#             else:
#                 # For vertical lines, check y-overlap
#                 current_y_range = [min(current_line[1], current_line[3]), max(current_line[1], current_line[3])]
#                 other_y_range = [min(other_line[1], other_line[3]), max(other_line[1], other_line[3])]
                
#                 # Calculate overlap
#                 overlap_start = max(current_y_range[0], other_y_range[0])
#                 overlap_end = min(current_y_range[1], other_y_range[1])
#                 overlap_length = max(0, overlap_end - overlap_start)
                
#                 # Current line length
#                 current_length = current_y_range[1] - current_y_range[0]
#                 other_length = other_y_range[1] - other_y_range[0]
                
#                 # Check if overlap is sufficient
#                 if overlap_length > min(current_length, other_length) * min_overlap_ratio:
#                     has_overlap = True
            
#             # Only merge if lines are close enough and have sufficient overlap
#             # Use a dynamic distance threshold based on the estimated wall thickness
#             if distance <= distance_threshold and has_overlap:
#                 merged_lines.append(other_line)
#                 processed[j] = True
        
#         # If we found lines to merge
#         if len(merged_lines) > 1:
#             # For horizontal lines
#             if is_horizontal:
#                 # Get all x and y coordinates
#                 all_x = [line[0] for line in merged_lines] + [line[2] for line in merged_lines]
#                 all_y = [line[1] for line in merged_lines] + [line[3] for line in merged_lines]
                
#                 # Calculate weighted average y-coordinate based on line lengths
#                 weighted_y = 0
#                 total_weight = 0
                
#                 for x1, y1, x2, y2 in merged_lines:
#                     length = abs(x2 - x1)
#                     weighted_y += (y1 + y2) / 2 * length
#                     total_weight += length
                
#                 avg_y = weighted_y / total_weight if total_weight > 0 else sum(all_y) / len(all_y)
                
#                 # Use min and max x values with a small adjustment 
#                 # to ensure they properly extend to intersect with connecting walls
#                 min_x = min(all_x) - 1
#                 max_x = max(all_x) + 1
                
#                 # Create the merged line
#                 result.append((int(min_x), int(avg_y), int(max_x), int(avg_y)))
            
#             # For vertical lines
#             else:
#                 # Get all x and y coordinates
#                 all_x = [line[0] for line in merged_lines] + [line[2] for line in merged_lines]
#                 all_y = [line[1] for line in merged_lines] + [line[3] for line in merged_lines]
                
#                 # Calculate weighted average x-coordinate based on line lengths
#                 weighted_x = 0
#                 total_weight = 0
                
#                 for x1, y1, x2, y2 in merged_lines:
#                     length = abs(y2 - y1)
#                     weighted_x += (x1 + x2) / 2 * length
#                     total_weight += length
                
#                 avg_x = weighted_x / total_weight if total_weight > 0 else sum(all_x) / len(all_x)
                
#                 # Use min and max y values with slight adjustment
#                 min_y = min(all_y) - 1
#                 max_y = max(all_y) + 1
                
#                 # Create the merged line
#                 result.append((int(avg_x), int(min_y), int(avg_x), int(max_y)))
#         else:
#             # Just add the current line
#             result.append((int(current_line[0]), int(current_line[1]), 
#                           int(current_line[2]), int(current_line[3])))
    
#     # Additional pass to merge any remaining close parallel lines that might have been missed
#     final_result = []
#     processed = [False] * len(result)
    
#     for i in range(len(result)):
#         if processed[i]:
#             continue
        
#         processed[i] = True
#         line1 = result[i]
#         is_merged = False
        
#         for j in range(len(final_result)):
#             line2 = final_result[j]
            
#             # Check if lines are of same orientation
#             line1_is_horizontal = abs(line1[3] - line1[1]) < abs(line1[2] - line1[0])
#             line2_is_horizontal = abs(line2[3] - line2[1]) < abs(line2[2] - line2[0])
            
#             if line1_is_horizontal != line2_is_horizontal:
#                 continue
                
#             # Check if they're parallel and close
#             if _are_parallel(line1, line2) and _calculate_line_distance(line1, line2) <= distance_threshold:
#                 # Check for overlap
#                 has_overlap = False
                
#                 if line1_is_horizontal:
#                     x_range1 = [min(line1[0], line1[2]), max(line1[0], line1[2])]
#                     x_range2 = [min(line2[0], line2[2]), max(line2[0], line2[2])]
#                     overlap_start = max(x_range1[0], x_range2[0])
#                     overlap_end = min(x_range1[1], x_range2[1])
                    
#                     if overlap_end > overlap_start:
#                         has_overlap = True
                        
#                         # Merge the lines
#                         new_x1 = min(line1[0], line1[2], line2[0], line2[2])
#                         new_x2 = max(line1[0], line1[2], line2[0], line2[2])
#                         new_y = (line1[1] + line1[3] + line2[1] + line2[3]) / 4
                        
#                         # Update the line in final_result
#                         final_result[j] = (int(new_x1), int(new_y), int(new_x2), int(new_y))
#                         is_merged = True
#                         break
#                 else:
#                     y_range1 = [min(line1[1], line1[3]), max(line1[1], line1[3])]
#                     y_range2 = [min(line2[1], line2[3]), max(line2[1], line2[3])]
#                     overlap_start = max(y_range1[0], y_range2[0])
#                     overlap_end = min(y_range1[1], y_range2[1])
                    
#                     if overlap_end > overlap_start:
#                         has_overlap = True
                        
#                         # Merge the lines
#                         new_y1 = min(line1[1], line1[3], line2[1], line2[3])
#                         new_y2 = max(line1[1], line1[3], line2[1], line2[3])
#                         new_x = (line1[0] + line1[2] + line2[0] + line2[2]) / 4
                        
#                         # Update the line in final_result
#                         final_result[j] = (int(new_x), int(new_y1), int(new_x), int(new_y2))
#                         is_merged = True
#                         break
        
#         if not is_merged:
#             final_result.append(line1)
    
#     return final_result
    
# def _merge_line_segments(self, line_segments):
#     """
#     Merge collinear line segments.
    
#     Args:
#         self: The FloorPlanProcessor instance
#         line_segments (list): List of line segments with length [(x1, y1, x2, y2, length), ...]
        
#     Returns:
#         list: List of merged line segments
#     """
#     if not line_segments:
#         return []
    
#     merged_lines = []
    
#     # Determine if two lines are collinear and close
#     def are_collinear(line1, line2):
#         x1, y1, x2, y2, _ = line1
#         x3, y3, x4, y4, _ = line2
        
#         # Check if lines are roughly horizontal or vertical
#         is_horizontal1 = abs(y2 - y1) < abs(x2 - x1)
#         is_horizontal2 = abs(y4 - y3) < abs(x4 - x3)
        
#         if is_horizontal1 != is_horizontal2:
#             return False
        
#         # For horizontal lines, check y-proximity
#         if is_horizontal1:
#             if abs(y1 - y3) > self.estimated_wall_thickness or abs(y2 - y4) > self.estimated_wall_thickness:
#                 return False
            
#             # Check if x-ranges overlap or are close
#             if max(x1, x2) < min(x3, x4) - self.estimated_wall_thickness * 2:
#                 return False
#             if min(x1, x2) > max(x3, x4) + self.estimated_wall_thickness * 2:
#                 return False
            
#             return True
        
#         # For vertical lines, check x-proximity
#         else:
#             if abs(x1 - x3) > self.estimated_wall_thickness or abs(x2 - x4) > self.estimated_wall_thickness:
#                 return False
            
#             # Check if y-ranges overlap or are close
#             if max(y1, y2) < min(y3, y4) - self.estimated_wall_thickness * 2:
#                 return False
#             if min(y1, y2) > max(y3, y4) + self.estimated_wall_thickness * 2:
#                 return False
            
#             return True
    
#     # Try to merge each line
#     remaining = line_segments.copy()
    
#     while remaining:
#         current = remaining.pop(0)
#         x1, y1, x2, y2, _ = current
        
#         # Find all collinear segments
#         collinear_indices = []
#         for i, other in enumerate(remaining):
#             if are_collinear(current, other):
#                 collinear_indices.append(i)
        
#         # Extract collinear segments
#         collinear_segments = [remaining[i] for i in sorted(collinear_indices, reverse=True)]
#         for i in sorted(collinear_indices, reverse=True):
#             remaining.pop(i)
        
#         # Merge current with all collinear segments
#         points = [(x1, y1), (x2, y2)]
#         for x3, y3, x4, y4, _ in collinear_segments:
#             points.append((x3, y3))
#             points.append((x4, y4))
        
#         # If horizontal, sort by x-coordinate
#         is_horizontal = abs(y2 - y1) < abs(x2 - x1)
#         if is_horizontal:
#             points.sort(key=lambda p: p[0])
#             # Take first and last point
#             new_x1, new_y1 = points[0]
#             new_x2, new_y2 = points[-1]
#         else:
#             # If vertical, sort by y-coordinate
#             points.sort(key=lambda p: p[1])
#             new_x1, new_y1 = points[0]
#             new_x2, new_y2 = points[-1]
        
#         # Calculate new length
#         new_length = np.sqrt((new_x2 - new_x1) ** 2 + (new_y2 - new_y1) ** 2)
        
#         merged_lines.append((new_x1, new_y1, new_x2, new_y2, new_length))
    
#     return merged_lines

# def remove_grid_lines(image):
#     """Enhanced grid line removal function"""
#     if len(image.shape) == 3:
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     else:
#         gray = image.copy()
    
#     # Apply adaptive thresholding to better isolate grid lines
#     binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                                   cv2.THRESH_BINARY_INV, 11, 2)
    
#     # Detect horizontal and vertical lines specifically
#     horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
#     vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    
#     # Detect horizontal grid lines
#     horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
#     # Detect vertical grid lines
#     vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
#     # Combine grid lines
#     grid_lines = cv2.bitwise_or(horizontal_lines, vertical_lines)
    
#     # Dilate to ensure all grid pixels are covered
#     grid_mask = cv2.dilate(grid_lines, np.ones((3,3), np.uint8), iterations=1)
    
#     # Remove grid lines using inpainting
#     grid_removed = cv2.inpaint(image, grid_mask, 3, cv2.INPAINT_TELEA)
    
#     return grid_removed
# def fix_wall_junctions(walls, threshold=50):
#     """
#     Enhanced function to fix wall junctions by extending/trimming walls and connecting nearby incomplete segments.
    
#     Args:
#         walls (list): List of wall line segments [(x1, y1, x2, y2), ...]
#         threshold (int): Distance threshold for considering intersections
        
#     Returns:
#         list: List of fixed wall segments
#     """
#     if not walls:
#         return []
    
#     # STEP 1: Connect nearby incomplete segments that might form a single wall
#     connected_walls = []
#     processed = [False] * len(walls)
    
#     for i in range(len(walls)):
#         if processed[i]:
#             continue
            
#         processed[i] = True
#         current_wall = list(walls[i])  # Convert to list for easier modification
        
#         # Check if current wall is too short - potentially incomplete
#         wall_length = np.sqrt((current_wall[2] - current_wall[0])**2 + (current_wall[3] - current_wall[1])**2)
        
#         # Try to extend short walls by connecting with nearby segments
#         while wall_length < 50:  # Minimum expected wall length
#             # Find closest unprocessed wall segment to either end
#             best_match = None
#             best_distance = float('inf')
#             best_end = None
            
#             for j in range(len(walls)):
#                 if processed[j] or i == j:
#                     continue
                
#                 # Check both ends of current wall with both ends of candidate wall
#                 endpoints1 = [(current_wall[0], current_wall[1]), (current_wall[2], current_wall[3])]
#                 endpoints2 = [(walls[j][0], walls[j][1]), (walls[j][2], walls[j][3])]
                
#                 for end_idx1, (ex1, ey1) in enumerate(endpoints1):
#                     for end_idx2, (ex2, ey2) in enumerate(endpoints2):
#                         dist = np.sqrt((ex2 - ex1)**2 + (ey2 - ey1)**2)
                        
#                         # Check if they're close enough and have similar orientation
#                         if dist < 20:  # Higher threshold for connecting incomplete segments
#                             # Check if they have similar orientation
#                             if end_idx1 == 0:  # First endpoint of current wall
#                                 dx1 = current_wall[2] - current_wall[0]
#                                 dy1 = current_wall[3] - current_wall[1]
#                             else:  # Second endpoint of current wall
#                                 dx1 = current_wall[0] - current_wall[2]
#                                 dy1 = current_wall[1] - current_wall[3]
                                
#                             if end_idx2 == 0:  # First endpoint of candidate wall
#                                 dx2 = walls[j][2] - walls[j][0]
#                                 dy2 = walls[j][3] - walls[j][1]
#                             else:  # Second endpoint of candidate wall
#                                 dx2 = walls[j][0] - walls[j][2]
#                                 dy2 = walls[j][1] - walls[j][3]
                            
#                             # Normalize direction vectors
#                             len1 = np.sqrt(dx1**2 + dy1**2)
#                             len2 = np.sqrt(dx2**2 + dy2**2)
                            
#                             if len1 > 0 and len2 > 0:
#                                 dx1, dy1 = dx1/len1, dy1/len1
#                                 dx2, dy2 = dx2/len2, dy2/len2
                                
#                                 # Check if directions are similar (dot product close to 1)
#                                 dot_product = dx1*dx2 + dy1*dy2
#                                 if dot_product > 0.7:  # Approximately within 45 degrees
#                                     if dist < best_distance:
#                                         best_distance = dist
#                                         best_match = j
#                                         best_end = (end_idx1, end_idx2)
            
#             # If no good match found, break the loop
#             if best_match is None:
#                 break
                
#             # Connect the walls
#             other_wall = list(walls[best_match])
#             processed[best_match] = True
            
#             # Update current wall based on connection point
#             if best_end[0] == 0:  # First endpoint of current wall
#                 if best_end[1] == 0:  # First endpoint of other wall
#                     # Reverse other wall and connect to start of current
#                     current_wall[0] = other_wall[2]
#                     current_wall[1] = other_wall[3]
#                 else:  # Second endpoint of other wall
#                     current_wall[0] = other_wall[0]
#                     current_wall[1] = other_wall[1]
#             else:  # Second endpoint of current wall
#                 if best_end[1] == 0:  # First endpoint of other wall
#                     current_wall[2] = other_wall[2]
#                     current_wall[3] = other_wall[3]
#                 else:  # Second endpoint of other wall
#                     current_wall[2] = other_wall[0]
#                     current_wall[3] = other_wall[1]
            
#             # Recalculate wall length
#             wall_length = np.sqrt((current_wall[2] - current_wall[0])**2 + (current_wall[3] - current_wall[1])**2)
        
#         connected_walls.append(tuple(current_wall))
    
#     # Add remaining unprocessed walls
#     for i in range(len(walls)):
#         if not processed[i]:
#             connected_walls.append(walls[i])
    
#     # STEP 2: Improve junction handling
#     # First, find all wall endpoints
#     endpoints = []
#     for i, (x1, y1, x2, y2) in enumerate(connected_walls):
#         endpoints.append((int(x1), int(y1), i, 'start'))
#         endpoints.append((int(x2), int(y2), i, 'end'))
    
#     # Find clusters of nearby endpoints (potential junctions)
#     junctions = []
#     processed = [False] * len(endpoints)
    
#     for i in range(len(endpoints)):
#         if processed[i]:
#             continue
            
#         x1, y1, wall_idx1, end_type1 = endpoints[i]
#         junction_points = [(x1, y1, wall_idx1, end_type1)]
#         processed[i] = True
        
#         for j in range(i + 1, len(endpoints)):
#             if processed[j]:
#                 continue
                
#             x2, y2, wall_idx2, end_type2 = endpoints[j]
#             if np.sqrt((x2 - x1)**2 + (y2 - y1)**2) <= threshold:
#                 junction_points.append((x2, y2, wall_idx2, end_type2))
#                 processed[j] = True
        
#         if len(junction_points) > 1:
#             # Calculate average position for the junction
#             avg_x = int(sum(p[0] for p in junction_points) / len(junction_points))
#             avg_y = int(sum(p[1] for p in junction_points) / len(junction_points))
#             junctions.append((avg_x, avg_y, [p[2] for p in junction_points], [p[3] for p in junction_points]))
    
#     # STEP 3: Adjust walls based on detected junctions
#     adjusted_walls = []
#     for i, (x1, y1, x2, y2) in enumerate(connected_walls):
#         new_x1, new_y1 = x1, y1
#         new_x2, new_y2 = x2, y2
        
#         # Check if endpoints need adjustment based on junction points
#         for jx, jy, j_wall_indices, j_end_types in junctions:
#             if i in j_wall_indices:
#                 idx = j_wall_indices.index(i)
#                 end_type = j_end_types[idx]
                
#                 if end_type == 'start':
#                     new_x1, new_y1 = jx, jy
#                 elif end_type == 'end':
#                     new_x2, new_y2 = jx, jy
        
#         # Only add the wall if it still has a significant length
#         length = np.sqrt((new_x2 - new_x1)**2 + (new_y2 - new_y1)**2)
#         if length > threshold:
#             adjusted_walls.append((int(new_x1), int(new_y1), int(new_x2), int(new_y2)))
    
#     # STEP 4: Find and fix wall intersections
#     fixed_walls = []
#     for wall in adjusted_walls:
#         x1, y1, x2, y2 = wall
#         segments = [(x1, y1, x2, y2)]
        
#         for other_wall in adjusted_walls:
#             if wall == other_wall:
#                 continue
                
#             ox1, oy1, ox2, oy2 = other_wall
            
#             # Check if this is a vertical wall crossing a horizontal wall
#             wall_is_vertical = abs(x2 - x1) < abs(y2 - y1)
#             other_is_horizontal = abs(oy2 - oy1) < abs(ox2 - ox1)
            
#             if wall_is_vertical and other_is_horizontal:
#                 # Compute wall midpoints
#                 wall_x = (x1 + x2) / 2
#                 other_y = (oy1 + oy2) / 2
                
#                 # Check if they cross
#                 if (min(ox1, ox2) <= wall_x <= max(ox1, ox2)) and (min(y1, y2) <= other_y <= max(y1, y2)):
#                     # Split the vertical wall at the intersection
#                     new_segments = []
#                     for sx1, sy1, sx2, sy2 in segments:
#                         if (min(sy1, sy2) <= other_y <= max(sy1, sy2)) and abs(sx1 - wall_x) < threshold:
#                             if abs(sy1 - other_y) > threshold:
#                                 new_segments.append((sx1, sy1, sx1, int(other_y)))
#                             if abs(sy2 - other_y) > threshold:
#                                 new_segments.append((sx1, int(other_y), sx2, sy2))
#                         else:
#                             new_segments.append((sx1, sy1, sx2, sy2))
#                     segments = new_segments
        
#         fixed_walls.extend(segments)
    
#     # STEP 5: Remove duplicate walls
#     unique_walls = []
#     for wall in fixed_walls:
#         x1, y1, x2, y2 = wall
        
#         # Skip very short walls
#         if np.sqrt((x2 - x1)**2 + (y2 - y1)**2) <= threshold:
#             continue
            
#         # Normalize wall direction (ensure x1,y1 is the leftmost or topmost point)
#         if (x1 > x2) or (x1 == x2 and y1 > y2):
#             x1, y1, x2, y2 = x2, y2, x1, y1
        
#         is_duplicate = False
#         for i, (ux1, uy1, ux2, uy2) in enumerate(unique_walls):
#             # Check if walls are very similar
#             dist1 = np.sqrt((ux1 - x1)**2 + (uy1 - y1)**2)
#             dist2 = np.sqrt((ux2 - x2)**2 + (uy2 - y2)**2)
            
#             if dist1 < threshold and dist2 < threshold:
#                 is_duplicate = True
#                 break
        
#         if not is_duplicate:
#             unique_walls.append((x1, y1, x2, y2))
    
#     # STEP 6: Final check for T-junctions and crosses
#     result = []
#     for wall in unique_walls:
#         x1, y1, x2, y2 = wall
#         is_horizontal = abs(y2 - y1) < abs(x2 - x1)
        
#         segments_to_check = [(x1, y1, x2, y2)]
#         final_segments = []
        
#         for other_wall in unique_walls:
#             if wall == other_wall:
#                 continue
                
#             ox1, oy1, ox2, oy2 = other_wall
#             other_is_horizontal = abs(oy2 - oy1) < abs(ox2 - ox1)
            
#             # Only check perpendicular walls
#             if is_horizontal == other_is_horizontal:
#                 continue
                
#             for seg_idx, (sx1, sy1, sx2, sy2) in enumerate(segments_to_check):
#                 should_split = False
#                 split_point = None
                
#                 if is_horizontal:  # Current segment is horizontal
#                     avg_y = (sy1 + sy2) / 2
#                     avg_x = (ox1 + ox2) / 2
                    
#                     if (min(sx1, sx2) < avg_x < max(sx1, sx2)) and (min(oy1, oy2) < avg_y < max(oy1, oy2)):
#                         should_split = True
#                         split_point = (int(avg_x), int(avg_y))
#                 else:  # Current segment is vertical
#                     avg_x = (sx1 + sx2) / 2
#                     avg_y = (oy1 + oy2) / 2
                    
#                     if (min(sy1, sy2) < avg_y < max(sy1, sy2)) and (min(ox1, ox2) < avg_x < max(ox1, ox2)):
#                         should_split = True
#                         split_point = (int(avg_x), int(avg_y))
                
#                 if should_split and split_point:
#                     # Replace this segment with two segments
#                     if abs(sx1 - split_point[0]) > threshold or abs(sy1 - split_point[1]) > threshold:
#                         final_segments.append((sx1, sy1, split_point[0], split_point[1]))
#                     if abs(sx2 - split_point[0]) > threshold or abs(sy2 - split_point[1]) > threshold:
#                         final_segments.append((split_point[0], split_point[1], sx2, sy2))
#                 else:
#                     final_segments.append((sx1, sy1, sx2, sy2))
                    
#                 segments_to_check = final_segments
#                 final_segments = []
        
#         result.extend(segments_to_check)
    
#     # Ensure all coordinates are integers
#     final_result = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in result]
    
#     return final_result
# def remove_isolated_walls(walls, distance_threshold=30):
#     """Remove walls that are not connected to any other walls"""
#     if not walls:
#         return []
        
#     result = []
#     for i, wall1 in enumerate(walls):
#         x1, y1, x2, y2 = wall1
        
#         # Check if this wall is connected to any other wall
#         is_connected = False
#         for j, wall2 in enumerate(walls):
#             if i == j:
#                 continue
                
#             x3, y3, x4, y4 = wall2
#             # Check if any endpoint of wall1 is close to any endpoint of wall2
#             dist1 = min(
#                 np.sqrt((x1-x3)**2 + (y1-y3)**2),
#                 np.sqrt((x1-x4)**2 + (y1-y4)**2)
#             )
#             dist2 = min(
#                 np.sqrt((x2-x3)**2 + (y2-y3)**2),
#                 np.sqrt((x2-x4)**2 + (y2-y4)**2)
#             )
            
#             if dist1 < distance_threshold or dist2 < distance_threshold:
#                 is_connected = True
#                 break
                
#         if is_connected:
#             result.append(wall1)
            
#     return result



