import cv2
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='floor_plan_debug.log',
    filemode='w'
)
logger = logging.getLogger('FloorPlanProcessor')
class FloorPlanProcessor:
    def __init__(self, image_path):
        """
        Initialize the floor plan processor with the path to the image.
        
        Args:
            image_path (str): Path to the floor plan image
        """
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Get image dimensions
        self.height, self.width = self.image.shape[:2]
        
        # Initialize class variables
        self.processed_image = None
        self.walls = None
        self.doors = None
        self.windows = None
        self.rooms = None
        self.room_dimensions = None
        self.scale_factor = None
        
        # Wall thickness estimation (in pixels)
        self.estimated_wall_thickness = None
    
    def preprocess(self):
        """
        Preprocess the image to enhance features for detection.
        """
        # Create a copy of the original image
        original = self.image.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(filtered)
        
        # Apply adaptive thresholding with different parameters
        binary = cv2.adaptiveThreshold(
            equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 21, 5
        )
        
        # Estimate wall thickness
        self.estimated_wall_thickness = self._estimate_wall_thickness(binary)
        print(f"Estimated wall thickness: {self.estimated_wall_thickness} pixels")
        
        # Create structural element for morphological operations
        kernel_size = max(3, int(self.estimated_wall_thickness / 3))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Close small gaps in walls
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Remove small noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
        
        self.processed_image = opened
        return self.processed_image
    
    def _estimate_wall_thickness(self, binary_image):
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
    def _are_parallel(self, line1, line2, angle_threshold=10):
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

    def _calculate_line_distance(self, line1, line2):
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
     merged_horizontal = self._merge_line_segments(horizontal_lines)
     merged_vertical = self._merge_line_segments(vertical_lines)
     
     # Extract line segments without length component
     horizontal_segments = [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in merged_horizontal]
     vertical_segments = [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in merged_vertical]
     
     # Merge parallel lines that are close to each other
     distance_threshold = self.estimated_wall_thickness * 4.25  
     
     # Apply the improved parallel line merging
     merged_horizontal = self._merge_parallel_lines(horizontal_segments, distance_threshold)
     merged_vertical = self._merge_parallel_lines(vertical_segments, distance_threshold)
     
     # Combine all wall lines
     wall_lines = merged_horizontal + merged_vertical
     
     # Filter out very short lines that might be noise
     min_wall_length = max(15, self.estimated_wall_thickness * 3)
     filtered_walls = []
     for x1, y1, x2, y2 in wall_lines:
         length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
         if length >= min_wall_length:
             filtered_walls.append((x1, y1, x2, y2))
     
     self.walls = filtered_walls
     return filtered_walls

    def _merge_parallel_lines(self, lines, distance_threshold):
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
              
              if self._are_parallel(current_line, other_line):
                  distance = self._calculate_line_distance(current_line, other_line)
                  
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
    
    def detect_doors(self):
        """
        Detect doors in the floor plan using template matching and geometric analysis.
        
        Returns:
            list: List of detected door positions [(x, y, width, height, angle), ...]
        """
        if self.processed_image is None:
            self.preprocess()
        
        if self.walls is None:
            self.detect_walls()
        
        # Approach 1: Look for gaps in walls that have a specific width
        doors = []
        estimated_door_width = self.estimated_wall_thickness * 3  # Approximate door width
        
        # Create a binary image with only the walls
        wall_img = np.zeros((self.height, self.width), dtype=np.uint8)
        for x1, y1, x2, y2 in self.walls:
            cv2.line(wall_img, (x1, y1), (x2, y2), 255, int(self.estimated_wall_thickness))
        
        # Find gaps in walls that could be doors
        # First, dilate the wall image slightly to close small gaps
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(wall_img, kernel, iterations=2)
        
        # Invert to find potential openings
        inverted = cv2.bitwise_not(dilated)
        
        # Label connected components (potential rooms and outside area)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted, connectivity=4)
        
        # Find boundaries between components (potential doors)
        boundaries = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Expand walls and find intersections with the processed image
        expanded_walls = cv2.dilate(wall_img, kernel, iterations=3)
        door_candidates = cv2.bitwise_and(expanded_walls, self.processed_image)
        
        # Find contours in door candidates
        contours, _ = cv2.findContours(door_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < estimated_door_width * self.estimated_wall_thickness * 0.5:
                continue
            if area > estimated_door_width * self.estimated_wall_thickness * 5:
                continue
                
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio
            aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
            if aspect_ratio < 1.5 or aspect_ratio > 6:
                continue
                
            # Determine orientation
            angle = 0
            if h > w:
                angle = 90
                
            doors.append((x, y, w, h, angle))
        
        self.doors = doors
        return doors
    
    def detect_windows(self):
     """
     Detect windows in the floor plan.
     
     Returns:
         list: List of detected window positions [(x, y, width, height, angle), ...]
     """
     if self.processed_image is None:
         self.preprocess()
         
     if self.walls is None:
         self.detect_walls()
         
     # Check if walls were found
     if not self.walls:
         return []
             
     windows = []
     
     # Define estimated window width based on wall thickness
     # Windows are typically narrower than doors
     estimated_window_width = self.estimated_wall_thickness * 2
     
     # Create a binary image with only the walls
     wall_img = np.zeros((self.height, self.width), dtype=np.uint8)
     for x1, y1, x2, y2 in self.walls:
         cv2.line(wall_img, (x1, y1), (x2, y2), 255, int(self.estimated_wall_thickness))
     
     # Dilate walls slightly to connect nearby components
     kernel = np.ones((3, 3), np.uint8)
     dilated = cv2.dilate(wall_img, kernel, iterations=1)
     
     # Find potential window segments in walls
     window_candidates = cv2.bitwise_and(dilated, self.processed_image)
     
     # Remove door candidates if available
     if hasattr(self, 'door_candidates') and self.door_candidates is not None:
         window_candidates = cv2.bitwise_xor(window_candidates, self.door_candidates)
     # Alternatively, if you detect doors before windows and store the results:
     elif hasattr(self, 'doors') and self.doors:
         # Create a mask of doors
         door_mask = np.zeros((self.height, self.width), dtype=np.uint8)
         for x, y, w, h, angle in self.doors:
             # Draw a simple rectangle instead of a rotated one for safety
             cv2.rectangle(door_mask, (x, y), (x + w, y + h), 255, -1)
         window_candidates = cv2.bitwise_and(window_candidates, cv2.bitwise_not(door_mask))
     
     # Find contours in window candidates
     contours, _ = cv2.findContours(window_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
     
     for contour in contours:
         # Filter by area
         area = cv2.contourArea(contour)
         if area < estimated_window_width * self.estimated_wall_thickness * 0.3:
             continue
         if area > estimated_window_width * self.estimated_wall_thickness * 3:
             continue
         
         # Get bounding box
         x, y, w, h = cv2.boundingRect(contour)
         
         # Filter by aspect ratio
         aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
         if aspect_ratio < 1.5 or aspect_ratio > 4:
             continue
             
         # Determine orientation
         if w > h:
             angle = 0
         else:
             angle = 90
             
         windows.append((x, y, w, h, angle))
     
     self.windows = windows
     return windows

    def detect_rooms(self):
     """
     Detect rooms in the floor plan using a balanced approach to handle wall boundaries.
     
     Returns:
         list: List of room contours
     """
     # Convert to grayscale if not already
     if len(self.image.shape) == 3:
         gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
     else:
         gray = self.image.copy()
     
     # Apply threshold to get binary image
     _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
     
     # Calculate target wall thickness for better morphological operations
     target_thickness = self.estimated_wall_thickness if hasattr(self, 'estimated_wall_thickness') and self.estimated_wall_thickness else 5
     
     # Step 1: Remove noise and small details
     kernel_open = np.ones((3, 3), np.uint8)
     opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=2)
     
     # Step 2: Close small gaps in walls with a controlled kernel size
     kernel_close = np.ones((target_thickness, target_thickness), np.uint8)
     closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=1)
     
     # Step 3: Dilate slightly to connect nearby walls without over-expanding
     kernel_dilate = np.ones((target_thickness, target_thickness), np.uint8)
     dilated = cv2.dilate(closed, kernel_dilate, iterations=1)
     
     # Step 4: Apply controlled erosion to restore reasonable wall thickness
     kernel_erode = np.ones((target_thickness - 2, target_thickness - 2), np.uint8)
     eroded = cv2.erode(dilated, kernel_erode, iterations=1)
     
     # Step 5: Find the inverse (rooms are white, walls are black)
     inverted = cv2.bitwise_not(eroded)
     
     # Step 6: Find contours - these will be potential rooms
     contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
     
     # Filter contours to find rooms
     min_room_size = (self.width * self.height) * 0.01  # 1% of image area
     max_room_size = (self.width * self.height) * 0.5   # 50% of image area
     room_contours = []
     
     for contour in contours:
         area = cv2.contourArea(contour)
         # Skip components that are too small or too large
         if area < min_room_size or area > max_room_size:
             continue
         
         # Check if this component touches the image border
         x, y, w, h = cv2.boundingRect(contour)
         border_touch = (x <= 10 or y <= 10 or 
                        x + w >= self.width - 10 or 
                        y + h >= self.height - 10)
         
         # Skip components that touch the border (likely outside areas)
         if border_touch:
             continue
             
         # Simplify the contour with appropriate epsilon value
         epsilon = 0.01 * cv2.arcLength(contour, True)
         approx = cv2.approxPolyDP(contour, epsilon, True)
         room_contours.append(approx)
     
     # Create rectangular approximations if needed
     rectangular_contours = []
     for contour in room_contours:
         # Get bounding rectangle
         x, y, w, h = cv2.boundingRect(contour)
         
         # Create a rectangular contour
         rect_contour = np.array([
             [[x, y]],
             [[x+w, y]],
             [[x+w, y+h]],
             [[x, y+h]]
         ], dtype=np.int32)
         
         rectangular_contours.append(rect_contour)
     
     # Store both types of contours
     self.rooms = room_contours
     self.rectangular_rooms = rectangular_contours
     

     
     return room_contours
    
 
    def visualize_rooms(self, output_path=None, rectangular=False):
     """
     Visualize the detected rooms.
     
     Args:
         output_path (str, optional): Path to save the visualization.
         rectangular (bool): Whether to use rectangular approximations.
     
     Returns:
         numpy.ndarray: The visualization image.
     """
     if self.rooms is None:
         self.detect_rooms()
     
     # Create a copy of the original image for visualization
     vis_image = self.image.copy()
     
     # Choose which contours to draw
     contours_to_draw = self.rectangular_rooms if rectangular and hasattr(self, 'rectangular_rooms') else self.rooms
     
     # Draw the room contours
     cv2.drawContours(vis_image, contours_to_draw, -1, (0, 255, 0), 2)
     
     # Draw room numbers and areas
     for i, contour in enumerate(contours_to_draw):
         # Find the center of the room
         M = cv2.moments(contour)
         if M["m00"] != 0:
             cx = int(M["m10"] / M["m00"])
             cy = int(M["m01"] / M["m00"])
         else:
             # If the contour has zero area, use its first point
             cx, cy = contour[0][0]
         
         # Calculate area
         area_pixels = cv2.contourArea(contour)
         
         # Draw room number and area
         cv2.putText(vis_image, f"Room {i+1}: {area_pixels} px", (cx-40, cy), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
     
     # Save the visualization if an output path is provided
     if output_path:
         cv2.imwrite(output_path, vis_image)
     
     return vis_image

    def remove_grid_lines(self, image):
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
    def calculate_room_dimensions(self):
        """
        Calculate the dimensions of all rooms based on detected walls and rooms.
        
        Returns:
            dict: Room dimensions and layout information
        """
        if self.walls is None:
            self.detect_walls()
            
        if self.rooms is None:
            self.detect_rooms()
        
        # Calculate overall dimensions
        all_x = []
        all_y = []
        
        for wall in self.walls:
            x1, y1, x2, y2 = wall
            all_x.extend([x1, x2])
            all_y.extend([y1, y2])
        
        if not all_x or not all_y:
            return None
        
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        width_pixels = max_x - min_x
        height_pixels = max_y - min_y
        
        # Assuming a standard door width of 0.8 meters for scaling
        # Look for doors to estimate scale
        if self.doors and len(self.doors) > 0:
            door_widths = []
            for x, y, w, h, angle in self.doors:
                if angle == 0:  # Horizontal door
                    door_widths.append(w)
                else:  # Vertical door
                    door_widths.append(h)
            
            avg_door_width_pixels = sum(door_widths) / len(door_widths)
            scale_factor = 1.2 / avg_door_width_pixels  # meters per pixel
        else:
            # If no doors detected, use a standard scale
            # Assuming a typical room width of 4 meters
            scale_factor = 6.0 / width_pixels
        
        self.scale_factor = scale_factor
        
        # Calculate dimensions for each room
        room_dimensions = []
        
        for room_contour in self.rooms:
            # Calculate area
            area_pixels = cv2.contourArea(room_contour)
            area_meters = area_pixels * (scale_factor ** 2)
            
            # Calculate perimeter
            perimeter_pixels = cv2.arcLength(room_contour, True)
            perimeter_meters = perimeter_pixels * scale_factor
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(room_contour)
            width_meters = w * scale_factor
            height_meters = h * scale_factor
            
            # Calculate centroid
            M = cv2.moments(room_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2
            
            room_dimensions.append({
                "area_pixels": area_pixels,
                "area_meters": area_meters,
                "perimeter_pixels": perimeter_pixels,
                "perimeter_meters": perimeter_meters,
                "width_pixels": w,
                "height_pixels": h,
                "width_meters": width_meters,
                "height_meters": height_meters,
                "centroid": (cx, cy),
                "contour": room_contour
            })
        
        dimensions = {
            "overall_width_pixels": width_pixels,
            "overall_height_pixels": height_pixels,
            "overall_width_meters": width_pixels * scale_factor,
            "overall_height_meters": height_pixels * scale_factor,
            "min_x": min_x,
            "min_y": min_y,
            "max_x": max_x,
            "max_y": max_y,
            "scale_factor": scale_factor,
            "rooms": room_dimensions
        }
        
        self.room_dimensions = dimensions
        return dimensions
    
    def process(self):
        """
        Process the floor plan image and extract all features.
        
        Returns:
            dict: Dictionary containing all extracted features
        """
        self.preprocess()
        self.detect_walls()
        self.detect_doors()
        self.detect_windows()
        self.detect_rooms()
        self.calculate_room_dimensions()
        
        return {
            'walls': self.walls,
            'doors': self.doors,
            'windows': self.windows,
            'rooms': self.rooms,
            'dimensions': self.room_dimensions,
            'processed_image': self.processed_image
        }
    
    def visualize_detection(self, output_path):
        """
        Visualize the detected features on the original image.
        
        Args:
            output_path (str): Path to save the visualization
        """
        # Create a copy of the original image
        visualization = self.image.copy()
        
        # Draw rooms
        if self.rooms is not None:
            for i, room in enumerate(self.rooms):
                # Generate a random color for each room
                color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                
                # Draw the room contour
                cv2.drawContours(visualization, [room], 0, color, 2)
                
                # Label the room
                if self.room_dimensions is not None and i < len(self.room_dimensions["rooms"]):
                    room_data = self.room_dimensions["rooms"][i]
                    cx, cy = room_data["centroid"]
                    area = room_data["area_meters"]
                    cv2.putText(visualization, f"{area:.1f} m²", (cx, cy), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw walls
        if self.walls is not None:
            for x1, y1, x2, y2 in self.walls:
                cv2.line(visualization, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Draw doors
        if self.doors is not None:
            for x, y, w, h, angle in self.doors:
                cv2.rectangle(visualization, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(visualization, "Door", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw windows
        if self.windows is not None:
            for x, y, w, h, angle in self.windows:
                cv2.rectangle(visualization, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(visualization, "Window", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Add scale information
        if self.room_dimensions is not None:
            scale_text = f"Scale: 1 pixel = {self.room_dimensions['scale_factor']:.5f} meters"
            cv2.putText(visualization, scale_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Save the visualization
        cv2.imwrite(output_path, visualization)
        
        return visualization