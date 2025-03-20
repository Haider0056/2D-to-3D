import cv2
import numpy as np

def detect_doors(self):
     """
     Detect doors in the floor plan by identifying door symbols (arcs).
     Only keeps doors that are attached to at least one wall.
     Ensures there is no more than one door in a specific area.
     
     Returns:
         list: List of detected door positions [(x, y, width, height, angle), ...]
     """
     if self.processed_image is None:
         self.preprocess()
     
     # Create a copy of the processed image for door detection
     door_img = self.processed_image.copy()
     
     # Parameters for door detection
     min_arc_length = 20  # Minimum length of arc to be considered a door
     max_arc_length = 100  # Maximum length of arc to be considered a door
     
     # Find contours in the processed image
     contours, _ = cv2.findContours(door_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
     
     # Make sure walls are detected first
     if self.walls is None:
         self.detect_walls()  # Assuming you have this method
     
     doors = []
     for contour in contours:
         # Check if contour resembles an arc (door symbol)
         perimeter = cv2.arcLength(contour, False)
         if perimeter < min_arc_length or perimeter > max_arc_length:
             continue
         
         # Get bounding rectangle
         x, y, w, h = cv2.boundingRect(contour)
         
         # Filter by aspect ratio for door symbols
         aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
         if not (0.5 <= aspect_ratio <= 2.5):
             continue
             
         # Check for arc shape using contour approximation
         approx = cv2.approxPolyDP(contour, 0.04 * perimeter, False)
         if len(approx) < 5:  # Arc should have multiple points
             continue
             
         # Calculate convexity - door arcs are typically convex
         hull = cv2.convexHull(contour)
         hull_area = cv2.contourArea(hull)
         contour_area = cv2.contourArea(contour)
         
         # If solid area, not likely a door arc
         if contour_area > 0 and hull_area > 0:
             solidity = contour_area / hull_area
             if solidity > 0.9:  # Door arcs typically have lower solidity
                 continue
         
         # Determine angle based on arc orientation
         # For simplicity, just check if width > height
         angle = 0 if w > h else 90
         
         doors.append((x, y, w, h, angle))
     
     # Additional detection for doors shown as breaks in walls with perpendicular lines
     # Find lines that could be door indicators
     lines = cv2.HoughLinesP(door_img, 1, np.pi/180, 70, minLineLength=40, maxLineGap=1)
     
     if lines is not None:
         for line in lines:
             x1, y1, x2, y2 = line[0]
             length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
             
             # Filter by length - door indicators are usually specific length
             if not (20 <= length <= 60):
                 continue
                 
             # Check if perpendicular to nearby wall
             angle_rad = np.arctan2(y2-y1, x2-x1)
             angle_deg = np.degrees(angle_rad) % 180
             
             # Door swing lines are typically perpendicular to walls
             if not (80 <= angle_deg <= 100 or 0 <= angle_deg <= 10 or 170 <= angle_deg <= 180):
                 continue
                 
             # Add as potential door
             x = min(x1, x2)
             y = min(y1, y2)
             w = abs(x2-x1)
             h = abs(y2-y1)
             angle = 90 if abs(x2-x1) < abs(y2-y1) else 0
             
             doors.append((x, y, w, h, angle))
     
     # Filter doors by wall attachment and apply area constraint
     filtered_doors = []
     
     # Create grid to track door occupancy in specific areas
     # Define area size for door uniqueness constraint
     area_size = 150  # pixels (adjust based on your floor plan scale)
     
     # Create an occupancy map using a sparse approach
     door_areas = {}  # Dictionary to track occupied areas
 
     # First, filter out doors that are not attached to walls
     attached_doors = []
     for door in doors:
         x, y, w, h, angle = door
         
         # Define door boundary points
         door_center = (x + w//2, y + h//2)
         door_points = [
             (x, y),                  # Top-left
             (x + w, y),              # Top-right
             (x + w, y + h),          # Bottom-right
             (x, y + h),              # Bottom-left
             (x + w//2, y),           # Top-middle
             (x + w//2, y + h),       # Bottom-middle
             (x, y + h//2),           # Left-middle
             (x + w, y + h//2),       # Right-middle
             door_center              # Center
         ]
         
         # Check if any of the door points are close to a wall
         attached_to_wall = False
         for point in door_points:
             px, py = point
             for wall in self.walls:
                 x1, y1, x2, y2 = wall
                 
                 # Calculate distance from point to line segment (wall)
                 dist = self._point_to_line_distance(px, py, x1, y1, x2, y2)
                 
                 # If door point is close to a wall, consider it attached
                 if dist < 10:  # Threshold distance in pixels
                     attached_to_wall = True
                     break
             
             if attached_to_wall:
                 break
         
         if attached_to_wall:
             attached_doors.append(door)
     
     # Now, among attached doors, ensure there's only one door per area
     # Sort doors by size (area) in descending order to prioritize larger/clearer doors
     attached_doors.sort(key=lambda d: d[2] * d[3], reverse=True)
     
     for door in attached_doors:
         x, y, w, h, angle = door
         
         # Calculate door center
         center_x = x + w // 2
         center_y = y + h // 2
         
         # Calculate area coordinates (grid cell)
         area_x = center_x // area_size
         area_y = center_y // area_size
         area_key = (area_x, area_y)
         
         # Check if this area already has a door
         if area_key not in door_areas:
             # No door in this area yet, add it
             door_areas[area_key] = door
             filtered_doors.append(door)
     
     self.doors = filtered_doors
     return filtered_doors

