import cv2
import numpy as np

def detect_windows(self):
     """
     Detect windows in the floor plan with improved constraints:
     - Ensures windows are attached to walls
     - Maintains minimum distance between windows (100 pixels)
     - Filters by appropriate size and aspect ratio
     - Avoids overlap with doors
     
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
     
     # Ensure doors are detected first if not already done
     if not hasattr(self, 'doors') or self.doors is None:
         self.detect_doors()
     
     # Remove door regions from window candidates
     if hasattr(self, 'doors') and self.doors:
         # Create a mask of doors with some padding
         door_mask = np.zeros((self.height, self.width), dtype=np.uint8)
         for x, y, w, h, angle in self.doors:
             # Add some padding around doors to avoid detecting windows too close to doors
             padding = 10
             x_pad = max(0, x - padding)
             y_pad = max(0, y - padding)
             w_pad = min(self.width - x_pad, w + 2*padding)
             h_pad = min(self.height - y_pad, h + 2*padding)
             cv2.rectangle(door_mask, (x_pad, y_pad), (x_pad + w_pad, y_pad + h_pad), 255, -1)
         window_candidates = cv2.bitwise_and(window_candidates, cv2.bitwise_not(door_mask))
     
     # Find contours in window candidates
     contours, _ = cv2.findContours(window_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
     
     window_candidates_list = []
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
         if aspect_ratio < 1.5 or aspect_ratio > 5:  # Slightly more permissive upper limit
             continue
             
         # Determine orientation
         if w > h:
             angle = 0
         else:
             angle = 90
             
         window_candidates_list.append((x, y, w, h, angle))
     
     # Now filter windows to ensure they're attached to walls
     attached_windows = []
     for window in window_candidates_list:
         x, y, w, h, angle = window
         
         # Define window boundary points
         window_center = (x + w//2, y + h//2)
         window_points = [
             (x, y),                  # Top-left
             (x + w, y),              # Top-right
             (x + w, y + h),          # Bottom-right
             (x, y + h),              # Bottom-left
             (x + w//2, y),           # Top-middle
             (x + w//2, y + h),       # Bottom-middle
             (x, y + h//2),           # Left-middle
             (x + w, y + h//2),       # Right-middle
             window_center            # Center
         ]
         
         # Check if any of the window points are close to a wall
         attached_to_wall = False
         for point in window_points:
             px, py = point
             for wall in self.walls:
                 x1, y1, x2, y2 = wall
                 
                 # Calculate distance from point to line segment (wall)
                 dist = self._point_to_line_distance(px, py, x1, y1, x2, y2)
                 
                 # If window point is close to a wall, consider it attached
                 if dist < 8:  # Slightly tighter than doors (8 pixels vs 10)
                     attached_to_wall = True
                     break
             
             if attached_to_wall:
                 break
         
         if attached_to_wall:
             attached_windows.append(window)
     
     # Sort windows by size (area) in descending order to prioritize clearer windows
     attached_windows.sort(key=lambda w: w[2] * w[3], reverse=True)
     
     # Further filter windows to maintain minimum distance of 100 pixels between them
     final_windows = []
     min_distance = 100  # Minimum pixel distance between windows
     
     for window in attached_windows:
         x1, y1, w1, h1, angle1 = window
         center1 = (x1 + w1//2, y1 + h1//2)
         
         # Check distance to already accepted windows
         too_close = False
         for accepted_window in final_windows:
             x2, y2, w2, h2, angle2 = accepted_window
             center2 = (x2 + w2//2, y2 + h2//2)
             
             # Calculate Euclidean distance between centers
             distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
             
             # Check if this window is too close to an already accepted one
             if distance < min_distance:
                 too_close = True
                 break
         
         if not too_close:
             final_windows.append(window)
     
     # Additional quality filter - remove windows that seem inconsistent with the pattern
     if len(final_windows) > 3:  # Only apply if we have enough windows to establish a pattern
         # Calculate average window size
         avg_width = sum(w for _, _, w, h, _ in final_windows) / len(final_windows)
         avg_height = sum(h for _, _, w, h, _ in final_windows) / len(final_windows)
         
         # Filter out windows that deviate too much from average size
         size_filtered_windows = []
         for x, y, w, h, angle in final_windows:
             # Check if size is within reasonable range of average
             if angle == 0:  # Horizontal window
                 if 0.6 * avg_width <= w <= 1.4 * avg_width and 0.6 * avg_height <= h <= 1.4 * avg_height:
                     size_filtered_windows.append((x, y, w, h, angle))
             else:  # Vertical window
                 if 0.6 * avg_height <= w <= 1.4 * avg_height and 0.6 * avg_width <= h <= 1.4 * avg_width:
                     size_filtered_windows.append((x, y, w, h, angle))
         
         # Only use size filtering if it doesn't remove too many windows
         if len(size_filtered_windows) > len(final_windows) * 0.7:
             final_windows = size_filtered_windows
     
     self.windows = final_windows
     return final_windows

