import cv2
import numpy as np
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
            scale_factor = 3.0 / avg_door_width_pixels  # meters per pixel
        else:
            # If no doors detected, use a standard scale
            # Assuming a typical room width of 4 meters
            scale_factor = 9.5 / width_pixels
        
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
