import cv2
import numpy as np
from src.image_processed.wall_detection import _estimate_wall_thickness
from src.image_processed.wall_detection import detect_walls as _detect_walls
from src.image_processed.door_detection import detect_doors as _detect_doors
from src.image_processed.window_detection import detect_windows
from src.image_processed.room_detection import detect_rooms, calculate_room_dimensions


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
        self.estimated_wall_thickness = _estimate_wall_thickness(binary)
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
    
    # Wrapper methods to call the imported functions with 'self'
    def detect_walls(self):
        """Wrapper method to call the wall detection function"""
        self.walls = _detect_walls(self)
        return self.walls
    
    def detect_doors(self):
        """Wrapper method to call the door detection function"""
        self.doors = _detect_doors(self,self.image,confidence_threshold=1, overlap_threshold=50)
        return self.doors
    
    def detect_windows(self):
        """Wrapper method to call the window detection function"""
        self.windows = detect_windows(self,self.image,confidence_threshold=50, overlap_threshold=50)
        return self.windows
    
    def detect_rooms(self):
        """Wrapper method to call the room detection function"""
        self.rooms = detect_rooms(self)
        return self.rooms
    
    def calculate_room_dimensions(self):
        """Wrapper method to call the room dimensions calculation function"""
        self.room_dimensions = calculate_room_dimensions(self)
        return self.room_dimensions
    
    def _point_to_line_distance(self, px, py, x1, y1, x2, y2):
        """
        Calculate the distance from a point to a line segment.
        
        Args:
            px, py: Point coordinates
            x1, y1, x2, y2: Line segment coordinates
            
        Returns:
            float: Minimum distance from point to line segment
        """
        # Calculate the length of the line segment
        line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # If the line has zero length, return distance to one of the endpoints
        if line_length == 0:
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        
        # Calculate the projection of the point onto the line
        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_length**2)))
        
        # Calculate the closest point on the line segment
        closest_x = x1 + t * (x2 - x1)
        closest_y = y1 + t * (y2 - y1)
        
        # Calculate the distance to the closest point
        return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)