import cv2
import numpy as np

def remove_grid_lines():
    # Load image
    image_path = "example.jpg"
    src = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if src is None:
        print("Could not open or find the image!")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    # Apply Otsu's thresholding for better binarization
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Save binary image
    cv2.imwrite("binary_output.jpg", bw)
    
    # Extract horizontal and vertical lines
    horizontal = bw.copy()
    vertical = bw.copy()
    
    # Define horizontal structure
    horizontal_size = max(10, horizontal.shape[1] // 40)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    
    # Define vertical structure
    vertical_size = max(10, vertical.shape[0] // 40)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
    
    # Combine both masks to remove grid lines
    grid = cv2.add(horizontal, vertical)
    mask = cv2.bitwise_not(grid)
    cleaned = cv2.bitwise_and(bw, mask)
    
    # Save the grid mask and cleaned image
    cv2.imwrite("grid_mask.jpg", grid)
    cv2.imwrite("grid_removed.jpg", cleaned)

    # Use inpainting to fill the removed grid areas
    inpainted = cv2.inpaint(src, grid, 3, cv2.INPAINT_TELEA)
    
    # Show results
    cv2.imshow("Original", src)
    cv2.imshow("Binary", bw)
    cv2.imshow("Grid Mask", grid)
    cv2.imshow("Grid Removed", cleaned)
    cv2.imshow("Inpainted Result", inpainted)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
remove_grid_lines()
