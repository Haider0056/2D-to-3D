import os
import argparse
from src.image_processing import FloorPlanProcessor
from src.model_generation import Model3DGenerator
# Save the processed image (corrected)
import cv2

def main():
    """
    Main function to process a floor plan image and generate a 3D model.
    """
    parser = argparse.ArgumentParser(description='Convert a 2D floor plan image to a 3D model.')
    parser.add_argument('--input', '-i', required=True, help='Path to the input floor plan image')
    parser.add_argument('--output', '-o', default='output', help='Directory to save the output files')
    parser.add_argument('--wall-height', type=float, default=2.5, help='Height of the walls in meters')
    parser.add_argument('--visualize', action='store_true', help='Visualize the detected features')
    parser.add_argument('--blender-path', default='blender', help='Path to the Blender executable')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Processing floor plan: {args.input}")
    
    # Step 1: Process the floor plan image
    processor = FloorPlanProcessor(args.input)
    floor_plan_data = processor.process()
    
    
    # Visualize the detected features if requested
    if args.visualize:
        processor.visualize_detection(os.path.join(args.output, 'detected_features.png'))
    
    # Step 2: Generate the 3D model
    print("Generating 3D model...")
    generator = Model3DGenerator(floor_plan_data, args.output, wall_height=args.wall_height)
    result = generator.process()
    cv2.imwrite(os.path.join(args.output, 'processed_image.png'), processor.processed_image)
    print("3D model generation complete.")
    print(f"Basic model saved to: {result['basic_model']}")
    print(f"Processed model saved to: {result['processed_model']}")
    print(f"Metadata saved to: {result['metadata']}")
    
    return result

if __name__ == "__main__":
    main()