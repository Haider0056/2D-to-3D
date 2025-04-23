import os
import cv2
import numpy as np
import requests
import json
from LLM_Model import Model3DGenerator

def generate_completion(user_prompt):
    # """
    # Send user prompt to Langbase API to generate floor plan
    # """
    # url = "https://api.langbase.com/v1/pipes/run"
    # api_key = "pipe_aUzf1Bv4m15hmq9qjFxgjrNUrRdFopxqjSNsYbdCpJDPVP1MKqiB5aNAescF7MtJVAAWeoUSW2bnVXHQkMBKnAb"
    
    # body_data = {
    #     "messages": [
    #         {"role": "user", "content": f"Generate a detailed floor plan with these specifications: {user_prompt}. Provide the response as a valid Python dictionary with 'walls', 'doors', 'windows', 'rooms', and 'dimensions' keys. Use numpy array for rooms. Example format: {{'walls': [(x1,y1,x2,y2), ...], 'doors': [(x,y,width,height,angle), ...], 'windows': [(x,y,width,height,angle), ...], 'rooms': [np.array(...), ...], 'dimensions': {{'min_x': 100, 'max_x': 500, ...}}"}
    #     ],
    #     "stream": False
    # }
    
    # headers = {
    #     "Content-Type": "application/json", 
    #     "Authorization": f"Bearer {api_key}"
    # }
    
    # try:
    #     response = requests.post(url, headers=headers, json=body_data)
        
    #     if response.status_code == 200:
    #         res = response.json()
    #         completion = res.get('completion', 'No completion found')
            
    #         # Print the raw completion for debugging
    #         print("Raw Completion:", completion)
            
    #         # Try parsing the response
    #         try:
    #             # First, try to parse as a dictionary
    #             if isinstance(completion, dict):
    #                 return completion
                
    #             # If it's a string, try to evaluate it
    #             floor_plan_data = eval(completion)
    #             return floor_plan_data
            
    #         except (SyntaxError, ValueError, NameError) as e:
    #             print(f"Error parsing floor plan data: {e}")
    #             print("Completion received:", completion)
    #             return None
    #     else:
    #         print(f"Error: {response.status_code} - {response.text}")
    #         return None
    
    # except requests.RequestException as e:
    #     print(f"Request failed: {e}")
    #     return None
  return {
        'walls': [
            (100, 100, 900, 100),  # Top wall
            (100, 700, 900, 700),  # Bottom wall
            (100, 100, 100, 700),  # Left wall
            (900, 100, 900, 700),  # Right wall
            (400, 100, 400, 400),  # Living room to Kitchen
            (400, 400, 700, 400),  # Kitchen to Bedrooms
            (700, 100, 700, 300),  # Bedroom 1 to Garage
            (700, 300, 900, 300),  # Bedroom 2 to Garage
            (700, 300, 700, 400),  # Between Bedrooms
            (700, 400, 900, 400),  # Bedroom 2 to Garage
            (400, 400, 400, 700),  # Kitchen to Bedroom 1
            (700, 400, 700, 700)   # Bedroom 1 to Bedroom 2
        ],
        'doors': [
            (400, 250, 3, 30, 90),
            (550, 400, 30, 3, 0),
            (700, 200, 3, 30, 90),
            (700, 350, 30, 3, 0),
            (100, 350, 3, 30, 90)
        ],
        'windows': [
            (250, 100, 40, 10, 0),
            (550, 100, 40, 10, 0),
            (800, 100, 40, 10, 0),
            (800, 400, 40, 10, 0),
            (900, 200, 40, 10, 90)
        ],
        'rooms': [
            np.array([[100, 100], [400, 100], [400, 400], [100, 400], [100, 100]], dtype=np.int32),
            np.array([[400, 100], [700, 100], [700, 400], [400, 400], [400, 100]], dtype=np.int32),
            np.array([[700, 100], [900, 100], [900, 300], [700, 300], [700, 100]], dtype=np.int32),
            np.array([[700, 300], [900, 300], [900, 400], [700, 400], [700, 300]], dtype=np.int32),
            np.array([[700, 400], [900, 400], [900, 700], [700, 700], [700, 400]], dtype=np.int32)
        ],
        'dimensions': {
            'min_x': 100,
            'min_y': 100,
            'max_x': 900,
            'max_y': 700,
            'scale_factor': 0.09,
            'overall_width_meters': 72.0,
            'overall_height_meters': 54.0,
            'rooms': [
                {'area_meters': 27.0, 'perimeter_meters': 60.0, 'width_meters': 27.0, 'height_meters': 27.0, 'centroid': (250, 250)},
                {'area_meters': 27.0, 'perimeter_meters': 60.0, 'width_meters': 27.0, 'height_meters': 27.0, 'centroid': (550, 250)},
                {'area_meters': 18.0, 'perimeter_meters': 54.0, 'width_meters': 18.0, 'height_meters': 18.0, 'centroid': (800, 200)},
                {'area_meters': 9.0, 'perimeter_meters': 36.0, 'width_meters': 18.0, 'height_meters': 9.0, 'centroid': (800, 350)},
                {'area_meters': 27.0, 'perimeter_meters': 60.0, 'width_meters': 18.0, 'height_meters': 27.0, 'centroid': (800, 550)}
            ]
        },
        'processed_image': np.ones((700, 1000, 3), dtype=np.uint8) * 255
    }
def generate_house_floor_plan(floor_plan_dict):
    """
    Convert Langbase generated floor plan to the required format
    
    Args:
        floor_plan_dict (dict): Floor plan data from Langbase
    
    Returns:
        dict: Formatted floor plan data
    """
    # Extract data from the Langbase response
    walls = floor_plan_dict.get('walls', [])
    doors = floor_plan_dict.get('doors', [])
    windows = floor_plan_dict.get('windows', [])
    rooms_raw = floor_plan_dict.get('rooms', [])
    dimensions = floor_plan_dict.get('dimensions', {})
    
    # Convert rooms to numpy arrays if they aren't already
    rooms = []
    for room in rooms_raw:
        if isinstance(room, list):
            room = np.array(room, dtype=np.int32)
        rooms.append(room)
    
    # Create a blank white image
    width = dimensions.get('max_x', 600)
    height = dimensions.get('max_y', 400)
    floor_plan = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    return {
        'walls': walls,
        'doors': doors,
        'windows': windows,
        'rooms': rooms,
        'dimensions': dimensions,
        'processed_image': floor_plan
    }

def main():
    # Prompt user for floor plan description
    print("Welcome to the Floor Plan Generator!")
    user_prompt = input("Please describe the house floor plan you want to generate:\n> ")
    
    # Create output directory
    output_dir = 'house_3d_model'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate floor plan from Langbase
    floor_plan_from_langbase = generate_completion(user_prompt)
    
    if floor_plan_from_langbase is None:
        print("Failed to generate floor plan. Exiting.")
        return
    
    # Convert Langbase floor plan to required format
    floor_plan_data = generate_house_floor_plan(floor_plan_from_langbase)
    
    # Save floor plan visualization
    cv2.imwrite(os.path.join(output_dir, 'floor_plan.png'), floor_plan_data['processed_image'])
    
    # Create 3D model generator
    generator = Model3DGenerator(
        floor_plan_data, 
        output_dir, 
        wall_height=2.7,  # Slightly higher ceiling
        blender_path='blender'  # Adjust if Blender is in a different path
    )
    
    # Process and generate 3D model
    result = generator.generate_model()
    
    print("3D Model Generation Complete:")
    print(f"Basic Model: {result['basic_model']}")
    print(f"Processed Model: {result['detailed_model']}")
    
    return result

if __name__ == "__main__":
    main()