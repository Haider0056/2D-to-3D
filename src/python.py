import os
import json
import numpy as np
from model_generation import Model3DGenerator  # Ensure this module exists

# LLM response - this would typically come from your LLM API call
llm_response = '''
{
  "walls": [
    [0, 0, 700, 0],
    [700, 0, 700, 500],
    [700, 500, 0, 500],
    [0, 500, 0, 0]
  ],
  "doors": [
    [310, 500, 80, 200, 0]
  ],
  "windows": [
    [700, 250, 150, 100, 90],
    [350, 0, 100, 100, 0]
  ],
  "rooms": [
    [[0, 0], [700, 0], [700, 500], [0, 500]]
  ],
  "dimensions": {
    "min_x": 0,
    "min_y": 0,
    "max_x": 700,
    "max_y": 500,
    "scale_factor": 0.01,
    "overall_width_meters": 7.0,
    "overall_height_meters": 5.0,
    "rooms": [
      {
        "area_meters": 35.0,
        "perimeter_meters": 24.0,
        "width_meters": 7.0,
        "height_meters": 5.0
      }
    ]
  }
}
'''

# Parse the JSON response from the LLM
floor_plan_data = json.loads(llm_response)

# Convert rooms format to numpy arrays as expected by the Model3DGenerator
floor_plan_data['rooms'] = [np.array(room) for room in floor_plan_data['rooms']]

# Define output directory
output_dir = "output_models"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Create 3D model generator instance
model_generator = Model3DGenerator(floor_plan_data, output_dir)

# Generate the 3D model and metadata
result = model_generator.process()

print(f"Basic 3D model saved at: {result['basic_model']}")
if result['processed_model']:
    print(f"Detailed 3D model saved at: {result['processed_model']}")
print(f"Metadata saved at: {result['metadata']}")