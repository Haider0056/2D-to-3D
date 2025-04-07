python -m http.server 8000
http://localhost:8000/static/viewer.html?model=../output_folder/basic_model.obj
python -m src.main --input "binary_output.jpg" --output "output_folder" --visualize
blender output_folder/detailed_model.blend