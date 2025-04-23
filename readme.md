python -m http.server 8000
http://localhost:8000/static/viewer.html?model=../output_folder/basic_model.obj
python -m src.main --input "input_image" --output "output_folder" --visualize
blender output_folder/detailed_model.blend
Backup_API_KEY=