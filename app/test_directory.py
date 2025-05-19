import sys
import os

# Print current directory and parent directory
print("Current directory:", os.path.abspath('.'))
print("Parent directory:", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import
try:
    from models.vader_model import VaderModel
    print("Successfully imported VaderModel")
except Exception as e:
    print("Import error:", e)
    
# Check if model files exist
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
print("Looking for models in:", model_path)
print("Files in models directory:", os.listdir(model_path) if os.path.exists(model_path) else "Directory not found")