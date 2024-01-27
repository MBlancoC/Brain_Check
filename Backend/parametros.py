from dotenv import load_dotenv
import os


# PATHS
Path_CV = "Models/OpenCV/mi_modelo_OpenCV.h5"
Path_YL="Models/YOLOv8_project/yolov8_W.pt"
PATH_LOGO = "Frontend/static/LogoBrainCheck.png"

# KEYs
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY")
