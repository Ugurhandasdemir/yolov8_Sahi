from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image
import cv2
import os

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path="C:/Users/UGURHANDASDEMIR/PycharmProjects/pythonProject1/best.pt",
    confidence_threshold=0.5,
    device="cpu", # or 'cuda:0'
)

image_dir = "img"
image_paths = [os.path.join(image_dir, image_name) for image_name in os.listdir(image_dir) if image_name.endswith(('.png', '.jpg', '.jpeg'))]

for image_path in image_paths:
    image = read_image(image_path)

    result = get_sliced_prediction(
        image,
        detection_model,
        slice_height = 1080,
        slice_width = 1080,
        overlap_height_ratio = 0.1,
        overlap_width_ratio = 0.1
    )

    base_name=os.path.basename(image_path)
    result.export_visuals(export_dir=f"C:/Users/UGURHANDASDEMIR/PycharmProjects/pythonProject1/result/")
    os.rename("C:/Users/UGURHANDASDEMIR/PycharmProjects/pythonProject1/result/prediction_visual.png", base_name)


