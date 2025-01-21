from ultralytics import YOLO
import torch

def load_model(device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = YOLO("yolov5s.pt")
    model.to(device)
    print(f"Model {device} working on.")
    return model


#test for CUDA & torch
print("PyTorch Version:", torch.__version__)
print("Is CUDA Avaliable?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Device Name:", torch.cuda.get_device_name(0))
