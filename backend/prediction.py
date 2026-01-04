import torch
import io
from PIL import Image
from torchvision import transforms
from model import CNN_LSTM
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os

labels = [
    'applauding','blowing_bubbles','brushing_teeth','cleaning_the_floor',
    'climbing','cooking','cutting_trees','cutting_vegetables','drinking',
    'feeding_a_horse','fishing','fixing_a_bike','fixing_a_car','gardening',
    'holding_an_umbrella','jumping','looking_through_a_microscope',
    'looking_through_a_telescope','phoning','playing_guitar','playing_violin',
    'pouring_liquid','pushing_a_cart','reading','riding_a_bike',
    'riding_a_horse','rowing_a_boat','running','shooting_an_arrow',
    'smoking','taking_photos','texting_message','throwing_frisby',
    'using_a_computer','walking_the_dog','washing_dishes','watching_TV',
    'waving_hands','writing_on_a_board','writing_on_a_book'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = os.path.join(os.path.dirname(__file__), "best_model.pth")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Correct backbone and number of classes
model = CNN_LSTM(num_classes=len(labels), hidden_size=256, backbone='resnet34')
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

SEQUENCE_LENGTH = 6  # Must match training

def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0)          
    image = image.unsqueeze(1).repeat(1, SEQUENCE_LENGTH, 1, 1, 1)  
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        pred = outputs.argmax(1).item()
    return labels[pred]

app = FastAPI(title="Action Recognition API")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        prediction = predict_image(image_bytes)
        return JSONResponse(content={"prediction": prediction})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
def read_root():
    return {"message": "Send a POST request to /predict with an image file."}
