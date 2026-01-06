# Action Recognition using CNN + LSTM

This project recognizes human actions in images using a trained **CNN-LSTM** model.  
It includes a **backend** built in Python (FastAPI or Flask) and a **frontend** (HTML) for testing with sample images.

---

## Project Structure
ACTIONRECOGUSINGCNNLSTM/
├─ annotationtestimage/ # Sample/test images for frontend testing
├─ backend/ # Python backend
│ ├─ app.py # Main backend app
│ ├─ model.py # Model definition
│ ├─ prediction.py # Prediction logic
│ └─ requirements.txt # Python dependencies
├─ frontend/ # Frontend HTML
│ └─ index.html
├─ .gitignore # Ignored files (venv, model, cache)
└─ README.md # This file

---

## Setup Instructions

### 1. Clone the Repository:
git clone https://github.com/areeba-khan21/Action-Recognition-using-CNN-and-LSTM-and-Web-Applications.git
cd Action-Recognition-using-CNN-and-LSTM-and-Web-Applications/backend


### 2. Set up Python environment

Create and activate a virtual environment:

python3 -m venv venv
source venv/bin/activate

Install dependencies:
pip install -r requirements.txt

### 3. Download the trained model

The trained model is not included in this repo due to its size (~84MB).
Download it from:
https://drive.google.com/file/d/1smLwQ0GysC5CBeSHJ8atAcLxVslU7G1E/view?usp=sharing

Place the downloaded file in the backend folder:
backend/best_model.pth

### 4. Run the backend
If using FastAPI:
cd backend
uvicorn app:app --reload

### 5. Open the frontend

Open frontend/index.html in a browser.
You can now upload sample images to test the system.

### 6. Test with sample images
Open frontend/index.html in your browser.
Click the Upload Image button and select an image from annotationtestimage/.
The app will send the image to the backend and display the predicted action.
You can test the system using the images provided in annotationtestimage/.

