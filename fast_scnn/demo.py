# Torch and related packages
import torch
from torchvision import transforms, datasets, utils
import torch.nn.functional as F
from PIL import Image

# OpenCV
import cv2

# Math and plot utils
import numpy as np

# Our model which we train
from models.fast_scnn import FastSCNN

# Custom Utilities
from utils.common_utils import load_params

# For profiling performance
import time

# Setting and checking device
gpuid=0
device = torch.device('cuda:'+str(gpuid) if torch.cuda.is_available() else 'cpu')
torch.cuda.device(device)
print('Device name:', torch.cuda.get_device_properties(device).name)
print('Device id:  ', device)

# Loading model and parameters
model = FastSCNN(in_channels=3, width_multiplier=0.5, num_classes=3).to(device)
load_params(model, './models/fast_scnn_params.pt')
torch.no_grad()

# Opening test video
cap = cv2.VideoCapture('igvc_test_vid.mp4')
paused = False;

# Inference for each frame
num_frames = 0;
start = time.time()
while cap.isOpened():
    # Reading 1 frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize->BGR-OpenCV-Image->RGB-PIL-Image->Tensor
    frame = cv2.resize(frame, (1280, 720))
    img = Image.fromarray(np.uint8(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
    img_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    img = img_to_tensor(img).unsqueeze(0).to(device)

    # Getting output from model
    output = model(img)

    # Select one of the below for probabilistic or deterministic mask respectively.
    # predictions = (torch.nn.Softmax2d()(output)[0, 2, :, :].detach().cpu().numpy()*255.0).astype(np.uint8)
    predictions = torch.argmax(output, 1).detach().cpu().numpy().astype(np.uint8)[0]*127

    
    # Displaying output
    cv2.imshow('Input', frame)
    cv2.imshow('Output', predictions)

    # Read key presses and perform actions
    key = cv2.waitKey(1)
    if key == ord(' '):
        key = cv2.waitKey(-1)
    elif key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite('input.png', frame)
        cv2.imwrite('output.png', predictions)
    elif key == ord('p'):
        print(output)
    num_frames = num_frames + 1
    finish = time.time()
    if finish > start + 1:
        print("Average FPS:", num_frames/(finish - start))
        num_frames = 0
        start = time.time()

cap.release()
cv2.destroyAllWindows()