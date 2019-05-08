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
from utils.common_utils import load_model

# For profiling performance
import time

gpuid = 0
print('Device name: ', torch.cuda.get_device_properties(gpuid).name)
device = torch.device(f'cuda:{gpuid}' if torch.cuda.is_available() else 'cpu')
print('Device id: ', device)

model = FastSCNN(in_channel=1, num_classes=2).to(device)
load_model(model, './model_gray')

cap = cv2.VideoCapture('igvc_test_vid.mp4')
paused = False;

num_frames = 0;
start = time.time()
while cap.isOpened():
    # Reading 1 frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # BGR->Gray->Resize->Blur->PIL-Image->Tensor
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (640, 480))
    frame = cv2.GaussianBlur(frame, (5, 5), 3, 3)
    img = Image.fromarray(np.uint8(frame))
    img_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    img = img_to_tensor(img).unsqueeze(0).to(device)

    # Getting output from model
    output = model(img).detach().cpu()

    # Select one of the below for probabilistic or deterministic mask respectively.
    mask = np.divide(np.exp(output[:, 1]), np.exp(output[0]) + np.exp(output[:, 1]))
    # mask = torch.argmax(output, 1).detach().cpu().numpy().astype(np.uint8)

    label = np.zeros([480, 640, 1], np.uint8)
    label[:, :, 0] = mask[0]*255.0

    cv2.imshow('Input',  frame)
    cv2.imshow('Output', label)
    
    # Read key presses and perform actions
    key = cv2.waitKey(1)
    if key == ord(' '):
        key = cv2.waitKey(-1)
    elif key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite('input.png', frame)
        cv2.imwrite('output.png', label)
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