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
# from utils.dataset import IGVCDataset
from utils.train_util import train_net
from utils.common_utils import load_model

gpuid = 0
print(torch.cuda.get_device_properties(gpuid))
device = torch.device(f'cuda:{gpuid}' if torch.cuda.is_available() else 'cpu')
print(device)

model = FastSCNN(in_channel=1, num_classes=2).to(device)
load_model(model, './model_gray_2')

cap = cv2.VideoCapture('igvc_vid.mp4')
paused = False;

# Auxiliary function to show connected components
def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0
    
    return labeled_img

while(cap.isOpened()):
    
    ret, frame = cap.read()
    
    # Various tranformations
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame,(9, 9), 4, 4)
    test_transforms = transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], 
                             std=[0.5])
    ])
    
    # Get output from model
    img = Image.fromarray(np.uint8(frame))
    img = test_transforms(img).unsqueeze(0).to(device)
    output = model(img).detach().cpu()
    
    # Select one of the below for probabilistic or deterministic mask respectively.
    mask = np.divide(np.exp(output[:, 1]), np.exp(output[0]) + np.exp(output[:, 1]))
    # mask = torch.argmax(output, 1).detach().cpu().numpy().astype(np.uint8)

    label = np.zeros([480, 640, 1], np.uint8)
    label[:, :, 0] = mask[0]*255.0
    
    # OpenCV erode.
    kernel = np.ones((10, 10),np.uint8)
    label = cv2.erode(label, kernel, iterations = 1)

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
        cv2.imwrite('output.png', imshow_components(labels))
    elif key == ord('p'):
        print(output)

cap.release()
cv2.destroyAllWindows()