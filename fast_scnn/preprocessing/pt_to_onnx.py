# Torch and related packages
import torch
import torchvision

# Our model which we train
from models.fast_scnn import FastSCNN
from utils.common_utils import load_params

# Setting and checking device
gpuid=0
device = torch.device('cuda:'+str(gpuid) if torch.cuda.is_available() else 'cpu')
torch.cuda.device(device)
print('Device name:', torch.cuda.get_device_properties(device).name)
print('Device id:  ', device)

# Loading model and pameters
in_channels = 3
spatial_dim = (720, 1280)
width_multiplier = 0.5
classes = ['other', 'grass', 'lane_marker']
model = FastSCNN(in_channels, spatial_dim, width_multiplier, len(classes)).to(device)
load_params(model, './models/fast_scnn_params.pt')

# Generating ONNX model by passing dummy input
dummy_input = torch.FloatTensor(1, in_channels, spatial_dim[0], spatial_dim[1]).to(device)
torch.onnx.export(model, dummy_input, "./models/fast_scnn.onnx", verbose=True)