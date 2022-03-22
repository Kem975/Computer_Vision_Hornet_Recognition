import torch
from torch.utils.data.dataloader import DataLoader 
import torchvision
import torch.nn as nn
import torch.optim as optim
#from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from customDataset import myDataset
import cv2
import scipy.misc


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PATH = 'state_dict_model_2.pt'
model = torchvision.models.resnet18(pretrained=True)

model.fc = nn.Linear(512, 2)

model.load_state_dict(torch.load(PATH))
#model.to(device)

batch_size = 1
dataset_test = myDataset(csv_file = 'test_hornet.csv', root_dir = '../../data/dataset_test', transform = transforms.ToTensor())
test_loader = DataLoader(dataset = dataset_test, batch_size = batch_size, shuffle=True)

x,y = next(iter(test_loader))

while y!=0:
    x,y = next(iter(test_loader))
    print(y)


def check_accuracy(loader, model):
    model.eval()
    
    scores = model(x)
    _, predictions = scores.max(1)



def activation_hook(inst, inp, out):

    weight_conv = model.layer4[1].bn2.weight.data

    resizer = nn.Upsample(scale_factor=32)
    
    activation_resized = resizer(out)

    filtre = activation_resized[0][0][:][:] 

    for i in range(1,512,1):
        filtre += weight_conv[i] * activation_resized[0][i][:][:] 

    filtre_np = filtre.detach().numpy()

    seuil_min = 0 # On choisit la valeur minimale dans le filtre. Plus cette valeur sera proche de 0, plus la zone de détection sera distincte mais moins on verra l'image de base.

    seuil_calcule = ( seuil_min*( np.max(filtre_np) - np.min(filtre_np) ) ) / ( 1 - seuil_min )

    filtre_np_normalise = ( filtre_np + seuil_calcule - np.min(filtre_np) ) / ( np.max(filtre_np) - np.min(filtre_np) + seuil_calcule )

    filtre_np_normalise_RGB = (filtre_np_normalise*255.999).astype(np.uint8)

    
    cv2.imwrite('../figure/Live_detection/filtre_img.jpg', filtre_np_normalise_RGB)

    A = x.detach().numpy()
    
    A_RGB = (np.dstack(( A[0][0][:][:] , A[0][1][:][:] , A[0][2][:][:] )) * 255.999) .astype(np.uint8)
    cv2.imwrite('../figure/Live_detection/color_img.jpg', A_RGB)

    A_filtre = A*filtre_np_normalise # Batch_size x RGB x L x l

    A_filtre_RGB = (np.dstack(( A_filtre[0,0,:,:] , A_filtre[0,1,:,:] , A_filtre[0,2,:,:] )) * 255.999) .astype(np.uint8)   

    A_filtre_Y = 0.299*A_filtre_RGB[:,:,0] + 0.587*A_filtre_RGB[:,:,1] + 0.114*A_filtre_RGB[:,:,2]

    fig = plt.figure()

    fig.add_subplot(121)
    plt.imshow(A_RGB)
    plt.title("Image de base")
    plt.axis('off')
    fig.add_subplot(122)
    plt.imshow(A_filtre_Y, cmap='jet', vmin=0, vmax=( np.max(A_filtre_Y) ))
    plt.title("Lieu d'intérêt en sortie du réseau de neurones")
    plt.axis('off')
    plt.show()





model.layer4[1].bn2.register_forward_hook(activation_hook)

check_accuracy(test_loader, model)

filtre = cv2.imread('../figure/Live_detection/filtre_img.jpg', 1)
img = cv2.imread('../figure/Live_detection/color_img.jpg', 1)
heatmap_img = cv2.applyColorMap(filtre, cv2.COLORMAP_JET)
fin = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)
cv2.imshow("image", fin)
cv2.imwrite('../figure/Live_detection/detection_superpose.jpg', fin)
cv2.waitKey()







