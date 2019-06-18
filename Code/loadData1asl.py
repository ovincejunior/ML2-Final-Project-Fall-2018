
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

#--------------------      Data preprocessing
is_cuda=False
if torch.cuda.is_available():
    is_cuda = True

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

data_path = 'asl_alphabet_train'

aslDataset = torchvision.datasets.ImageFolder(
    root=data_path,
    transform=transform
)

'''
# To find the classes
aslDataset.classes


batch_size = 4

aslLoader = torch.utils.data.DataLoader(aslDataset, batch_size=batch_size, shuffle=True)

# Get some images from dataset
dataiter = iter(aslLoader)
images, labels = dataiter.next()

# ----------------- display images

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(aslLoader))
class_names = aslDataset.classes

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])'''




# Get all the data in a batch
batch_size = 87000

aslLoader = torch.utils.data.DataLoader(aslDataset, batch_size=batch_size, shuffle=True)
inputs, classes = next(iter(aslLoader))

class_labels = aslLoader.dataset.classes

#save images and labels into a numpy file
np.save('ASLimages', inputs)
np.save('ASLlabels', classes)

print("love")
print('again')