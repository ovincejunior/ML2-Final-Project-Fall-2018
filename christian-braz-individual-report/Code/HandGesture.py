from sklearn.datasets import load_files
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image # For handling the images
import io
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import random

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# data = load_files("./dataset")
# aux = []
# trans1 = transforms.ToTensor()
# trans2 = transforms.ToPILImage()
# for i,l in zip(data.data,data.target):
#     aux.append( (trans1(Image.open(io.BytesIO(i)).resize((416, 275)) ),l) )
#
#
# loader = torch.utils.data.DataLoader(aux, batch_size=5, shuffle=False)
# dataiter = iter(loader)
# images, labels = dataiter.next()
#imshow(torchvision.utils.make_grid([trans2(v) for v in images]))

transform = transforms.Compose(
[
 #transforms.ToTensor(),
 #torchvision.transforms.ToPILImage(),
 #torchvision.transforms.Grayscale(),
 #torchvision.transforms.Resize((640, 480)),
 transforms.ToTensor(),
 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

 ])


data_path = './asl_alphabet_train'
train_dataset = torchvision.datasets.ImageFolder(
    root=data_path,
    transform=transform

)

'''#To create a train/test split while still being able to use the convenient dataset/dataloader scaffolding that PyTorch provides:
# Split dataset into training set and test set
# Use the indices of the split to create PyTorch Samplers
# Feed that sampler into our DataLoaders to tell them which dataset elements to include in their iteration
# Iterate over the appropriate DataLoaders in the training/testing loops
'''

test_split = .2
random_seed= 42
# Creating data indices for training and validation splits:
dataset_size = len(train_dataset)
indices = list(range(dataset_size))
split = int(np.floor(test_split * dataset_size))
np.random.seed(random_seed)
np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]

subsampling = .2
number_of_training_examples = int(np.floor(subsampling*len(train_indices)))
aux = random.sample(range(len(train_indices)), number_of_training_examples)
train_indices = aux

# Creating data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

# train_loader = torch.utils.data.DataLoader(
#     train_dataset,
#     batch_size=20,
#     num_workers=1,
#     shuffle=True
# )
batch_size = 5

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=test_sampler)

print("Train size: %i" % len(train_loader.sampler))
print("Test size: %i" % len(test_loader.sampler))

#classes = ['A','B','C','D','E','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

# Get some images from train
dataiter = iter(train_loader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print(labels)
print(images.shape)
print(images[0].shape)
print(' '.join('%5s' % train_dataset.classes[labels[j]] for j in range(5)) )

#dataiter = iter(train_loader)
#for images, labels in train_loader:
#    images, labels = dataiter.next()
#    #imshow(torchvision.utils.make_grid(images))
#    print(labels)


# Get some images from test
#dataiter = iter(test_loader)
#for images, labels in test_loader:
#    images, labels = dataiter.next()
#    imshow(torchvision.utils.make_grid(images))

#    print(labels)
#    print(images.shape)
#    print(images[0].shape)
