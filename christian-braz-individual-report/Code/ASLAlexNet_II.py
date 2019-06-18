#----------------    Importing packages

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
from time import clock
from torch.utils.data.sampler import SubsetRandomSampler
import random

#batch_size = 64
epochs = 1

#--------------------      Data preprocessing
is_cuda=False
if torch.cuda.is_available():
    is_cuda = True



#-------------------    show images ????
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
    plt.show()



# ---------------------- recreate the split datasets into pytorch
class aslCustomDataset(Dataset):
    def __init__(self, filepathImages, filepathLabels, transform=None):
        #initialize the path names for images and labels files
        self.imagespath = filepathImages
        self.labelspath = filepathLabels

        #load the data into temporary mode
        self.dataImages = np.load(self.imagespath, mmap_mode='r')
        self.dataLabels = np.load(self.labelspath, mmap_mode='r')

        #any transformation -- here torch tensor is mandatory
        self.transform = transform

    def __len__(self):
        #return 60900
        return len(self.dataLabels)

    def __getitem__(self, index):
        #transpose the data image to be read in tensor torch
        image = self.dataImages[index].transpose()
        label = self.dataLabels[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ---------------------- create the train dataset
# imagetrainpath = 'file_train_Images.npy'
# labeltrainpath = 'file_train_Labels.npy'
#
# train_dataset = aslCustomDataset(filepathImages=imagetrainpath,
#                                  filepathLabels=labeltrainpath,
#                                  transform=transforms.ToTensor())
#
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#
# train_iter = iter(train_loader)
# print(type(train_iter))
#
# images, labels = train_iter.next()
#
# print('images shape on batch size = {}'.format(images.size()))
# print('labels shape on batch size = {}'.format(labels.size()))
#
#
# # Make a grid from batch
# out = torchvision.utils.make_grid(images)
#
# imshow(out, title=[labels])

# ---------------------- create the validation dataset
# imagevalpath = 'file_val_Images.npy'
# labelvalpath = 'file_val_Labels.npy'
#
# val_dataset = aslCustomDataset(filepathImages=imagevalpath,
#                                  filepathLabels=labelvalpath,
#                                  transform=transforms.ToTensor())
#
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
#
# val_iter = iter(val_loader)
# print(type(val_iter))
#
# images, labels = val_iter.next()
#
# print('images shape on batch size = {}'.format(images.size()))
# print('labels shape on batch size = {}'.format(labels.size()))
#
#
# # Make a grid from batch
# out = torchvision.utils.make_grid(images)
#
# imshow(out, title=[labels])





#
#  Split in Train/Test
#  Subsampling: is a randomly selected percentage of the train and test set that will be effectively used
#  Return: DataLoader to iterate in the train and test
#
def getTrainTest(data_path, test_split=.2, train_batch_size = -1, test_batch_size = -1, subsampling = -1):

    transform = transforms.Compose(
        [
            # transforms.ToTensor(),
            # torchvision.transforms.ToPILImage(),
            # torchvision.transforms.Grayscale(),
            # torchvision.transforms.Resize((640, 480)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        ])

    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transform

    )

    # Creating data indices for training and validation splits:
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    #np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    #Subsampling
    if (subsampling != -1):
        number_of_training_examples = int(np.floor(subsampling * len(train_indices)))
        train_indices = random.sample(range(len(train_indices)), number_of_training_examples)
        number_of_testing_examples = int(np.floor(subsampling * len(test_indices)))
        test_indices = random.sample(range(len(test_indices)), number_of_testing_examples)


    # Creating data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    if (train_batch_size == -1):
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_indices), sampler=train_sampler)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, sampler=train_sampler)

    if (test_batch_size == -1):
        test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(test_indices), sampler=test_sampler)
    else:
        test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=test_batch_size, sampler=test_sampler)

    return train_loader, test_loader

class AlexNet(nn.Module):

    def __init__(self, num_classes=29):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=1, padding=0),  #stride=4, padding=2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2, stride=2), #there was no striding
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   #kernel_size=3
            nn.Conv2d(192, 384, kernel_size=3, padding=0), # padding=1 for the rest
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #kernel_size=3
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 9 * 9, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 9 * 9)
        x = self.classifier(x)
        return x

# -----------------------------------------------------------------------------------
alexnet = AlexNet()
if is_cuda:
    alexnet.cuda()
# -----------------------------------------------------------------------------------
# Loss and Optimizer
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
#criterion = F.nn_loss()
optimizer = torch.optim.Adam(alexnet.parameters(), lr=learning_rate)
# -----------------------------------------------------------------------------------

def fit(epoch, model, data_loader, phase='training', volatile=False):
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
        volatile = True
    running_loss = 0.0
    running_correct = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        if is_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile), Variable(target)
        ## print('data :', type(data), '****', 'target :', type(target)) ---- to delete
        if phase == 'training':
            optimizer.zero_grad()
        output = model(data)
        #loss = F.nll_loss(output, target)
        loss = criterion(output, target)

        #running_loss += F.nll_loss(output, target, reduction='sum').data.item()
        running_loss += criterion(output, target).cpu().data.item()
        preds = output.data.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        if phase == 'training':
            loss.backward()
            optimizer.step()

    loss = running_loss / len(data_loader.sampler)
    accuracy = 100. * running_correct / len(data_loader.sampler)

    print(
        '{} loss is {} and {} accuracy is {}/{}, {}%'.format(phase, loss, phase, running_correct, len(data_loader.sampler), accuracy ))
    return loss, accuracy
# ----------------------------------------------------------------------------------

train_losses, train_accuracy = [], []
val_losses, val_accuracy = [], []

start = clock()
import time
begin = time.perf_counter()

train_loader, val_loader = getTrainTest('./asl_alphabet_train',test_split=.3,train_batch_size=64, test_batch_size=64, subsampling=.2)

for epoch in range(0, epochs):
    epoch_loss, epoch_accuracy = fit(epoch, alexnet, train_loader, phase='training')
    val_epoch_loss, val_epoch_accuracy = fit(epoch, alexnet, val_loader, phase='validation')
    train_losses.append(epoch_loss)

    # writer_train.add_scalars("loss training", epoch_loss, epoch)
    train_accuracy.append(epoch_accuracy)

    val_losses.append(val_epoch_loss)
    # writer_train.add_scalars("losses", {'train':epoch_loss, 'val':val_epoch_loss}, epoch) #OK
    # writer_train.add_scalars("accuracies", {'train':epoch_loss, 'val':val_epoch_loss}, epoch)

    val_accuracy.append(val_epoch_accuracy)

elapsed = (clock() - start)/60

print(elapsed)
end = time.perf_counter()
print("\nTime elapsed: %.2f" % ((end - begin)/60) + " min")


