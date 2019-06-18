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

batch_size = 64
epochs = 5

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
imagetrainpath = 'file_train_Images.npy'
labeltrainpath = 'file_train_Labels.npy'

train_dataset = aslCustomDataset(filepathImages=imagetrainpath,
                                 filepathLabels=labeltrainpath,
                                 transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

train_iter = iter(train_loader)
print(type(train_iter))

images, labels = train_iter.next()

print('images shape on batch size = {}'.format(images.size()))
print('labels shape on batch size = {}'.format(labels.size()))


# Make a grid from batch
out = torchvision.utils.make_grid(images)

imshow(out, title=[labels])

# ---------------------- create the validation dataset
imagevalpath = 'file_val_Images.npy'
labelvalpath = 'file_val_Labels.npy'

val_dataset = aslCustomDataset(filepathImages=imagevalpath,
                                 filepathLabels=labelvalpath,
                                 transform=transforms.ToTensor())

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

val_iter = iter(val_loader)
print(type(val_iter))

images, labels = val_iter.next()

print('images shape on batch size = {}'.format(images.size()))
print('labels shape on batch size = {}'.format(labels.size()))


# Make a grid from batch
out = torchvision.utils.make_grid(images)

imshow(out, title=[labels])

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1   = nn.Linear(16*47*47, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 29)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

# -----------------------------------------------------------------------------------
lenet = LeNet()
if is_cuda:
    lenet.cuda()
# -----------------------------------------------------------------------------------
# Loss and Optimizer
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
#criterion = F.nn_loss()
optimizer = torch.optim.Adam(lenet.parameters(), lr=learning_rate)
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

    loss = running_loss / len(data_loader.dataset)
    accuracy = 100. * running_correct / len(data_loader.dataset)

    print(
        '{} loss is {} and {} accuracy is {}/{}, {}'.format(phase, loss, phase, running_correct, len(data_loader.dataset), accuracy ))
    return loss, accuracy
# -----------------------------------------------------------------------------------

train_losses, train_accuracy = [], []
val_losses, val_accuracy = [], []

start = clock()

for epoch in range(0, epochs):
    epoch_loss, epoch_accuracy = fit(epoch, lenet, train_loader, phase='training')
    val_epoch_loss, val_epoch_accuracy = fit(epoch, lenet, val_loader, phase='validation')
    train_losses.append(epoch_loss)

    # writer_train.add_scalars("loss training", epoch_loss, epoch)
    train_accuracy.append(epoch_accuracy)

    val_losses.append(val_epoch_loss)
    # writer_train.add_scalars("losses", {'train':epoch_loss, 'val':val_epoch_loss}, epoch) #OK
    # writer_train.add_scalars("accuracies", {'train':epoch_loss, 'val':val_epoch_loss}, epoch)

    val_accuracy.append(val_epoch_accuracy)

elapsed = clock() - start

print(elapsed)

print('love')
print('again')