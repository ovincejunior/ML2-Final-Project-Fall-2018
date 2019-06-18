#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
#from torch.utils.data.sampler import SubsetRandomSampler
#import logger


# In[ ]:


#--------------------      Data preprocessing
is_cuda=False
if torch.cuda.is_available():
    is_cuda = True

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

data_path = 'asl_alpha'

aslDataset = torchvision.datasets.ImageFolder(
    root=data_path,
    transform=transform
)

# To find the classes
aslDataset.classes


batch_size = 4

aslLoader = torch.utils.data.DataLoader(aslDataset, batch_size=batch_size, shuffle=True)

# Get some images from dataset
dataiter = iter(aslLoader)
images, labels = dataiter.next()


# In[ ]:


images[0].shape


# In[ ]:


#if normalize, used the same means and std in the plotting.
grid = torchvision.utils.make_grid(images)

plt.imshow(grid.numpy().transpose((1, 2, 0)))
plt.axis('off')
plt.title(labels.numpy());


# In[ ]:


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

imshow(out, title=[class_names[x] for x in classes])


# In[ ]:


def fit(epoch, model, data_loader, phase='training', volatile=False):
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
        volatile = True
    running_loss = 0.0
    running_correct = 0
    
    #label_accumulator = None
    #output_accumulator =  np.zeros((len(data_loader),4))
    #test_size_counter = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        if is_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile), Variable(target)
        ##print('data :', type(data), '****', 'target :', type(target)) ---- to delete
        if phase == 'training':
            optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        #loss = criterion(output, target)

        running_loss += F.nll_loss(output, target, reduction='sum').data.item()
        #running_loss += criterion(output, target).data[0]
        preds = output.data.max(dim=1, keepdim=True)[1]
        
        #Saving values to compute AUC/ROC curve        
        #label_accumulator.append(target)
        #for j in range(data_loader.batch_size):
        #    for i in range(4):
        #         output_accumulator[test_size_counter][i] = output[j][i].cpu().detach().numpy()
        #test_size_counter += 1
        
            
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        
        if phase == 'training':
            loss.backward()
            optimizer.step()

    loss = running_loss / len(data_loader.dataset)
    accuracy = 100. * running_correct / len(data_loader.dataset)

    print(
        '{} loss is {} and {} accuracy is {}/{}, {}'.format(phase, loss, phase, running_correct, len(data_loader.dataset), accuracy ))
    return loss, accuracy, output, target


# In[ ]:


train_losses, train_accuracy = [], []
val_losses, val_accuracy = [], []
for epoch in range(1, 3):
    epoch_loss, epoch_accuracy, output, labels = fit(epoch, cnn, train_loader, phase='training')
    val_epoch_loss, val_epoch_accuracy, output, labels = fit(epoch, cnn, val_loader, phase='validation')
    train_losses.append(epoch_loss)

    # writer_train.add_scalars("loss training", epoch_loss, epoch)
    train_accuracy.append(epoch_accuracy)

    val_losses.append(val_epoch_loss)
    # writer_train.add_scalars("losses", {'train':epoch_loss, 'val':val_epoch_loss}, epoch) #OK
    # writer_train.add_scalars("accuracies", {'train':epoch_loss, 'val':val_epoch_loss}, epoch)

    val_accuracy.append(val_epoch_accuracy)


# In[ ]:


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle

# Getting the whole examples and their prediction
examples = np.zeros(((val_dataset.data[0].shape[0]),4))
for i in range((val_dataset.data[0].shape[0])):
    for j in range(4):
        examples[i][j] = np.round(output[i][j].cpu().detach().numpy(),2)

label_binarized = labels.cpu().detach().numpy()
label_binarized = label_binarize(label_binarized,classes=[0,1,2,3])


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(4):
    fpr[i], tpr[i], _ = roc_curve(label_binarized[:, i], examples[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(label_binarized.ravel(), examples.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2

plt.subplot(1,2,1)
plt.plot(fpr[3], tpr[3], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[3])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Cat classification performance')
plt.legend(loc="lower right")

plt.subplot(1,2,2)
plt.plot(fpr[0], tpr[0], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Ship classification performance')
plt.legend(loc="lower right")
plt.show()
#plt.savefig('ROC_best_worst.png')
plt.close()


# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(4)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(4):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= 4

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'navy', 'deeppink', 'green', 'yellow', 'black','red','darkblue'])
plt.figure()
for i, color in zip(range(4), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='{0} (area = {1:0.2f})'''.format(classes[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('All classes')
plt.legend(loc="lower right")
plt.show()
#plt.savefig('all_classes.png')


# In[ ]:


rom tensorboardX import SummaryWriter
#SummaryWriter encapsulates everything
writer_train = SummaryWriter('runs6/model1')
#writer_val = SummaryWriter('runs2/model1/val')
'''writer_hist = SummaryWriter('runs1/model1/train')
writer_model = SummaryWriter('runs1/model1/train')'''


# In[ ]:




