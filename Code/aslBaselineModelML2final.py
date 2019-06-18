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
from tensorboardX import SummaryWriter
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import pandas as pd


torch.backends.cudnn.deterministic = True
torch.manual_seed(999)

# ------------ load the model if necessary
#cnn = torch.load('baselineModelfinal.pkl')
#cnn.eval()


# ----------- Initialization of some key variables
#  Write models outputs in tensorboard files
writer_train = SummaryWriter('finalrunsBM/models')

batch_size = 64
epochs = 10

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

# -------------------- Model (Baseline model - a Lenet variant with 2 CNN layers (5X5) and one FC layer)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(20*47*47, 250)
        self.fc2 = nn.Linear(250, 29)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 44180)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x,p=0.1, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)

# -----------------------------------------------------------------------------------
cnn = CNN()

# ---- write the model on tensorboard
writer_train.add_graph(cnn, Variable((torch.Tensor(train_loader.dataset.dataImages[0:1])).cpu(), ))

if is_cuda:
    cnn.cuda()
# -----------------------------------------------------------------------------------
# Loss and Optimizer

learning_rate = 0.001
criterion = nn.CrossEntropyLoss()

#criterion = F.cross_entropy()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
# -----------------------------------------------------------------------------------
#   Training of the model
#   A. Design the setup
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


# ---- write the model
#writer_train.add_graph(cnn, Variable((torch.Tensor(train_loader.dataset.dataImages)).cpu(), ))

#  C. Training final preparation

# ----- Weight initialization
def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight)

cnn.apply(init_weights)

# ----- Learning rate regularization
#####  NEED TO CHANGE THE patience
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=True)

# ----- Full training
train_losses, train_accuracy = [], []
val_losses, val_accuracy = [], []

start = clock()
for epoch in range(0, epochs):
    epoch_loss, epoch_accuracy = fit(epoch, cnn, train_loader, phase='training')
    val_epoch_loss, val_epoch_accuracy = fit(epoch, cnn, val_loader, phase='validation')

    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)

    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)

    writer_train.add_scalars("losses", {'train_bm':epoch_loss, 'val_bm':val_epoch_loss}, int(epoch))
    writer_train.add_scalars("accuracies", {'train_bm':epoch_accuracy, 'val_bm':val_epoch_accuracy}, int(epoch))

    # Learning rate scheduler update
    scheduler.step(val_epoch_loss)

writer_train.add_histogram("error_bm", np.array(train_losses))

elapsed = clock() - start

print(elapsed)


# -----------------------------------------------------------------------------------
# Model classification metrics

classes = ['A',  'B',  'C',  'D',  'del',  'E',  'F',
           'G',  'H',  'I',  'J',  'K',  'L',  'M',  'N',
           'nothing',  'O',  'P',  'Q',  'R',  'S',  'space',
           'T',  'U',  'V',  'W',  'X',  'Y',  'Z']

correct = 0
total = 0

y_pred = np.empty([0])
y_true = np.empty([0])
scores = np.empty([0])
y_score = np.empty([0])

cnn.eval()
for images, labels in val_loader:
    if is_cuda:
        images = images.cuda()
    images = Variable(images)
    outputs = cnn(images)
    score, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

    y_pred = np.concatenate((y_pred, predicted.data.cpu().numpy()), axis=None)
    y_true = np.concatenate((y_true, labels.data.cpu().numpy()), axis=None)
    scores = np.concatenate((scores, score.data.cpu().numpy()), axis=None)
    y_score = np.concatenate((y_score, outputs.detach().cpu().numpy()), axis=None)


conf_matrix = metrics.confusion_matrix(y_true, y_pred)

np.trace(conf_matrix)

print("classification rate :", np.trace(conf_matrix)/13050)

print('Accuracy of the network on the validation set : %d %%' % (100 * correct / total))

df_conf_matrix = pd.DataFrame(data=conf_matrix, index=classes, columns=classes)

print("Confusion Matrix for the 29 classes:", df_conf_matrix)

classes_accuracy = []
for i in range(len(classes)):
    acc = conf_matrix[i, i]/conf_matrix[i, :].sum()
    classes_accuracy.append(acc)

df_classes = pd.DataFrame(data=classes_accuracy, index=classes, columns=['Accuracy'])

# ------- FOR THE OVERALL MODEL
pr, rc, f1score, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average='macro')

## precision, recall, f-score for a class
#metrics.precision_recall_fscore_support(y_true, y_pred, labels=[3])

####  fpr, tpr, tresh = metrics.roc_curve(y_true, scores, pos_label=0)

#  ------ AUC AND ROC

y_true_binarized = label_binarize(y_true, classes=[i for i in range(29)])
classes_auc = []

for i in range(29):
    auc_c = metrics.roc_auc_score(y_true_binarized[:,i], y_score.reshape(13050, 29)[:,i])
    classes_auc.append(auc_c)


df_classes['AUC']=classes_auc

# Sort the dataframe values
df_classes.sort_values(by=['Accuracy'])
#df_classes.sort_values(by=['AUC'])


# Plot the ROC
def plot_ROC(idx_class):
    plt.figure()
    lw = 2
    fpr, tpr, _ = metrics.roc_curve(y_true_binarized[:,idx_class], y_score.reshape(13050, 29)[:,idx_class])
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % metrics.auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for class '+ str(idx_class))
    plt.legend(loc="lower right")
    plt.show()

plot_ROC(2)

# ------ Calculate the error parameters
error_std = np.array(val_losses).std()
error_mean = np.array(val_losses).mean()

# Save and load only the model parameters as recommended.
torch.save(cnn.state_dict(), 'baselineModelfinal.pkl')

print('love')
print('again')