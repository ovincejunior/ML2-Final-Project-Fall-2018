import numpy as np
from sklearn.model_selection import train_test_split

imagesLoaded = np.load("ASLimages.npy")
labelsLoaded = np.load("ASLlabels.npy")

# ----------------- Split the dataset
X_train, X_test, y_train, y_test = train_test_split(imagesLoaded, labelsLoaded, test_size=.3, stratify=labelsLoaded, shuffle=True, random_state=48)

# ----------------- Save the data in a numpy file

np.save('file_val_test_Labels', y_test)
np.save('file_train_Labels', y_train)

np.save('file_val_test_Images', X_test)
np.save('file_train_Images', X_train)

print("love")
print('again')