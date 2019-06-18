import numpy as np
from sklearn.model_selection import train_test_split

imagesLoaded = np.load("file_val_test_Images.npy")
labelsLoaded = np.load("file_val_test_Labels.npy")

# ----------------- Split the dataset
X_val, X_test, y_val, y_test = train_test_split(imagesLoaded, labelsLoaded, test_size=.5, stratify=labelsLoaded, shuffle=True, random_state=68)

# ----------------- Save the data in a numpy file

np.save('file_test_Labels', y_test)
np.save('file_val_Labels', y_val)

np.save('file_test_Images', X_test)
np.save('file_val_Images', X_val)

print("love")
print('again')