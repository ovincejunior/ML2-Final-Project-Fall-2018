# final-project-group9
Deliverables for the Machine Learning 2 final project at GWU 

This project is the final project of the Machine Learning II course at GWU/Fall 2018. 

This file present the order each file and commands needs to be run. 
Please follow them carefully. 

1. Download the dataset in the same directory with the rest of the files, using this command below
wget https://s3.amazonaws.com/datasourceml2/asl_alphabet.zip

2. Unzip the asl alphabet zip file, using this command below  
unzip asl_alphabet.zip

3. Run the file loadData1asl.py on pycharm to save the data into a numpy file
you should have 2 numpy files in your directory (ASLimages.npy, ASLlabels.npy) 

4. Run the file loadData2asl.py on pycharm to save the train set and the rest into a numpy file 
you should have 4 more numpy files in your directory (file_train_Labels.npy, file_train_Images.npy, 
file_val_test_Labels.npy, file_val_test_Images.npy)

5. Run the file loadData3asl.py on pycharm to save the test and validation sets into a numpy file 
you should have 4 more numpy files in your directory (file_test_Labels.npy, file_test_Images.npy, 
file_val_Labels.npy, file_val_Images.npy)

6. For memory management, remove the ASLImages.npy and ASLlabels.npy file after executing number 4. 

7. For memory management, remove the file_val_test_Labels.npy and file_val_test_Images.npy file after executing number 5.

8.A At this point if you should have these 6 files into your directory before starting any training on the dataset:
file_train_Labels.npy, file_train_Images.npy, file_test_Labels.npy, file_test_Images.npy, 
file_val_Labels.npy, file_val_Images.npy in the same directory with all the python files.

8.B install tensorboardX and tensorboard in pycharm.

9. Once 8A and 8B confirmed, start the training with the baseline model. So, run these files in pycharm:
- aslBaselineModelML2.py
- aslBaselineModelML2final.py
- aslBaselineModelML2testing.py

10. Start the training with the LeNet model. So, run these files in pycharm:
- aslLeNetML2.py
- aslLeNetML2final.py 
- aslLeNetML2testing.py 

11. Start the training with the AlexNet model. So, run these files in pycharm:
- aslAlexNetML2.py 

12. (optional) Run tensorboard to see the loss and accuracy trend with this command:
tensorboard --logdir=(any file with the word runs)

Tensorboard will provide a link 
Open another page of your instance and run this command to open a browser: chromium-browser 

Copy/Paste the link provided by tensorboard and you will be able to see the different graphs. 
 
