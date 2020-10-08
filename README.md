# Image-Classification-with-CNN-Deep-Learning

CNN IMAGE CLASSIFICATION

Author: acelik8

Dataset link: https://www.kaggle.com/c/dogs-vs-cats

Input: ImagePreprocessing.py is for creating different datasets in
       various sizes and image resolutions as well as removing the 
       corrupted images from the dataset. An example of dataset input
       including both the 64x64 resolution 1000 image dataset and their
       corresponding labels are provided in "datasets" folder.The 
       different datasets created via the ImagePreprocessing.py program
       are the input for the 3CNN.py program. Images in 
       "outlier" file are used for Outlier Test via outlier.py .

Output: 4 different models trained on 4 different input datasets can 
	be found in "models" folder.
        Model Loss and Model Accuracy Plots
	
execution: python3 ImagePreprocessing.py
           python3 3CNN.py
	   python3 outlier.py
