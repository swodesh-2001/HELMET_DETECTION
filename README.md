# HELMET_DETECTION
**TEAM DIJON**

Project Helmet Detection is a project to detect riders on two wheelers without the helmet.
This project is divided into different sections


**OPTIONAL**
Sometimes the code may not run due to the difference in versions so kindly install the version we have given in the requirements.txt

********************************************************************************************************************
**First Section**
The first part is getting the dataset. We team dijon downloaded dataset from kaggle.The dataset has images and its annotations which is in xml file
Our Train_XML file reads xml file and crops the images into two categories with helmet and non helmet
Then it is given as input to mobileNet model and training is done

**NOTE:**
TO RUN THE CODE. DOWNLOAD THE DATASET FROM HERE 
https://www.kaggle.com/andrewmvd/helmet-detection
AND PUT 'images' folder and 'annotations' folder in same directory where Train_XMLfile.py is kept.

*********************************************************************************************************************
