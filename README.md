# breast-cancer-diagnosis-using-logistic-regression

The purpose of this project is to **implement** a machine learning technique called logistic regression, to identify whether a given breast tissue sample is cancerous or not. 

The dataset used is Breast Cancer Wisconsin (Original) Data Set and is divided into 80% training data and 20% test data. The datset itself provides the following 9 features in a normalized scale of 1 - 10 :
- Clump Thickness
- Uniformity of Cell Size
- Uniformity of Cell Shape
- Marginal Adhesion,
- Single Epithelial Cell Size
- Bare Nuclei
- Bland Chromatin
- Normal Nucleoli
- Mitoses

The last column denotes whether the cell is malignant (1) or benign (0).

Principal Component Analysis has been done to visualize the data, by which we can determine that this is a linear classification problem (data.jpg)

![alt tag](https://raw.githubusercontent.com/niravjain/breast-cancer-diagnosis-using-logistic-regression/master/data.jpg)

Thus, using logistic regression, we get our hypothesis line for future predictions as follows (hypothesis.jpg) 

![alt tag](https://raw.githubusercontent.com/niravjain/breast-cancer-diagnosis-using-logistic-regression/master/hypothesis.jpg)

The accuracies achieved are as follows:

Training accuracy: 96.146789, F1 score: 0.948403

**Testing accuracy: 98.550725, F1 score: 0.972222**

## Installation

The code is written in MTALAB 9.0.0.341360 (R2016a). The program to run is *model.m*, which will run step by step. The console output explains what is done in each step.

## Author
Nirav Jain *niravr7@gmail.com*
