# breast-cancer-diagnosis

The purpose of this project is to natively implement a machine learning technique called logistic regression, to identify whether a given breast tissue sample is cancerous or not. 

The dataset used is Breast Cancer Wisconsin (Original) Data Set and is divided into 80% training data and 20% test data. The datset itself provides the following 9 features in a normalized scale of 1 - 10 - Clump Thickness, Uniformity of Cell Size, Uniformity of Cell Shape, Marginal Adhesion, Single Epithelial Cell Size, Bare Nuclei, Bland Chromatin, Normal Nucleoli, Mitoses. The last column denotes whether the cell is malignant (1) or benign (0).

Principal Component Analysis has been done to visualize the data, by which we can clearly determine that this is a linear classification problem (Check data.jpg)

Thus, using logistic regression, we get our hypothesis line for future predictions (Check hypothesis.jpg) 

The accuracies achieved are as follows:
Training accuracy: 96.146789, F1 score: 0.948403
Testing accuracy: 98.550725, F1 score: 0.972222
