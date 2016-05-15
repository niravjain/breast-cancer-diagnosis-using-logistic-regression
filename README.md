# breast-cancer-diagnosis

The purpose of this project is to natively implement a machine learning technique called logistic regression, to identify whether a given breast tissue sample is cancerous or not. 

The dataset used is Breast Cancer Wisconsin (Original) Data Set and is divided into 80% training data and 20% test data. The datset itself provides the following 9 features in a normalized scale of 1 - 10:
1. Clump Thickness 
2. Uniformity of Cell Size
3. Uniformity of Cell Shape
4. Marginal Adhesion
5. Single Epithelial Cell Size
6. Bare Nuclei
7. Bland Chromatin
8. Normal Nucleoli
9. Mitoses

The last column denotes whether the cell is malignant (1) or benign (0).

Principal Component Analysis has been done to visualize the data, by which we can clearly determine that this is a linear classification problem (Check data.jpg)

Thus, using logistic regression, we get our hypothesis line for future predictions (Check hypothesis.jpg) 

The accuracies achieved are as follows:

Training accuracy - 96.146789
Training F1 score - 0.948403

Testing accuracy - 98.550725
Testing F1 score - 0.972222
