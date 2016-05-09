%% Machine Learning Self Exercise - Logistic Regression on Wisconsin Breast Cancer Dataset

%% Initialization
clear ; close all; clc

%% Load Data
%  The first 9 columns contains the extracted features and the 10th column
%  contains the label. Label 1 is malignant and 0 is benign.

%% ======================= Part 1.a: Loading cross validation data ========================

data = load('data_cross_validation.csv');
Xcv = data(:, [1, 2, 3, 4, 5, 6, 7 ,8 ,9]); ycv = data(:, 10);

fprintf('Cross validation set loaded...\n')

%% ========================== Part 1.b: Loading test data =================================

data = load('data_test.csv');
Xtest = data(:, [1, 2, 3, 4, 5, 6, 7 ,8 ,9]); ytest = data(:, 10);

fprintf('Test set loaded...\n');

%fprintf(['Plotting test data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n']); plotData(X, y); hold on; xlabel('feature group 1'); ylabel('feature group 2'); legend('Malign', 'Benign'); title('Test data'); hold off;

%% ============================ Part 1.c: Loading training data ============================

data = load('data_train.csv');
X = data(:, [1, 2, 3, 4, 5, 6, 7 ,8 ,9]); y = data(:, 10);

fprintf('Training data loaded...');

%fprintf(['Plotting train data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n']); plotData(X, y); hold on; xlabel('feature group 1'); ylabel('feature group 2'); legend('Malign', 'Benign'); title('Training data'); hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ===================== Adding cross validation data to training data =====================

X = [X;Xcv];
y = [y;ycv];

%% ======================== Part 1.d: PCA for data visualization ============================

[Xtest, mu, sigma] = featureNormalize(Xtest);

%  Before running PCA, it is important to first normalize X
[X_norm, mu, sigma] = featureNormalize(X);

%  Run PCA
[U, S] = pca(X_norm);

%  Project the data onto K = 2 dimension
K = 2;
Z = projectData(X_norm, U, K);

%  Plot the normalized dataset (returned from pca)

plotData(Z, y);
xlabel('Feature projection - 1');
ylabel('Feature projection - 2');
legend('Malign', 'Benign');
title('Breast cancer - cell malignancy data');
fprintf('\nProgram paused. Press enter to continue.\n');

pause;

%% ========================== Part 2: Compute Cost and Gradient ============================

%  In this part we implement the cost and gradient for logistic regression in costFunction.m

X = X_norm;

%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);

% Add intercept term to X
X = [ones(m, 1) X];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

% Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);

%% For regularization, comment the line above and uncomment the two lines below.
% Be sure to use mapFfeature.m to map features to a polynomial function before regularizing.
%lambda = 1;
%[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

fprintf('\nCost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);
fprintf('\nProgram paused. Press enter to continue.\n');

pause;

%% ========================== Part 3: Optimizing using fminunc  =============================

%  In this part we use a built-in function (fminunc) to find the optimal parameters theta

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta. This function will return theta and the cost
[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

% Print theta to screen
fprintf('\nCost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

% Plot Boundary
plotDecisionBoundary(theta, Z, y);

hold on;
xlabel('Feature projection - 1');
ylabel('Feature projection - 2');
title('Breast cancer - cell malignancy hypothesis');
hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% =========================== Part 4: Predict and Accuracies ================================
%  After learning the parameters, we'll like to use it to predict the outcomes on unseen data. 
%  In this part, we use the logistic regression model to predict the probability that cell in 
%  the breast tissue is benign or malignant based on the extracted features.
%  Furthermore, we will compute the training and test set accuracies of our model. The code is in predict.m

%  Predict probability for a tissue given extracted features 

%prob = sigmoid([1 1.10E+00 -2.07E+00 1.27E+00 9.84E-01 1.57E+00 3.28E+00 2.65E+00 2.53E+00 2.22E+00 2.26E+00 2.49E+00 -5.65E-01 2.83E+00 2.49E+00 -2.14E-01 1.32E+00 7.24E-01 6.61E-01 1.15E+00 9.07E-01 1.89E+00 -1.36E+00 2.30E+00 2.00E+00 1.31E+00 2.62E+00 2.11E+00 2.30E+00 2.75E+00 1.94E+00] * theta);
%This value should be 1

%prob = sigmoid([1 1.06E+00 -1.41E+00 9.32E-01 9.59E-01 -1.28E+00 -7.99E-01 -5.57E-01 -1.84E-01 -2.16E+00 -1.47E+00 2.82E-01 -3.10E-01 1.47E-01 2.34E-01 -8.91E-01 -9.62E-01 -6.75E-01 -7.07E-01 -9.11E-01 -9.40E-01 7.35E-01 -1.18E+00 5.91E-01 5.79E-01 -1.48E+00 -9.83E-01 -8.03E-01 -4.75E-01 -1.81E+00 -1.40E+00] * theta);
%This value should be 0

[p, F1] = predict(theta, X, y);
fprintf('\nTrain Accuracy: %f\nF1 score: %f\n', mean(double(p == y)) * 100, F1);

[mtest, ntest] = size(Xtest);
Xtest = [ones(mtest, 1) Xtest];

% Compute accuracy on our training & test set
[ptest, F1test] = predict(theta, Xtest, ytest);
fprintf('Test Accuracy: %f\nF1 score: %f\n', mean(double(ptest == ytest)) * 100, F1test);

fprintf('\nProgram ended. Cheers!\n');