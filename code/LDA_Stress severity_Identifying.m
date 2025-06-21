warning off             
close all               
clear                   
clc              
res = xlsread('Date-Stress severity-Identifying.xlsx', 'None');
X = res(1:end, 1:end-1); 
Y = res(1:end, end);  
data = [X, Y];
M = round(size(data, 1) * 8 / 10); 
N = size(data, 1) - M;            
[XSelected, XRest, vSelectedRowIndex] = ks(data, M);

X_train = XSelected(:, 1:end - 1); 
Y_train = XSelected(:, end);      
X_test = XRest(:, 1:end - 1);     
Y_test = XRest(:, end);           


X_train = rmmissing(X_train);
Y_train = Y_train(1:size(X_train, 1));  

X_test = rmmissing(X_test);
Y_test = Y_test(1:size(X_test, 1));    

if any(isnan(X_train(:))) || any(isnan(Y_train(:)))
    error('NaN ');
end
if any(isnan(X_test(:))) || any(isnan(Y_test(:)))
    error('NaN ');
end

if size(X_train, 1) ~= length(Y_train)
    error('X_train  Y_train ');
end

if size(X_test, 1) ~= length(Y_test)
    error('X_test  Y_test ');
end


mu = mean(X_train, 1);  
sigma = std(X_train, 0, 1);  

if any(sigma == 0)
    warning('DE');
end

X_train = X_train(:, sigma ~= 0);
X_test = X_test(:, sigma ~= 0);

X_train_standard = (X_train - mu) ./ sigma;
X_test_standard = (X_test - mu) ./ sigma;
if size(X_train_standard, 2) ~= size(X_test_standard, 2)
    error('NO');
end

lda = fitcdiscr(X_train_standard, Y_train, 'DiscrimType', 'pseudoLinear');  

Y_train_pred = predict(lda, X_train_standard);
Y_test_pred = predict(lda, X_test_standard);
accuracy_train = sum(Y_train_pred == Y_train) / M * 100;
accuracy_test = sum(Y_test_pred == Y_test) / N * 100;
confMat_train = confusionmat(Y_train, Y_train_pred);
confMat_test = confusionmat(Y_test, Y_test_pred);
precision_train = diag(confMat_train) ./ sum(confMat_train, 2);
recall_train = diag(confMat_train) ./ sum(confMat_train, 1)';
f1_train = 2 * (precision_train .* recall_train) ./ (precision_train + recall_train);

precision_test = diag(confMat_test) ./ sum(confMat_test, 2);
recall_test = diag(confMat_test) ./ sum(confMat_test, 1)';
f1_test = 2 * (precision_test .* recall_test) ./ (precision_test + recall_test);
fprintf('%.2f%%\t%.2f\t%.2f\t%.2f\t%.2f%%\t%.2f\t%.2f\t%.2f\n', ...
    accuracy_train, mean(precision_train)*100, mean(recall_train)*100, mean(f1_train)*100, ...
    accuracy_test, mean(precision_test)*100, mean(recall_test)*100, mean(f1_test)*100);

accuracy_per_class = diag(confMat_test) ./ sum(confMat_test, 2);
overall_accuracy_test = sum(diag(confMat_test)) / sum(confMat_test(:)) * 100;
figure;
confusionchart(confMat_test, 'Title', 'Confusion Matrix for Test Set', ...
               'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
title(sprintf('Confusion Matrix for Test Set (Accuracy: %.2f%%)', overall_accuracy_test));
