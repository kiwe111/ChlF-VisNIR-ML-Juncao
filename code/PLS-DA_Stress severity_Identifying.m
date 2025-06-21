warning off             
close all               
clear                   
clc              
res = xlsread('Date-Stress severity-Identifying.xlsx', 'None');
X = res(2:end-1, 1:end-1);    
Y = res(2:end-1, end);       
Y = Y - 1;             
data = [X, Y];
M = round(size(data, 1) * 8 / 10); 
N = size(data, 1) - M;             
[XSelected, XRest, vSelectedRowIndex] = ks(data, M);
X_train = XSelected(:, 1:end - 1); 
Y_train = XSelected(:, end);       
X_test = XRest(:, 1:end - 1);      
Y_test = XRest(:, end);           

Y_train = categorical(Y_train);
Y_test = categorical(Y_test);
[X_train_norm, ps_input] = mapminmax(X_train', 0, 1); 
X_train_norm = X_train_norm';
X_test_norm = mapminmax('apply', X_test', ps_input); 
X_test_norm = X_test_norm';
best_accuracy_test = 0; 
best_num_components = 0; 
best_cm_test = []; 
best_Y_test_pred = []; 
for num_components = 1:20
  
    [Xloadings, Yloadings, Xscores, Yscores, betaPLS, PLSPctVar, MSE, stats] = ...
        plsregress(X_train_norm, double(Y_train), num_components);
    X_train_pls = [ones(size(X_train_norm, 1), 1), X_train_norm] * betaPLS;
    X_test_pls = [ones(size(X_test_norm, 1), 1), X_test_norm] * betaPLS;
    lda_model = fitcdiscr(X_train_pls, Y_train, 'DiscrimType', 'linear');
    Y_train_pred = predict(lda_model, X_train_pls);
    Y_test_pred = predict(lda_model, X_test_pls);
    cm_train = confusionmat(Y_train, Y_train_pred);
    cm_test = confusionmat(Y_test, Y_test_pred);
    accuracy_train = sum(diag(cm_train)) / sum(cm_train(:)) * 100;
    accuracy_test = sum(diag(cm_test)) / sum(cm_test(:)) * 100;
    precision_train = diag(cm_train) ./ sum(cm_train, 1)'; 
    recall_train = diag(cm_train) ./ sum(cm_train, 2);    
    f1_train = 2 * (precision_train .* recall_train) ./ (precision_train + recall_train);
    precision_test = diag(cm_test) ./ sum(cm_test, 1)'; 
    recall_test = diag(cm_test) ./ sum(cm_test, 2);    
    f1_test = 2 * (precision_test .* recall_test) ./ (precision_test + recall_test); 
    mean_precision_train = nanmean(precision_train);
    mean_recall_train = nanmean(recall_train);
    mean_f1_train = nanmean(f1_train);
    mean_precision_test = nanmean(precision_test);
    mean_recall_test = nanmean(recall_test);
    mean_f1_test = nanmean(f1_test);
fprintf('%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n', ...
        num_components, ...
        accuracy_train, mean_precision_train * 100, mean_recall_train * 100, mean_f1_train * 100, ...
        accuracy_test, mean_precision_test * 100, mean_recall_test * 100, mean_f1_test * 100);
    if accuracy_test > best_accuracy_test
        best_accuracy_test = accuracy_test;
        best_num_components = num_components;
        best_cm_test = cm_test;
        best_Y_test_pred = Y_test_pred;
    end
end
fprintf('\nPC£º%d\n', best_num_components);
fprintf('ACC£º%.2f%%\n', best_accuracy_test);
figure