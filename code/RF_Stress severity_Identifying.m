warning off             
close all               
clear                   
clc              
res = xlsread('Date-Stress severity-Identifying.xlsx', 'None');
num_iterations = 10;       
trees = 50;             
leaf_range = 1:10;      

overall_best_accuracy_test = 0; 
overall_best_leaf = 0;          
overall_best_confMat_test = []; 
for leaf = leaf_range
    fprintf('\n==== Leaf : %d ====\n', leaf);

    accuracy_train_all = zeros(num_iterations, 1);
    accuracy_test_all = zeros(num_iterations, 1);
    precision_train_all = zeros(num_iterations, 1);
    precision_test_all = zeros(num_iterations, 1);
    recall_train_all = zeros(num_iterations, 1);
    recall_test_all = zeros(num_iterations, 1);
    f1_train_all = zeros(num_iterations, 1);
    f1_test_all = zeros(num_iterations, 1);
    best_accuracy_test = 0;     
    best_confMat_test = [];     

    for iter = 1:num_iterations
        fprintf(' %d:\n', iter);

        X = res(2:end, 1:end-1); 
        Y = res(2:end, end);  
        data = [X, Y];

        M = round(size(data, 1) * 8 / 10); 
        N = size(data, 1) - M;            
        [XSelected, XRest, ~] = ks(data, M);

        P_train = XSelected(:, 1:end-1);
        T_train = XSelected(:, end);
        P_test = XRest(:, 1:end-1);
        T_test = XRest(:, end);

        net = TreeBagger(trees, P_train, T_train, ...
            'OOBPredictorImportance', 'on', ...
            'Method', 'classification', ...
            'OOBPrediction', 'on', ...
            'minleaf', leaf);

    
        t_sim1 = predict(net, P_train);
        t_sim2 = predict(net, P_test);

        T_sim1 = str2double(t_sim1);
        T_sim2 = str2double(t_sim2);

        accuracy_train = sum(T_sim1 == T_train) / M * 100;
        accuracy_test = sum(T_sim2 == T_test) / N * 100;

        confMat_train = confusionmat(T_train, T_sim1);
        confMat_test = confusionmat(T_test, T_sim2);

        precision_train = mean(diag(confMat_train) ./ sum(confMat_train, 2));
        recall_train = mean(diag(confMat_train) ./ sum(confMat_train, 1)');
        f1_train = 2 * (precision_train * recall_train) / (precision_train + recall_train);

        precision_test = mean(diag(confMat_test) ./ sum(confMat_test, 2));
        recall_test = mean(diag(confMat_test) ./ sum(confMat_test, 1)');
        f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test);

        accuracy_train_all(iter) = accuracy_train;
        accuracy_test_all(iter) = accuracy_test;
        precision_train_all(iter) = precision_train;
        precision_test_all(iter) = precision_test;
        recall_train_all(iter) = recall_train;
        recall_test_all(iter) = recall_test;
        f1_train_all(iter) = f1_train;
        f1_test_all(iter) = f1_test;

        if accuracy_test > best_accuracy_test
            best_accuracy_test = accuracy_test;
            best_confMat_test = confMat_test;
        end
    end

    fprintf('\n (Leaf=%d):\n', leaf);
    for iter = 1:num_iterations
        fprintf('%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n', iter, ...
            accuracy_train_all(iter), precision_train_all(iter)*100, recall_train_all(iter)*100, f1_train_all(iter)*100, ...
            accuracy_test_all(iter), precision_test_all(iter)*100, recall_test_all(iter)*100, f1_test_all(iter)*100);
    end

    if best_accuracy_test > overall_best_accuracy_test
        overall_best_accuracy_test = best_accuracy_test;
        overall_best_leaf = leaf;
        overall_best_confMat_test = best_confMat_test;
    end
end
disp(overall_best_confMat_test);
figure;
confusionchart(overall_best_confMat_test, ...
               'Title', sprintf('Confusion Matrix (Best Leaf=%d, Accuracy=%.2f%%)', ...
                                overall_best_leaf, overall_best_accuracy_test), ...
               'RowSummary', 'row-normalized', ...
               'ColumnSummary', 'column-normalized');
title(sprintf('Confusion Matrix for Best Leaf=%d\nHighest Accuracy=%.2f%%', ...
              overall_best_leaf, overall_best_accuracy_test));
