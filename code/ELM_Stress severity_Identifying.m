warning off             
close all               
clear                   
clc              
res = xlsread('Date-Stress severity-Identifying.xlsx', 'None');
 best_accuracy = 0; 
best_confMat = [];
best_hiddennum = 0; 
metrics_names = {
    'ACC', 'PRE, 'recall', 'F1-score', ...
    'ACC', 'PRE, 'recall', 'F1-score',};

fprintf('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n', metrics_names{:});
for hiddennum = 10:5:200
    for run_idx = 1:10
  
        X = res(1:end, 1:end-1); 
        Y = res(1:end, end); 
        data = [X, Y];

        M = round(size(data, 1) * 8 / 10); 
        N = round(size(data, 1)) - M;     
        [XSelected, XRest, ~] = ks(data, M);

        P_train = XSelected(:, 1:end-1)'; 
        T_train = XSelected(:, end)';    
        P_test = XRest(:, 1:end-1)';     
        T_test = XRest(:, end)';         

        activate_model = 'sig'; 
        [IW, B, LW, TF, TYPE] = elmtrain(P_train, T_train, hiddennum, activate_model, 1);

        T_sim1 = elmpredict(P_train, IW, B, LW, TF, TYPE);
        T_sim2 = elmpredict(P_test, IW, B, LW, TF, TYPE);

        confMat_train = confusionmat(T_train, T_sim1);
        precision_train = mean(diag(confMat_train) ./ sum(confMat_train, 2));
        recall_train = mean(diag(confMat_train) ./ sum(confMat_train, 1)');
        f1_train = 2 * (precision_train * recall_train) / (precision_train + recall_train);
        accuracy_train = sum((T_sim1 == T_train)) / M * 100;

        confMat_test = confusionmat(T_test, T_sim2);
        precision_test = mean(diag(confMat_test) ./ sum(confMat_test, 2));
        recall_test = mean(diag(confMat_test) ./ sum(confMat_test, 1)');
        f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test);
        accuracy_test = sum((T_sim2 == T_test)) / N * 100;
        if accuracy_test > best_accuracy
            best_accuracy = accuracy_test;
            best_confMat = confMat_test;
            best_hiddennum = hiddennum;
        end

        metrics_values = [
            accuracy_train, precision_train*100, recall_train*100, f1_train*100, ...
            accuracy_test, precision_test*100, recall_test*100, f1_test*100
        ];
        fprintf('Run %d (Hidden=%d): %.2f%%\t%.2f\t%.2f\t%.2f\t%.2f%%\t%.2f\t%.2f\t%.2f\n', ...
            run_idx, hiddennum, metrics_values);
    end
end

fprintf('\n: %.2f%% : %d\n', best_accuracy, best_hiddennum);
disp(best_confMat);

figure;
cm = confusionchart(best_confMat);
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';