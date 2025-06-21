warning off             
close all               
clear                   
clc              
res = xlsread('Date-Stress severity-Identifying.xlsx', 'None');
X = res(2:end, 1:end-1); 
Y = res(2:end, end);  
data = [X, Y];

M = round(size(data, 1) * 8 / 10); 
N = round(size(data, 1)) - M;    
[XSelected, XRest, vSelectedRowIndex] = ks(data, round(size(data, 1) * 8 / 10));

P_train = XSelected(:, 1:end - 1)';
T_train = XSelected(:, end)';    
P_test = XRest(:, 1:end - 1)';  
T_test = XRest(:, end)';        
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);
t_train = T_train;
t_test = T_test;

p_train = p_train'; p_test = p_test';
t_train = t_train'; t_test = t_test';

pso_option.c1 = 1.5;                   
pso_option.c2 = 1.7;                   
pso_option.maxgen = 100;              
pso_option.sizepop = 5;                
pso_option.k = 0.6;                   
pso_option.wV = 1;                     
pso_option.wP = 1;                    
pso_option.v = 3;                    

pso_option.popcmax = 500;             
pso_option.popcmin = 0.1;            
pso_option.popgmax = 500;             
pso_option.popgmin = 0.1;             

[bestacc, bestc, bestg] = pso_svm_class(t_train, p_train, pso_option);

cmd = [' -c ', num2str(bestc), ' -g ', num2str(bestg)];
model = svmtrain(t_train, p_train, cmd);
T_sim1 = svmpredict(t_train, p_train, model);
T_sim2 = svmpredict(t_test, p_test, model);
[T_train, index_1] = sort(T_train);
[T_test, index_2] = sort(T_test);

T_sim1 = T_sim1(index_1);
T_sim2 = T_sim2(index_2);

accuracy_train = sum((T_sim1' == T_train)) / M * 100;
accuracy_test = sum((T_sim2' == T_test)) / N * 100;


confMat_train = confusionmat(T_train, T_sim1');
confMat_test = confusionmat(T_test, T_sim2');

precision_train = diag(confMat_train) ./ sum(confMat_train, 2);
recall_train = diag(confMat_train) ./ sum(confMat_train, 1)';
f1_train = 2 * (precision_train .* recall_train) ./ (precision_train + recall_train);

precision_test = diag(confMat_test) ./ sum(confMat_test, 2);
recall_test = diag(confMat_test) ./ sum(confMat_test, 1)';
f1_test = 2 * (precision_test .* recall_test) ./ (precision_test + recall_test);

precision_train(isnan(precision_train)) = 0;
recall_train(isnan(recall_train)) = 0;
f1_train(isnan(f1_train)) = 0;

precision_test(isnan(precision_test)) = 0;
recall_test(isnan(recall_test)) = 0;
f1_test(isnan(f1_test)) = 0;

fprintf('%.2f%%\t%.2f\t%.2f\t%.2f\t%.2f%%\t%.2f\t%.2f\t%.2f\n', ...
    accuracy_train, mean(precision_train)*100, mean(recall_train)*100, mean(f1_train)*100, ...
    accuracy_test, mean(precision_test)*100, mean(recall_test)*100, mean(f1_test)*100);


figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
grid

figure
cm1 = confusionchart(T_train, T_sim1);
cm1.Title = 'Confusion Matrix for Train Data';
cm1.ColumnSummary = 'column-normalized';
cm1.RowSummary = 'row-normalized';

figure
cm2 = confusionchart(T_test, T_sim2);
cm2.Title = sprintf('Confusion Matrix for Test Data (Best Params: c=%.2f, g=%.2f)', bestc, bestg);
cm2.ColumnSummary = 'column-normalized';
cm2.RowSummary = 'row-normalized';

