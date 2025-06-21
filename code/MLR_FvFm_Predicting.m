warning off            
close all              
clear                
clc                     

res = xlsread('Date-FvFm-Predicting', 'None');
X = res(1:end, 1:end-1); 
data = [X, Y];
M = round(size(data, 1) * 8 / 10); 
N = round(size(data, 1)) - M;      
[XSelected, XRest, vSelectedRowIndex] = ks(data, round(size(data, 1) * 8 / 10));
P_train = XSelected(:, 1:end-1)'; 
T_train = XSelected(:, end)';     
P_test = XRest(:, 1:end-1)';     
T_test = XRest(:, end)';         
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);
[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);
p_train = p_train'; p_test = p_test';
t_train = t_train'; t_test = t_test';

lambda_values = logspace(-5, 2, 100);
cv = cvpartition(size(P_train, 1), 'KFold', 5);
min_RMSE = inf;
best_lambda = 0;

for i = 1:length(lambda_values)
    lambda = lambda_values(i);
    RMSE_val = zeros(cv.NumTestSets, 1);
    for j = 1:cv.NumTestSets
     
        train_idx = training(cv, j);
        val_idx = test(cv, j);
        
        X_train_cv = P_train(train_idx, :);
        Y_train_cv = T_train(train_idx, :);
        X_val_cv = P_train(val_idx, :);
        Y_val_cv = T_train(val_idx, :);
        B_ridge = (X_train_cv' * X_train_cv + lambda * eye(size(X_train_cv, 2))) \ (X_train_cv' * Y_train_cv);
        
        Y_val_pred = X_val_cv * B_ridge;
        RMSE_val(j) = sqrt(mean((Y_val_pred - Y_val_cv).^2));
    end
    avg_RMSE = mean(RMSE_val);
    if avg_RMSE < min_RMSE
        min_RMSE = avg_RMSE;
        best_lambda = lambda;
    end
end

disp(['lambda: ', num2str(best_lambda)]);

B_ridge_best = (P_train' * P_train + best_lambda * eye(size(P_train, 2))) \ (P_train' * T_train);

T_train_pred = P_train * B_ridge_best; 
T_test_pred = P_test * B_ridge_best;   

T_train_pred = mapminmax('reverse', T_train_pred', ps_output)';
T_test_pred = mapminmax('reverse', T_test_pred', ps_output)';
T_train = mapminmax('reverse', T_train', ps_output)';
T_test = mapminmax('reverse', T_test', ps_output)';

RMSE_train = sqrt(mean((T_train - T_train_pred).^2));
RMSE_test = sqrt(mean((T_test - T_test_pred).^2));
R2_train = 1 - sum((T_train - T_train_pred).^2) / sum((T_train - mean(T_train)).^2);
R2_test = 1 - sum((T_test - T_test_pred).^2) / sum((T_test - mean(T_test)).^2);
std_train = std(T_train); 
std_test = std(T_test);   
RPD_test = std_test / RMSE_test;
disp(['RMSEC: ', num2str(RMSE_train, '%.4f')]);
disp(['RMSEP: ', num2str(RMSE_test, '%.4f')]);
disp(['R²C: ', num2str(R2_train, '%.4f')]);
disp(['R²P: ', num2str(R2_test, '%.4f')]);
disp(['RPD: ', num2str(RPD_test, '%.4f')]);

figure;
subplot(1, 2, 1);
plot(T_train, 'r-*', 'LineWidth', 1.5); hold on;
plot(T_train_pred, 'b-o', 'LineWidth', 1.5);
grid on;

subplot(1, 2, 2);
plot(T_test, 'r-*', 'LineWidth', 1.5); hold on;
plot(T_test_pred, 'b-o', 'LineWidth', 1.5);
grid on;

figure;
subplot(1, 2, 1);
scatter(T_train, T_train_pred, 'b');
hold on; plot(xlim, ylim, '--k');
grid on;

subplot(1, 2, 2);
scatter(T_test, T_test_pred, 'b');
hold on; plot(xlim, ylim, '--k'); 
grid on;
