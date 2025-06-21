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


k_range = 3:30;  
best_k = 0;      
best_R2_test = -inf; 
best_error_test = inf; 
best_RPD_test = 0; 

R2_train_all = zeros(length(k_range), 1);
R2_test_all = zeros(length(k_range), 1);
RMSE_train_all = zeros(length(k_range), 1);
RMSE_test_all = zeros(length(k_range), 1);
RPD_all = zeros(length(k_range), 1);

for k = k_range
    
    [~, ~, ~, ~, betaPLS, ~, ~, ~] = plsregress(p_train, t_train, k);

    t_sim1 = [ones(M, 1), p_train] * betaPLS;
    t_sim2 = [ones(N, 1), p_test ] * betaPLS;

    T_sim1 = mapminmax('reverse', t_sim1, ps_output);
    T_sim2 = mapminmax('reverse', t_sim2, ps_output);

    error1 = sqrt(sum((T_sim1' - T_train).^2) / M);
    error2 = sqrt(sum((T_sim2' - T_test).^2) / N);

    R1 = 1 - norm(T_train - T_sim1')^2 / norm(T_train - mean(T_train))^2;
    R2 = 1 - norm(T_test - T_sim2')^2 / norm(T_test - mean(T_test))^2;

    RPD = sqrt(1 / (1 - R2));
    R2_train_all(k - 2) = R1; 
    R2_test_all(k - 2) = R2;
    RMSE_train_all(k - 2) = error1;
    RMSE_test_all(k - 2) = error2;
    RPD_all(k - 2) = RPD;

    if R2 > best_R2_test
        best_R2_test = R2;
        best_k = k;
        best_error_test = error2;
        best_RPD_test = RPD;
        best_betaPLS = betaPLS;
    end
    disp(['k = ', num2str(k)]);
    disp(['R2C£º', num2str(R1)]);
    disp(['R2P£º', num2str(R2)]);
    disp(['RMSEC£º', num2str(error1)]);
    disp(['RMSEP£º', num2str(error2)]);
    disp(['RPD£º', num2str(RPD)]);
end

disp(['k£º', num2str(best_k)])
disp(['R2£º', num2str(best_R2_test)])
disp(['RMSE£º', num2str(best_error_test)])
disp(['RPD£º', num2str(best_RPD_test)])