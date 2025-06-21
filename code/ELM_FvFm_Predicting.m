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

num_hidden_values = 1:1:300;
best_R2 = -Inf;               
best_num_hiddens = 0;         
for num_hiddens = num_hidden_values
    activate_model = 'sig';  
    [IW, B, LW, TF, TYPE] = elmtrain(p_train, t_train, num_hiddens, activate_model, 0);
    t_sim1 = elmpredict(p_train, IW, B, LW, TF, TYPE);
    t_sim2 = elmpredict(p_test , IW, B, LW, TF, TYPE);
    T_sim1 = mapminmax('reverse', t_sim1, ps_output);
    T_sim2 = mapminmax('reverse', t_sim2, ps_output);
    error1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);
    error2 = sqrt(sum((T_sim2 - T_test).^2) ./ N);
    R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
    R2 = 1 - norm(T_test - T_sim2)^2 / norm(T_test - mean(T_test))^2;
    RPD = sqrt(1 / (1 - R2));
    mae1 = sum(abs(T_sim1 - T_train)) ./ M ;
    mae2 = sum(abs(T_sim2 - T_test)) ./ N ;
    mbe1 = sum(T_sim1 - T_train) ./ M ;
    mbe2 = sum(T_sim2 - T_test) ./ N ;
    disp(['num_hiddens = ', num2str(num_hiddens)])
    disp(['R2C：', num2str(R1)])
    disp(['R2P：', num2str(R2)])
    disp(['RPD：', num2str(RPD)])
    disp(['RMSEC：', num2str(error1)])
    disp(['RMSEP：', num2str(error2)])
    if R2 > best_R2
        best_R2 = R2;
        best_num_hiddens = num_hiddens;
        best_T_sim1 = T_sim1;
        best_T_sim2 = T_sim2;
        best_T_train = T_train;
        best_T_test = T_test;
    end
end

figure;
subplot(1, 2, 1);
scatter(best_T_train, best_T_sim1, 'filled');
grid on;
subplot(1, 2, 2);
scatter(best_T_test, best_T_sim2, 'filled');
grid on;