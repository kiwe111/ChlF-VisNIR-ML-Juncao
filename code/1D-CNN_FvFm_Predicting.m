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

p_train =  double(reshape(p_train, 676, 1, 1, M));
p_test  =  double(reshape(p_test , 676, 1, 1, N));
t_train =  double(t_train)';
t_test  =  double(t_test )';

layers = [
 imageInputLayer([676, 1, 1])                               
 convolution2dLayer([7, 1], 64, 'Padding', 'same')         
 batchNormalizationLayer                                                                            
 reluLayer
 maxPooling2dLayer([3, 1], 'Stride', [2, 1])                
 convolution2dLayer([5, 1], 32, 'Padding', 'same')         
 batchNormalizationLayer                                                                            
 reluLayer
 maxPooling2dLayer([3, 1], 'Stride', [2, 1])                
 convolution2dLayer([3, 1], 16, 'Padding', 'same')          
 batchNormalizationLayer                                                                                  
 reluLayer
 
 dropoutLayer(0.2)                                 
 fullyConnectedLayer(128)                        
 fullyConnectedLayer(1)                            
 regressionLayer];                              

options = trainingOptions('adam', ...
    'MaxEpochs', 250, ...                  
    'InitialLearnRate', 1e-3, ...         
    'L2Regularization', 1e-4, ...          
    'LearnRateSchedule', 'piecewise', ...  
    'LearnRateDropFactor', 0.1, ...       
    'LearnRateDropPeriod', 200, ...        
    'Shuffle', 'every-epoch', ...          
    'ValidationPatience', Inf, ...         
    'Plots', 'none', ...                  
    'Verbose', false);

net = trainNetwork(p_train, t_train, layers, options);

t_sim1 = predict(net, p_train);
t_sim2 = predict(net, p_test );

T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);

error1 = sqrt(sum((T_sim1' - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2' - T_test ).^2) ./ N);

analyzeNetwork(layers)

figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
grid

R1 = 1 - norm(T_train - T_sim1')^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test  - T_sim2')^2 / norm(T_test  - mean(T_test ))^2;

disp(['R2C：', num2str(R1)])
disp(['R2P：', num2str(R2)])

%  RMSE
disp(['RMSEC：', num2str(error1)])
disp(['RMSEP：', num2str(error2)])

sz = 25;
c = 'b';
figure
scatter(T_train, T_sim1, sz, c)
hold on
plot(xlim, ylim, '--k')
xlim([min(T_train) max(T_train)])
ylim([min(T_sim1) max(T_sim1)])

figure
scatter(T_test, T_sim2, sz, c)
hold on
plot(xlim, ylim, '--k')
xlim([min(T_test) max(T_test)])
ylim([min(T_sim2) max(T_sim2)])