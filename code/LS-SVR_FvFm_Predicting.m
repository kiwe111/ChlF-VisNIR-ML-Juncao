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

pso_option.c1      = 1.5;                   
pso_option.c2      = 1.7;                    
pso_option.maxgen  = 100;                    
pso_option.sizepop =  10;                    
pso_option.k  = 0.6;                         
pso_option.wV = 1;                           
pso_option.wP = 1;                           
pso_option.v  = 5;                          

pso_option.popcmax = 100;                    
pso_option.popcmin = 0.1;                    
pso_option.popgmax = 100;                    
pso_option.popgmin = 0.1;                   

[bestacc, bestc, bestg] = psoSVMcgForRegress(t_train, p_train, pso_option);

cmd = [' -t 2 ',' -c ',num2str(bestc),' -g ',num2str(bestg),' -s 3 -p 0.01 '];
model = svmtrain(t_train, p_train, cmd);

[t_sim1, error_1] = svmpredict(t_train, p_train, model);
[t_sim2, error_2] = svmpredict(t_test , p_test , model);

T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);

error1 = sqrt(sum((T_sim1' - T_train).^2) ./ M)
error2 = sqrt(sum((T_sim2' - T_test ).^2) ./ N)

figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
grid
figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
grid

R1 = 1 - norm(T_train - T_sim1')^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test  - T_sim2')^2 / norm(T_test  - mean(T_test ))^2;
RPD=sqrt(1/(1-R2));
disp(['R2C：', num2str(R1)])
disp(['R2P：', num2str(R2)])
disp(['RPD：',  num2str(RPD)])
mae1 = sum(abs(T_sim1' - T_train)) ./ M ;
mae2 = sum(abs(T_sim2' - T_test )) ./ N ;
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