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

inputnum  = size(p_train, 1);  
hiddennum = 5;                 
outputnum = size(t_train,1);   

net = newff(p_train, t_train, hiddennum);

net.trainParam.epochs     = 500;      
net.trainParam.goal       = 1e-6;     
net.trainParam.lr         = 0.001;    
net.trainParam.showWindow = 0;      


c1      = 4.494;       
c2      = 4.494;       
maxgen  =   50;          
sizepop =    5;        
Vmax    =  1.0;        
Vmin    = -1.0;        
popmax  =  1.0;        
popmin  = -1.0;       

numsum = inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum;

for i = 1 : sizepop
    pop(i, :) = rands(1, numsum);  
    V(i, :) = rands(1, numsum);   
    fitness(i) = fun(pop(i, :), hiddennum, net, p_train, t_train);
end

[fitnesszbest, bestindex] = min(fitness);
zbest = pop(bestindex, :);     
gbest = pop;                  
fitnessgbest = fitness;        
BestFit = fitnesszbest;       

for i = 1 : maxgen
    for j = 1 : sizepop
        V(j, :) = V(j, :) + c1 * rand * (gbest(j, :) - pop(j, :)) + c2 * rand * (zbest - pop(j, :));
        V(j, (V(j, :) > Vmax)) = Vmax;
        V(j, (V(j, :) < Vmin)) = Vmin;
       
        pop(j, :) = pop(j, :) + 0.2 * V(j, :);
        pop(j, (pop(j, :) > popmax)) = popmax;
        pop(j, (pop(j, :) < popmin)) = popmin;
        
        pos = unidrnd(numsum);
        if rand > 0.85
            pop(j, pos) = rands(1, 1);
        end
        fitness(j) = fun(pop(j, :), hiddennum, net, p_train, t_train);

    end
    
    for j = 1 : sizepop

        if fitness(j) < fitnessgbest(j)
            gbest(j, :) = pop(j, :);
            fitnessgbest(j) = fitness(j);
        end
 
        if fitness(j) < fitnesszbest
            zbest = pop(j, :);
            fitnesszbest = fitness(j);
        end

    end

    BestFit = [BestFit, fitnesszbest];    
end

w1 = zbest(1 : inputnum * hiddennum);
B1 = zbest(inputnum * hiddennum + 1 : inputnum * hiddennum + hiddennum);
w2 = zbest(inputnum * hiddennum + hiddennum + 1 : inputnum * hiddennum ...
    + hiddennum + hiddennum * outputnum);
B2 = zbest(inputnum * hiddennum + hiddennum + hiddennum * outputnum + 1 : ...
    inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum);

net.Iw{1, 1} = reshape(w1, hiddennum, inputnum);
net.Lw{2, 1} = reshape(w2, outputnum, hiddennum);
net.b{1}     = reshape(B1, hiddennum, 1);
net.b{2}     = B2';

net.trainParam.showWindow = 1;     
net = train(net, p_train, t_train);
t_sim1 = sim(net, p_train);
t_sim2 = sim(net, p_test );
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);
error1 = sqrt(sum((T_sim1 - T_train).^2, 2)' ./ M);
error2 = sqrt(sum((T_sim2 - T_test) .^2, 2)' ./ N);

figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
grid
figure;
plot(1 : length(BestFit), BestFit, 'LineWidth', 1.5);
grid on

R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test  - T_sim2)^2 / norm(T_test  - mean(T_test ))^2;
RPD=sqrt(1/(1-R2));

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