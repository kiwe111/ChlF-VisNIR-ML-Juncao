warning off             
close all               
clear                   
clc              
res = xlsread('Date-Stress severity-Identifying.xlsx', 'None');
 X = res(1:end, 1:end-1); 
    Y = res(1:end, end);  
    data = [X, Y];

    M = round(size(data, 1) * 8 / 10); 
    N = round(size(data, 1)) - M;     
    [XSelected, XRest, ~] = ks(data, round(size(data, 1) * 8 / 10));

    P_train = XSelected(:, 1:end-1)'; 
    T_train = XSelected(:, end)';    
    P_test = XRest(:, 1:end-1)';     
    T_test = XRest(:, end)';         
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test  = mapminmax('apply', P_test, ps_input);
t_train = ind2vec(T_train);
t_test  = ind2vec(T_test );

inputnum  = size(p_train, 1);  
hiddennum = 6;                 
outputnum = size(t_train, 1); 

net = newff(p_train, t_train, hiddennum);

net.trainParam.epochs     = 100;     
net.trainParam.goal       = 1e-6;      
net.trainParam.lr         = 0.01;     
net.trainParam.showWindow = 0;       

c1      = 4.494;       
c2      = 4.494;    
maxgen  =   30;       
sizepop =    5;      
Vmax    =  1.0;      
Vmin    = -1.0;     
popmax  =  2.0;    
popmin  = -2.0;      

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
        if rand > 0.95
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
net.Iw{1, 1} = reshape(w1, hiddennum, inputnum );
net.Lw{2, 1} = reshape(w2, outputnum, hiddennum);
net.b{1}     = reshape(B1, hiddennum, 1);
net.b{2}     = B2';

net.trainParam.showWindow = 1;     

net = train(net, p_train, t_train);

t_sim1 = sim(net, p_train);
t_sim2 = sim(net, p_test );

T_sim1 = vec2ind(t_sim1);
T_sim2 = vec2ind(t_sim2);

[T_train, index_1] = sort(T_train);
[T_test , index_2] = sort(T_test );

T_sim1 = T_sim1(index_1);
T_sim2 = T_sim2(index_2);
error1 = sum((T_sim1 == T_train)) / M * 100 ;
error2 = sum((T_sim2 == T_test )) / N * 100 ;

figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
grid

figure
plot(1: length(BestFit), BestFit, 'LineWidth', 1.5);
grid on
figure
cm = confusionchart(T_train, T_sim1);
cm.Title = 'Confusion Matrix for Train Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
    
figure
cm = confusionchart(T_test, T_sim2);
cm.Title = 'Confusion Matrix for Test Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
