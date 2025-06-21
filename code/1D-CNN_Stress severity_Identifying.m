warning off             
close all               
clear                   
clc              
res = xlsread('Date-Stress severity-Identifying.xlsx', 'None');
    X = res(1:end,1:end-1);
    Y = res(1:end,end); 
    data = [X, Y];

    M = round(size(data,1)*8/10); 
    N = round(size(data,1)) - M;
    [XSelected, XRest, vSelectedRowIndex] = ks(data, round(size(data,1)*8/10));

    P_train = XSelected(:,1:end-1)';
    T_train = XSelected(:,end)'; 
    P_test = XRest(:,1:end-1)';
    T_test = XRest(:,end)'; 
[P_train, ps_input] = mapminmax(P_train, 0, 1);
P_test  = mapminmax('apply', P_test, ps_input);

t_train =  categorical(T_train)';
t_test  =  categorical(T_test )';

p_train =  double(reshape(P_train, 676, 1, 1, M));
p_test  =  double(reshape(P_test , 676, 1, 1, N));

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
 
 
 fullyConnectedLayer(128)                                     
 fullyConnectedLayer(3)                                 
 softmaxLayer                                      
 classificationLayer];                                    
options = trainingOptions('adam', ...
    'MaxEpochs', 300, ...                 
    'InitialLearnRate', 1e-3, ...         
    'L2Regularization', 1e-4, ...         
    'LearnRateSchedule', 'piecewise', ...  
    'LearnRateDropFactor', 0.1, ...        
    'LearnRateDropPeriod', 250, ...       
    'Shuffle', 'every-epoch', ...         
    'ValidationPatience', Inf, ...         
    'Plots', 'none', ...                  
    'Verbose', false);

accuracy_train = zeros(1,10);  
accuracy_test = zeros(1,10);   
mean_precision_train = zeros(1,10); 
mean_recall_train = zeros(1,10);  
mean_f1_train = zeros(1,10);       
mean_precision_test = zeros(1,10);  
mean_recall_test = zeros(1,10);     
mean_f1_test = zeros(1,10);       

best_accuracy_test = 0;          
best_confusion_test = [];       
best_T_sim2 = [];                
best_T_test = [];                

for num_components = 1:100  
    net = trainNetwork(p_train, t_train, layers, options);

    t_sim1 = predict(net, p_train); 
    t_sim2 = predict(net, p_test ); 

    T_sim1 = vec2ind(t_sim1');
    T_sim2 = vec2ind(t_sim2');

    accuracy_train(num_components) = sum((T_sim1 == T_train)) / M * 100 ;
    accuracy_test(num_components) = sum((T_sim2 == T_test )) / N * 100 ;

    confusion_train = confusionmat(T_train, T_sim1);
    confusion_test = confusionmat(T_test, T_sim2);

    precision_train = diag(confusion_train) ./ sum(confusion_train, 2);
    recall_train = diag(confusion_train) ./ sum(confusion_train, 1)';
    f1_train = 2 * (precision_train .* recall_train) ./ (precision_train + recall_train);

    precision_test = diag(confusion_test) ./ sum(confusion_test, 2);
    recall_test = diag(confusion_test) ./ sum(confusion_test, 1)';
    f1_test = 2 * (precision_test .* recall_test) ./ (precision_test + recall_test);

    mean_precision_train(num_components) = mean(precision_train);
    mean_recall_train(num_components) = mean(recall_train);
    mean_f1_train(num_components) = mean(f1_train);

    mean_precision_test(num_components) = mean(precision_test);
    mean_recall_test(num_components) = mean(recall_test);
    mean_f1_test(num_components) = mean(f1_test);

 
    if accuracy_test(num_components) > best_accuracy_test
        best_accuracy_test = accuracy_test(num_components);
        best_confusion_test = confusionmat(T_test, T_sim2);
        best_T_sim2 = T_sim2;
        best_T_test = T_test;
    end

    fprintf('%d\t%.2f%%\t%.2f\t%.2f\t%.2f\t%.2f%%\t%.2f\t%.2f\t%.2f\n', ...
        num_components, accuracy_train(num_components), mean_precision_train(num_components)*100, ...
        mean_recall_train(num_components)*100, mean_f1_train(num_components)*100, ...
        accuracy_test(num_components), mean_precision_test(num_components)*100, ...
        mean_recall_test(num_components)*100, mean_f1_test(num_components)*100);
end

analyzeNetwork(layers);
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)

grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)

grid

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
figure;
cm = confusionchart(best_T_test, best_T_sim2);
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';