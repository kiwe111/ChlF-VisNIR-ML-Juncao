warning off             
close all              
clear                   
clc                    
res = xlsread('Date-FvFm-Predicting', 'None');
 X = res(1:end,1:676); 
 Y = res(1:end,677); 
plot(X');
set(gcf,'color','w');
xlabel('wavelength');
ylabel('intensity');
data=[X,Y];
M=round(size(data,1)*8/10);
N=round(size(data,1))-M;
[XSelected,XRest,vSelectedRowIndex]=ks(data,round(size(data,1)*8/10));

Xcal = XSelected(:,1:end-1); 
ycal = XSelected(:,end); 
Xtest = XRest(:,1:end-1); 
ytest = XRest(:,end); 

CV=plscv(Xcal,ycal,15,10);  
plot(CV.RMSECV,'bo-','linewidth',2);
xlabel('number of latent variables');
ylabel('RMSECV');
set(gcf,'color','w');
PLS=pls(Xcal,ycal,10,'center'); 
A=10;
method='center';
N=1000; 
Q=2;
Frog=randomfrog_pls(Xcal,ycal,A,method,2000,Q);
plot(Frog.probability);
xlabel('variable index');
ylabel('selection probability');
set(gcf,'color','w');
index_probability=find(Frog.probability>=0.10);
[w,e]=size(index_probability);
for i=1:e
Xcal_RF(:,i)=Xcal(:,index_probability(i));
Xtest_RF(:,i)=Xtest(:,index_probability(i));
end