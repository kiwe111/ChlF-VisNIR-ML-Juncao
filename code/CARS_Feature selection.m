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

MCCV=plsmccv(Xcal,Ycal,15,'center',1000,0.6);
sCARS=scarspls(Xcal,Ycal,MCCV.optPC,10,'center',50); 
plotcars(sCARS);
SelectedVariables=sCARS.vsel;

function CARS=carspls(X,y,A,fold,method,num) 

tic;
if nargin<6;num=50;end;
if nargin<5;method='center';end;
if nargin<4;fold=5;end;
if nargin<3;A=2;end;
[Mx,Nx]=size(X);
A=min([Mx Nx A]);
index=1:Nx;
ratio=0.9;
r0=1;
r1=2/Nx;
Vsel=1:Nx;
Q=floor(Mx*ratio);
W=zeros(Nx,num);
Ratio=zeros(1,num);
b=log(r0/r1)/(num-1);  a=r0*exp(b);
for iter=1:num
     
     perm=randperm(Mx);   
     Xcal=X(perm(1:Q),:); ycal=y(perm(1:Q));   %+++ Monte-Carlo Sampling.
     
     PLS=pls(Xcal(:,Vsel),ycal,A,method);    %+++ PLS model
     w=zeros(Nx,1);coef=PLS.regcoef_original(1:end-1,end);
     w(Vsel)=coef;W(:,iter)=w; 
     w=abs(w);                                  %+++ weights
     [ws,indexw]=sort(-w);                      %+++ sort weights
     
     ratio=a*exp(-b*(iter+1));                      %+++ Ratio of retained variables.
     Ratio(iter)=ratio;
     K=round(Nx*ratio);  
     
     
     w(indexw(K+1:end))=0;                      %+++ Eliminate some variables with small coefficients.  
     
     Vsel=randsample(Nx,Nx,true,w);                 %+++ Reweighted Sampling from the pool of retained variables.                 
     Vsel=unique(Vsel);              
     fprintf('The %dth variable sampling finished.\n',iter);    %+++ Screen output.
 end

%+++  Cross-Validation to choose an optimal subset;
RMSECV=zeros(1,num);
Q2_max=zeros(1,num);
LV=zeros(1,num);
for i=1:num
   vsel=find(W(:,i)~=0);
 
   CV=plscv(X(:,vsel),y,A,fold,method,0);  
   RMSECV(i)=CV.RMSECV_min;
   Q2_max(i)=CV.Q2_max;   
   
   LV(i)=CV.optLV;
   fprintf('The %d/%dth subset finished.\n',i,num);
end
[RMSECV_min,indexOPT]=min(RMSECV);
Q2_max=max(Q2_max);

%+++ save results;
time=toc;
%+++ output
CARS.W=W;
CARS.time=time;
CARS.RMSECV=RMSECV;
CARS.RMSECV_min=RMSECV_min;
CARS.Q2_max=Q2_max;
CARS.iterOPT=indexOPT;
CARS.optLV=LV(indexOPT);
CARS.ratio=Ratio;
CARS.vsel=find(W(:,indexOPT)~=0)';



function sel=weightsampling_in(w)
%Bootstrap sampling
%2007.9.6,H.D. Li.

w=w/sum(w);
N1=length(w);
min_sec(1)=0; max_sec(1)=w(1);
for j=2:N1
   max_sec(j)=sum(w(1:j));
   min_sec(j)=sum(w(1:j-1));
end
% figure;plot(max_sec,'r');hold on;plot(min_sec);
      
for i=1:N1
  bb=rand(1);
  ii=1;
  while (min_sec(ii)>=bb | bb>max_sec(ii)) & ii<N1;
    ii=ii+1;
  end
    sel(i)=ii;
end      % w is related to the bootstrap chance
