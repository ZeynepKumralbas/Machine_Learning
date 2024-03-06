%Zeynep Kumralbaþ 150115051
%Linearization with Weight Decay Q2,Q3,Q4,Q5,Q6

k = -3;
lambda = 10^k;

%read data for Ein
fileID = fopen('zeynep_kumralbas_inData.txt','r');
formatSpec = '%f %f %f';
sizeA = [3 Inf];
A = fscanf(fileID,formatSpec,sizeA);
data_vector = A(1:2,:);
y = A(3,:);
y = transpose(y);
N = size(A,2);
data = transpose(data_vector);

%non-linear transformation
nonlinearFeatureVector = zeros(N,8);
nonlinearFeatureVector(:,1) = 1;
nonlinearFeatureVector(:,2) = data(:,1);
nonlinearFeatureVector(:,3) = data(:,2);
nonlinearFeatureVector(:,4) = data(:,1).^2;
nonlinearFeatureVector(:,5) = data(:,2).^2;
nonlinearFeatureVector(:,6) = data(:,1).*data(:,2);
nonlinearFeatureVector(:,7) = abs(data(:,1)-data(:,2));
nonlinearFeatureVector(:,8) = abs(data(:,1)+data(:,2));

w = zeros(8,1); %w vector, initially zero

X = nonlinearFeatureVector;
transpose_X = transpose(X);

%linearization with weight decay
x_reg = ( transpose_X*X + lambda*eye(8) )\transpose_X; %=inv(transpose_X*X)*transpose_X
w = x_reg*y;

nonlinearFeatureData = transpose(nonlinearFeatureVector);

%*******************Ein calculation*******************%
Ein = 0;
for data_i=1:N
    sign_result = transpose(w)*nonlinearFeatureData(:,data_i);
    if((sign_result>=0) && (y(data_i)<0))
        Ein = Ein + 1;
    elseif((sign_result<0) && (y(data_i)>=0))
        Ein = Ein + 1;
    end           
end    

totalEin = Ein/N;

%*******************Eout calculation*******************%
freshMisclassified = 0;

%read data for Eout
fileID = fopen('zeynep_kumralbas_outData.txt','r');
formatSpec = '%f %f %f';
sizeA = [3 Inf];
A = fscanf(fileID,formatSpec,sizeA);
freshData_vector = A(1:2,:);
freshy = A(3,:);
freshy = transpose(freshy);
N = size(A,2);
freshData = transpose(freshData_vector);

%non-linear transformation
nonlinearFeatureVector_freshData = zeros(N,8);
nonlinearFeatureVector_freshData(:,1) = 1;
nonlinearFeatureVector_freshData(:,2) = freshData(:,1);
nonlinearFeatureVector_freshData(:,3) = freshData(:,2);
nonlinearFeatureVector_freshData(:,4) = freshData(:,1).^2;
nonlinearFeatureVector_freshData(:,5) = freshData(:,2).^2;
nonlinearFeatureVector_freshData(:,6) = freshData(:,1).*freshData(:,2);
nonlinearFeatureVector_freshData(:,7) = abs(freshData(:,1)-freshData(:,2));
nonlinearFeatureVector_freshData(:,8) = abs(freshData(:,1)+freshData(:,2));

nonlinearFeature_freshData = transpose(nonlinearFeatureVector_freshData);

for i=1:N
    
     freshData_gResult = transpose(w)*nonlinearFeature_freshData(:,i);
     
     %+ to f and - to g
     if (freshy(i) >=0) && (freshData_gResult < 0)            
         freshMisclassified = freshMisclassified + 1;

     %- to f and + to g       
     elseif (freshy(i) < 0) && (freshData_gResult >= 0)
         freshMisclassified = freshMisclassified + 1;

     end
end

totalEout = freshMisclassified/N;

fprintf('totalEin %i', totalEin);
fprintf(' totalEout %i', totalEout);