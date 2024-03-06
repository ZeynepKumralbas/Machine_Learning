%Zeynep Kumralbaþ 150115051
%Support Vector Machines

%read data for train
fileID = fopen('zeynep_kumralbas_train.txt','r');
formatSpec = '%f %f %f';
sizeA = [3 Inf];
A = fscanf(fileID,formatSpec,sizeA);
digit = transpose(A(1,:));
intensity = transpose(A(2,:));
symmetry = transpose(A(3,:));
features = transpose(A(2:3,:));

%read data for test
fileID_test = fopen('zeynep_kumralbas_test.txt','r');
formatSpec_test = '%f %f %f';
sizeA_test = [3 Inf];
A_test = fscanf(fileID_test,formatSpec_test,sizeA_test);
digit_test = transpose(A_test(1,:));
intensity_test = transpose(A_test(2,:));
symmetry_test = transpose(A_test(3,:));
features_test = transpose(A_test(2:3,:));


%list, array indexes start from 1, digits start from 0

yForDigits = {};    %labels list

%digits in 0-9, one-to-all
for k=1:10
    for i=1:size(digit,1)
       if digit(i) == k-1       %if digit, label as +1
           yForDigits{k}(i)=1;
       else                     %others, label as -1
           yForDigits{k}(i)=-1; 
       end
    end
end

model_fitcsvm = {};         %models list
predictedYForDigit = {};    %predicted labels for digits list
for k=1:10
    %Models: one-to-all, C=0.01, polynomial, degree of polynomial is 2
    model_fitcsvm{k} = fitcsvm(features, yForDigits{k}, 'boxconstraint', 0.01, 'KernelFunction', 'polynomial', 'PolynomialOrder', 2);
    %predict labels based on model
    predictedYForDigit{k} = predict(model_fitcsvm{k},features);
end

%1-vs-all scatter plot
figure;
 hgscatter = gscatter(features(:,1),features(:,2), yForDigits{2});
 hold on;
 h_sv = plot(model_fitcsvm{2}.SupportVectors(:,1), model_fitcsvm{2}.SupportVectors(:,2) , 'ko', 'markersize', 8);
% X : features, Y : lables
hold off;

 Ein = {};  %Ein list

 for k=1:10
    ein = 0;
    for i=1:size(digit,1)
        if yForDigits{k}(i) ~= predictedYForDigit{k}(i) %if train label and predicted label are not the same
            ein = ein + 1;
        end
    end
 Ein{k} = ein/size(digit,1);   
 end
 
for i=1:10
    fprintf('Ein for %d-versus-all: %f, number of support vectors:%d\n',i-1,Ein{i},size(model_fitcsvm{i}.SupportVectors,1));
end

%**************************1 vs 5**************************
innerIndex = 1;
for i=1:size(digit,1)
       if digit(i) == 1                                 %if digit is 1, label as +1
           yForDigit_1vs5(innerIndex,1)=1;
           features_1vs5(innerIndex,1)=features(i,1);   %get features of digit 1
           features_1vs5(innerIndex,2)=features(i,2);
           innerIndex = innerIndex+1;
       elseif digit(i) == 5                             %if digit is 5, label as -1
           yForDigit_1vs5(innerIndex,1)=-1;
           features_1vs5(innerIndex,1)=features(i,1);   %get features of digit 5
           features_1vs5(innerIndex,2)=features(i,2);
           innerIndex = innerIndex+1;
       end
end

model_fitcsvm_1vs5 = {};    %1-versus-5 models list 
%models for 1-versus-5, polynomial, Q=2, C=0.001, C=0.01, C=0.1, C=1
model_fitcsvm_1vs5{1} = fitcsvm(features_1vs5, yForDigit_1vs5, 'boxconstraint', 0.001, 'KernelFunction', 'polynomial', 'PolynomialOrder', 2);
model_fitcsvm_1vs5{2} = fitcsvm(features_1vs5, yForDigit_1vs5, 'boxconstraint', 0.01, 'KernelFunction', 'polynomial', 'PolynomialOrder', 2);
model_fitcsvm_1vs5{3} = fitcsvm(features_1vs5, yForDigit_1vs5, 'boxconstraint', 0.1, 'KernelFunction', 'polynomial', 'PolynomialOrder', 2);
model_fitcsvm_1vs5{4} = fitcsvm(features_1vs5, yForDigit_1vs5, 'boxconstraint', 1, 'KernelFunction', 'polynomial', 'PolynomialOrder', 2);

 

%print number of support vector machines, Question 5
fprintf('1-vs-5 with Q=2\n');
fprintf('  C=0.001 numberOfSupportVectors:%d\n',size(model_fitcsvm_1vs5{1}.SupportVectors,1));
fprintf('  C=0.01  numberOfSupportVectors:%d\n',size(model_fitcsvm_1vs5{2}.SupportVectors,1));
fprintf('  C=0.1   numberOfSupportVectors:%d\n',size(model_fitcsvm_1vs5{3}.SupportVectors,1));
fprintf('  C=1     numberOfSupportVectors:%d\n',size(model_fitcsvm_1vs5{4}.SupportVectors,1));

predictedYForDigit_1vs5 = {};       %predicted labels for digits list
for k=1:4
    %predict labels based on model
    predictedYForDigit_1vs5{k} = predict(model_fitcsvm_1vs5{k},features_1vs5);
end


Ein_1vs5 = {}; %Ein 1vs5 list
 for k=1:4
    ein = 0;
    for i=1:size(yForDigit_1vs5,1)
        if yForDigit_1vs5(i) ~= predictedYForDigit_1vs5{k}(i) %if train label for 1-versus-5 and predicted label are not the same
            ein = ein + 1;
        end
    end
 
 Ein_1vs5{k} = ein/size(yForDigit_1vs5,1); 
 end
 
 %printf Ein for 1-vs-5 Q=2, Question 5
 fprintf('Ein for 1-vs-5 with Q=2\n');
 fprintf('C=0.001 Ein:%f\n',Ein_1vs5{1});
 fprintf('C=0.01  Ein:%f\n',Ein_1vs5{2});
 fprintf('C=0.1   Ein:%f\n',Ein_1vs5{3});
 fprintf('C=1     Ein:%f\n',Ein_1vs5{4});
 
 %**************************1 vs 5 Eout**************************
 innerIndex = 1;
for i=1:size(digit_test,1)
       if digit_test(i) == 1                            %if digit is 1, label as +1
           test_yForDigit_1vs5(innerIndex,1)=1;
           test_features_1vs5(innerIndex,1)=features_test(i,1); %get features of digit 1
           test_features_1vs5(innerIndex,2)=features_test(i,2);
           innerIndex = innerIndex+1;
       elseif digit_test(i) == 5                         %if digit is 5, label as -1
           test_yForDigit_1vs5(innerIndex,1)=-1;
           test_features_1vs5(innerIndex,1)=features_test(i,1); %get features of digit 5
           test_features_1vs5(innerIndex,2)=features_test(i,2);
           innerIndex = innerIndex+1;
       end
end

test_predictedYForDigit_1vs5 = {};      %predicted labels for digits list
for k=1:4
    %predict labels based on model
    test_predictedYForDigit_1vs5{k} = predict(model_fitcsvm_1vs5{k},test_features_1vs5);
end

Eout_1vs5 = {};     %Eout 1vs5 list
 for k=1:4
    eout = 0;
    for i=1:size(test_yForDigit_1vs5,1)
        if test_yForDigit_1vs5(i) ~= test_predictedYForDigit_1vs5{k}(i) %if test label for 1-versus-5 and predicted label are not the same
            eout = eout + 1;
        end
    end
 
 Eout_1vs5{k} = eout/size(test_yForDigit_1vs5,1);
 end
 
 %print Eout for 1-vs-5 Q=2, Question 5
 fprintf('Eout for 1-vs-5 with Q=2\n');
 fprintf('C=0.001 Eout:%f\n',Eout_1vs5{1});
 fprintf('C=0.01  Eout:%f\n',Eout_1vs5{2});
 fprintf('C=0.1   Eout:%f\n',Eout_1vs5{3});
 fprintf('C=1     Eout:%f\n',Eout_1vs5{4});

 
%**************************1 vs 5,  Q=2 vs Q5**************************
%C=0.0001, Q=2
model_fitcsvm_1vs5_0_0001_Q2 = fitcsvm(features_1vs5, yForDigit_1vs5, 'boxconstraint', 0.0001, 'KernelFunction', 'polynomial', 'PolynomialOrder', 2);

predictedYForDigit_1vs5_0_0001_Q2 = predict(model_fitcsvm_1vs5_0_0001_Q2,features_1vs5);

%Ein for 1vs5, Q2, C=0.0001
ein = 0;
for i=1:size(yForDigit_1vs5,1)
    if yForDigit_1vs5(i) ~= predictedYForDigit_1vs5_0_0001_Q2(i)
       ein = ein + 1;
    end
end
 Ein_1vs5_0_0001_Q2 = ein/size(yForDigit_1vs5,1); 
 
model_fitcsvm_1vs5_Q = {};  %1-versus-5 models list
%models for 1-versus-5, polynomial, Q=5, C=0.0001, C=0.001, C=0.01, C=0.1, C=1
model_fitcsvm_1vs5_Q{1} = fitcsvm(features_1vs5, yForDigit_1vs5, 'boxconstraint', 0.0001, 'KernelFunction', 'polynomial', 'PolynomialOrder', 5);
model_fitcsvm_1vs5_Q{2} = fitcsvm(features_1vs5, yForDigit_1vs5, 'boxconstraint', 0.001, 'KernelFunction', 'polynomial', 'PolynomialOrder', 5);
model_fitcsvm_1vs5_Q{3} = fitcsvm(features_1vs5, yForDigit_1vs5, 'boxconstraint', 0.01, 'KernelFunction', 'polynomial', 'PolynomialOrder', 5);
model_fitcsvm_1vs5_Q{4} = fitcsvm(features_1vs5, yForDigit_1vs5, 'boxconstraint', 0.1, 'KernelFunction', 'polynomial', 'PolynomialOrder', 5);
model_fitcsvm_1vs5_Q{5} = fitcsvm(features_1vs5, yForDigit_1vs5, 'boxconstraint', 1, 'KernelFunction', 'polynomial', 'PolynomialOrder', 5);

%Ein for 1vs5, Q5, C=0.0001, C=0.001, C=0.01, C=0.1, C=1
predictedYForDigit_1vs5_Q = {};
for k=1:5
    predictedYForDigit_1vs5_Q{k} = predict(model_fitcsvm_1vs5_Q{k},features_1vs5);
end

Ein_1vs5_Q = {};
 for k=1:5
    ein = 0;
    for i=1:size(yForDigit_1vs5,1)
        if yForDigit_1vs5(i) ~= predictedYForDigit_1vs5_Q{k}(i)
            ein = ein + 1;
        end
    end
 
 Ein_1vs5_Q{k} = ein/size(yForDigit_1vs5,1); 
 end
 
 %print Ein for 1-vs-5 C=0.0001, Q=2, Q=5, Question 6
 fprintf('Ein for 1-vs-5 with C=0.0001\n');
 fprintf('Q=2 Ein:%f\n',Ein_1vs5_0_0001_Q2);
 fprintf('Q=5 Ein:%f\n',Ein_1vs5_Q{1});
 
 %print number of support vector machines for 1-vs-5 C=0.001, Q=2, Q=5, Question 6
 fprintf('number of support vector machines for 1-vs-5 with C=0.001\n');
 fprintf('Q=2 =>%f\n',size(model_fitcsvm_1vs5{1}.SupportVectors,1));
 fprintf('Q=5 =>%f\n',size(model_fitcsvm_1vs5_Q{2}.SupportVectors,1));
 
 %print Ein for 1-vs-5 C=0.01, Q=2, Q=5, Question 6
 fprintf('Ein for 1-vs-5 with C=0.01\n');
 fprintf('Q=2 Ein:%f\n',Ein_1vs5{2});
 fprintf('Q=5 Ein:%f\n',Ein_1vs5_Q{3});
 
%Eout for 1vs5, Q5, C=0.0001, C=0.001, C=0.01, C=0.1, C=1
test_predictedYForDigit_1vs5_Q = {};
for k=1:5
    test_predictedYForDigit_1vs5_Q{k} = predict(model_fitcsvm_1vs5_Q{k},test_features_1vs5);
end

Eout_1vs5_Q = {};
 for k=1:5
    eout = 0;
    for i=1:size(test_yForDigit_1vs5,1)
        if test_yForDigit_1vs5(i) ~= test_predictedYForDigit_1vs5_Q{k}(i)
            eout = eout + 1;
        end
    end
 
 Eout_1vs5_Q{k} = eout/size(test_yForDigit_1vs5,1);
 end
 
 %print Eout for 1-vs-5 C=1, Q=2, Q=5, Question 6
 fprintf('Eout for 1-vs-5 with C=1\n');
 fprintf('Q=2 Eout:%f\n',Ein_1vs5{4});
 fprintf('Q=5 Eout:%f\n',Ein_1vs5_Q{5});
 
%**************************CROSS VALIDATION**************************
%**************************CROSS VALIDATION**************************

numOfRuns = 100;
selectedC{1} = 0;
selectedC{2} = 0;
selectedC{3} = 0;
selectedC{4} = 0;
selectedC{5} = 0;

Ecv_average = {}; Ecv_average{1}=0; Ecv_average{2}=0; Ecv_average{3}=0; Ecv_average{4}=0; Ecv_average{5}=0;
for runs=1:numOfRuns %number of runs
    combinedLabelFeatures = [yForDigit_1vs5, features_1vs5];                        %combine features and labels for 1-vs-5 data
    shuffled = combinedLabelFeatures(randperm(size(combinedLabelFeatures, 1)), :);  %shufle randomy
    
    %partitions for 10-fold
    partitions = {};
    partitions{1} = shuffled(1:156,:);
    partitions{2} = shuffled(157:312,:);
    partitions{3} = shuffled(313:468,:);
    partitions{4} = shuffled(469:624,:);
    partitions{5} = shuffled(625:780,:);
    partitions{6} = shuffled(781:936,:);
    partitions{7} = shuffled(937:1092,:);
    partitions{8} = shuffled(1093:1248,:);
    partitions{9} = shuffled(1249:1404,:);
    partitions{10} = shuffled(1405:1561,:); %has one more element
    
   j=1;
   Ecv = {};
   Ecv{1} = 0;  Ecv{2} = 0;  Ecv{3} = 0;  Ecv{4} = 0; Ecv{5} = 0;
   Ecv_mean = {}; Ecv_mean{1} = 0; Ecv_mean{2} = 0; Ecv_mean{3} = 0; Ecv_mean{4} = 0; Ecv_mean{5} = 0;

   %partititon train data into 10 subsets, choose one subset as the test data,
   %use other 9 subsets as the training data
   %change the test subset respectively
   
   for i=1:10 %10 partitions
        clear cv_train_features;
        clear cv_train_labels;
        
        cv_train_features = zeros(0,2);
        cv_train_labels = zeros(0,1);
        
        cv_test = partitions{i};                %get test subset
        cv_test_features = partitions{j}(:,2:3);
        cv_test_labels = partitions{j}(:,1);
        
        for j=1:10
            if(i~=j)
                cv_train = partitions{j};    %get train subset
                cv_train_features = [cv_train_features; cv_train(:,2:3)];
                cv_train_labels = [cv_train_labels; cv_train(:,1)];
            end
        end
        
        
        model_fitcsvm_cv = {};
        %models for 1-versus-5, polynomial, Q=2, C=0.0001, C=0.001, C=0.01, C=0.1, C=1
        model_fitcsvm_cv{1} = fitcsvm(cv_train_features, cv_train_labels, 'boxconstraint', 0.0001, 'KernelFunction', 'polynomial', 'PolynomialOrder', 2);
        model_fitcsvm_cv{2} = fitcsvm(cv_train_features, cv_train_labels, 'boxconstraint', 0.001, 'KernelFunction', 'polynomial', 'PolynomialOrder', 2);
        model_fitcsvm_cv{3} = fitcsvm(cv_train_features, cv_train_labels, 'boxconstraint', 0.01, 'KernelFunction', 'polynomial', 'PolynomialOrder', 2);
        model_fitcsvm_cv{4} = fitcsvm(cv_train_features, cv_train_labels, 'boxconstraint', 0.1, 'KernelFunction', 'polynomial', 'PolynomialOrder', 2);
        model_fitcsvm_cv{5} = fitcsvm(cv_train_features, cv_train_labels, 'boxconstraint', 1, 'KernelFunction', 'polynomial', 'PolynomialOrder', 2);

        %use test data to predict the labels
        for k=1:5
            predicted_cv_test_labels{k} = predict(model_fitcsvm_cv{k},cv_test_features);
        end
        
        %Ecv
        for k=1:5
            eout = 0;
            for m=1:size(cv_test_labels,1)
                if cv_test_labels(m) ~= predicted_cv_test_labels{k}(m)
                    eout = eout + 1;
                end
            end
            Ecv{k} = Ecv{k} + eout/size(cv_test_labels,1);
        end
     
   end %10 partitions
   
   %average Ecv for all C values
   for k=1:5
       Ecv_mean{k} = Ecv{k}/10;
   end
   
   %choose minimum Ecv
   if(Ecv_mean{1}<=Ecv_mean{2})
       min_Ecv = Ecv_mean{1};
   else
       min_Ecv = Ecv_mean{2};
   end
   if(Ecv_mean{3}<=min_Ecv)
       min_Ecv = Ecv_mean{3};
   end
   if(Ecv_mean{4}<=min_Ecv)
       min_Ecv = Ecv_mean{4};
   end
   if(Ecv_mean{5}<=min_Ecv)
       min_Ecv = Ecv_mean{5};
   end
   
   %select C based on the minimum Ecv
   if(min_Ecv == Ecv_mean{1})
       selectedC{1} = selectedC{1}+1;
   elseif (min_Ecv == Ecv_mean{2})
        selectedC{2} = selectedC{2}+1;
   elseif (min_Ecv == Ecv_mean{3})
        selectedC{3} = selectedC{3}+1;
   elseif (min_Ecv == Ecv_mean{4})
        selectedC{4} = selectedC{4}+1;
   elseif (min_Ecv == Ecv_mean{5})
        selectedC{5} = selectedC{5}+1;
   end
   
   %store the average Ecv value
   for k=1:5
       Ecv_average{k} = Ecv_average{k} + Ecv_mean{k};
   end
   
end

fprintf('The number of time C is selected\n');
fprintf('C=0.0001 is selected %d times\n',selectedC{1});
fprintf('C=0.001  is selected %d times\n',selectedC{2});
fprintf('C=0.01   is selected %d times\n',selectedC{3});
fprintf('C=0.1    is selected %d times\n',selectedC{4});
fprintf('C=1      is selected %d times\n',selectedC{5});

% %get the average Ecv for numOfRuns
% for k=1:5
%     Ecv_average{k} = Ecv_average{k}/numOfRuns;
% end
% 
fprintf('The average value of Ecv over 100 runs for winning C=0.001=>%f\n',Ecv_average{2});
% %**************************RBF KERNEL**************************
% %**************************RBF KERNEL**************************
model_fitcsvm_1vs5_rbf = {};
model_fitcsvm_1vs5_rbf{1} = fitcsvm(features_1vs5, yForDigit_1vs5, 'boxconstraint', 0.01, 'KernelFunction', 'rbf');
model_fitcsvm_1vs5_rbf{2} = fitcsvm(features_1vs5, yForDigit_1vs5, 'boxconstraint', 1, 'KernelFunction', 'rbf');
model_fitcsvm_1vs5_rbf{3} = fitcsvm(features_1vs5, yForDigit_1vs5, 'boxconstraint', 100, 'KernelFunction', 'rbf');
model_fitcsvm_1vs5_rbf{4} = fitcsvm(features_1vs5, yForDigit_1vs5, 'boxconstraint', 10000, 'KernelFunction', 'rbf');
model_fitcsvm_1vs5_rbf{5} = fitcsvm(features_1vs5, yForDigit_1vs5, 'boxconstraint', 1000000, 'KernelFunction', 'rbf');

% figure;
% hgscatter = gscatter(features_1vs5(:,1),features_1vs5(:,2), yForDigit_1vs5);
% hold on;
% h_sv = plot(model_fitcsvm_1vs5_rbf{1}.SupportVectors(:,1), model_fitcsvm_1vs5_rbf{1}.SupportVectors(:,2) , 'ko', 'markersize', 8);
% X : features, Y : lables

%Ein for 1vs5, RBF Kernel, C=0.01, C=1, C=100, C=10000, C=1000000
predictedYForDigit_1vs5_rbf = {};
for k=1:5
    predictedYForDigit_1vs5_rbf{k} = predict(model_fitcsvm_1vs5_rbf{k},features_1vs5);
end

Ein_1vs5_rbf = {};
 for k=1:5
    ein = 0;
    for i=1:size(yForDigit_1vs5,1)
        if yForDigit_1vs5(i) ~= predictedYForDigit_1vs5_rbf{k}(i)
            ein = ein + 1;
        end
    end
 
 Ein_1vs5_rbf{k} = ein/size(yForDigit_1vs5,1); 
 end

%print Ein for 1-vs-5, RBF Kernel, C=0.01, C=1, C=100, C=10000, C=1000000, Question 9 
fprintf('Ein for 1-vs-5 with RBF Kernel\n');
fprintf('  C=0.01    Ein:%f\n',Ein_1vs5_rbf{1});
fprintf('  C=1       Ein:%f\n',Ein_1vs5_rbf{2});
fprintf('  C=100     Ein:%f\n',Ein_1vs5_rbf{3});
fprintf('  C=10000   Ein:%f\n',Ein_1vs5_rbf{4});
fprintf('  C=1000000 Ein:%f\n',Ein_1vs5_rbf{5});

%Eout for 1vs5, RBF Kernel, C=0.01, C=1, C=100, C=10000, C=1000000
test_predictedYForDigit_1vs5_rbf = {};
for k=1:5
    test_predictedYForDigit_1vs5_rbf{k} = predict(model_fitcsvm_1vs5_rbf{k},test_features_1vs5);
end

Eout_1vs5_rbf = {};
 for k=1:5
    eout = 0;
    for i=1:size(test_yForDigit_1vs5,1)
        if test_yForDigit_1vs5(i) ~= test_predictedYForDigit_1vs5_rbf{k}(i)
            eout = eout + 1;
        end
    end
 
 Eout_1vs5_rbf{k} = eout/size(test_yForDigit_1vs5,1);
 end
 
%print Eout for 1-vs-5, RBF Kernel, C=0.01, C=1, C=100, C=10000, C=1000000, Question 9 
fprintf('Eout for 1-vs-5 with RBF Kernel\n');
fprintf('  C=0.01    Eout:%f\n',Eout_1vs5_rbf{1});
fprintf('  C=1       Eout:%f\n',Eout_1vs5_rbf{2});
fprintf('  C=100     Eout:%f\n',Eout_1vs5_rbf{3});
fprintf('  C=10000   Eout:%f\n',Eout_1vs5_rbf{4});
fprintf('  C=1000000 Eout:%f\n',Eout_1vs5_rbf{5});