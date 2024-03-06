%Zeynep Kumralbaþ 150115051%

%2d Gaussian distribution

mean = [0 0];  
covarianceMatrix = [1 0; 0 1];
samplesNo = 100;
rng('default')                                      %uses the same random number
sample = mvnrnd(mean,covarianceMatrix,samplesNo);   %generate 100 samples from 2d Gaussian distribution
subplot(3,2,1)
plot(sample(:,1),sample(:,2),'*')
title('Fig. 1')
xlabel('x1')
ylabel('x2')
hold on
axis([-5 5 -5 5])

mean = [1 -1];
covarianceMatrix = [1 0; 0 1];
rng('default')                                      %uses the same random number
sample = mvnrnd(mean,covarianceMatrix,samplesNo);   %generate 100 samples from 2d Gaussian distribution
subplot(3,2,2)
plot(sample(:,1),sample(:,2),'*')
title('Fig. 2')
xlabel('x1')
ylabel('x2')
hold on
axis([-5 5 -5 5])

mean = [0 0];
covarianceMatrix = [2 0; 0 2];
rng('default')                                      %uses the same random number
sample = mvnrnd(mean,covarianceMatrix,samplesNo);   %generate 100 samples from 2d Gaussian distribution
subplot(3,2,3)
plot(sample(:,1),sample(:,2),'*')
title('Fig. 3')
xlabel('x1')
ylabel('x2')
hold on
axis([-5 5 -5 5])

mean = [0 0];
covarianceMatrix = [2 0.2; 0.2 2];
rng('default')                                      %uses the same random number
sample = mvnrnd(mean,covarianceMatrix,samplesNo);   %generate 100 samples from 2d Gaussian distribution
subplot(3,2,4)
plot(sample(:,1),sample(:,2),'*')
title('Fig. 4')
xlabel('x1')
ylabel('x2')
hold on
axis([-5 5 -5 5])

mean = [0 0];
covarianceMatrix = [2 -0.2; -0.2 2];
rng('default')                                      %uses the same random number
sample = mvnrnd(mean,covarianceMatrix,samplesNo);   %generate 100 samples from 2d Gaussian distribution
subplot(3,2,5)
plot(sample(:,1),sample(:,2),'*')
title('Fig. 5')
xlabel('x1')
ylabel('x2')
axis([-5 5 -5 5])