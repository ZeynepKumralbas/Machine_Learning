%Linear Regression Algorithm%
%Q13
%Zeynep Kumralbaþ 150115051%

N = 1000;
numberOfExperiments = 1000;

x = [-1,1];

totalFreshMisclassified = 0;

for iterations = 1:numberOfExperiments
    
    data = zeros(N,3);
    y = zeros(N,1);

    %generate random data between [-1 1]
     for i = 1:N
         r = range1 + (range2-range1).*rand(2,1);
         data(i,1) = 1;
         data(i,2) = r(1,1); %x
         data(i,3) = r(2,1); %y
         %y-ax-b = 0
         if data(i,2)^2+data(i,3)^2-0.6 >=0
            y(i) = 1;
%               plot(data(i,2),data(i,3),'r+')
%               hold on;
         elseif data(i,2)^2+data(i,3)^2-0.6 <0
            y(i) = -1;
%               plot(data(i,2),data(i,3),'ro')
         end

     end
     
     %make noise
     n = N;
     noise = zeros(1,N);
     for i=1:(N*0.1)
        random = randi([1 n]);
        rand_selected = noise(random);
        while rand_selected==1
            random = randi([1 n]);
            rand_selected = noise(random);
        end
        if y(random)==-1
            y(random)=1;
        elseif y(random)==1
            y(random)=-1;
        end
        noise(random) = 1;
     end
     
%      subplot(2,1,2);
%      hold on;
%      axis([-1 1 -1 1])
     
     nonlinearFeatureVector = zeros(N,6);
     nonlinearFeatureVector(:,1) = data(:,1);
     nonlinearFeatureVector(:,2) = data(:,2);
     nonlinearFeatureVector(:,3) = data(:,3);
     nonlinearFeatureVector(:,4) = data(:,2).*data(:,3);
     nonlinearFeatureVector(:,5) = data(:,2).^2;
     nonlinearFeatureVector(:,6) = data(:,3).^2;
    w = zeros(6,1); %w vector, initially zero

    
    X = nonlinearFeatureVector;
    transpose_X = transpose(X);
    xDagger = (transpose_X*X)\transpose_X; %=inv(transpose_X*X)*transpose_X
    w = xDagger*y;
    
    g_a = -w(2,1)./w(3,1);
    g_b = -w(1,1)./w(3,1);
    y1 = g_a.*x + g_b;
%       plot(x,y1,'g');
%       subplot(2,1,2);

    %generate fresh points
    freshPoints = 1000;
    freshData = zeros(N,3);
    freshy = zeros(N,1);
    freshMisclassified = 0;
    %generate random data between [-1 1] for Eout error
     for i = 1:freshPoints
         r = range1 + (range2-range1).*rand(2,1);
         freshData(i,1) = 1;
         freshData(i,2) = r(1,1); %x
         freshData(i,3) = r(2,1); %y
         
         freshData_vector = transpose(freshData);
         
         if freshData(i,2)^2+freshData(i,3)^2-0.6 >=0
            freshy(i) = 1;

         elseif freshData(i,2)^2+freshData(i,3)^2-0.6 <0
            freshy(i) = -1;

         end

         %make noise again
         n = N;
         noisey = zeros(1,N);
         for k=1:(N*0.1)
            random = randi([1 n]);
            rand_selected = noisey(random);
            while rand_selected==1
                random = randi([1 n]);
                rand_selected = noisey(random);
            end
            if freshy(random)==-1
                freshy(random)=1;
            elseif freshy(random)==1
                freshy(random)=-1;
            end
            noisey(random) = 1;
         end
         
         nonlinearFeatureVector_freshData = zeros(N,6);
         nonlinearFeatureVector_freshData(:,1) = freshData(:,1);
         nonlinearFeatureVector_freshData(:,2) = freshData(:,2);
         nonlinearFeatureVector_freshData(:,3) = freshData(:,3);
         nonlinearFeatureVector_freshData(:,4) = freshData(:,2).*freshData(:,3);
         nonlinearFeatureVector_freshData(:,5) = freshData(:,2).^2;
         nonlinearFeatureVector_freshData(:,6) = freshData(:,3).^2;
         
         nonlinearFeature_freshData = transpose(nonlinearFeatureVector_freshData);
         freshData_gResult = transpose(w)*nonlinearFeature_freshData(:,i);
         
         %+ to f and - to g
         if (freshy(i) >=0) && (freshData_gResult < 0)            
             freshMisclassified = freshMisclassified + 1;

         %- to f and + to g       
         elseif (freshy(i) < 0) && (freshData_gResult >= 0)
             freshMisclassified = freshMisclassified + 1;
%              plot(freshData(i,2),freshData(i,3),'go')
%              hold on;
            
         end

     end

     totalFreshMisclassified = totalFreshMisclassified + freshMisclassified/freshPoints;

end

averageTotalFreshMisclassified = totalFreshMisclassified / numberOfExperiments;
