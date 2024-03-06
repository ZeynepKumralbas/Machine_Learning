%Zeynep Kumralbaþ 150115051
%Logistic Regression Q2

N = 100;
numberOfExperiments = 100;
eta = 0.01;

x = [-1,1];

totalEout = 0;
totalEpochs = 0;

point = zeros(2,2);

%generate random points between [-1 1] for line f
range1 = -1;
range2 = 1;

for iterations = 1:numberOfExperiments
    
    %************* f - starts *************%
    %create line points
    for i = 1:2   
        r = range1 + (range2-range1).*rand(2,1);
        point(i,1) = r(1,1);
        point(i,2) = r(2,1);  
    end

    %find coefficients of the line
    coefficients = polyfit([point(1,1), point(2,1)], [point(1,2), point(2,2)], 1); %[x values][y values]
    a = coefficients (1);
    b = coefficients (2);

    %draw the line
    line = a*x+b;
    
%     subplot(2,1,1);
%     plot(x,line)
%     hold on;
%     axis([-1 1 -1 1])
    %************* f - ends *************%
    
     data = zeros(N,3);
     y = zeros(N,1);
 
     %generate random data between [-1 1]
      for i = 1:N
          r = range1 + (range2-range1).*rand(2,1);
          data(i,1) = 1;
          data(i,2) = r(1,1); %x
          data(i,3) = r(2,1); %y
          %y-ax-b = 0
          if data(i,3)-a*(data(i,2))-b >= 0
            y(i) = 1;
%              plot(data(i,2),data(i,3),'r+')
%              hold on;
         elseif data(i,3)-a*(data(i,2))-b < 0
             y(i) = -1;
%              plot(data(i,2),data(i,3),'ro')
          end
 
      end

     data_vector = transpose(data); %vector version of data
     
     w_old = zeros(3,1); %w vector, initially zero
     w_new = zeros(3,1); %w vector, initially zero
     w_afterEpoc = w_old;
     
     numOfEpoch = 0;
     condition = 1;
     
     %stop criteria
     while condition >= 0.01
      
         chosen = zeros(1,N);               %to determine whether it is selected before or not
         
         %1 epoch starts%
         w_beforeEpoc = w_afterEpoc;
         numOfEpoch = numOfEpoch + 1;         
         for i=1:N
            random = randi([1 N]);          %random integer between 1 and N
            rand_selected = chosen(random);
            while rand_selected==1          %if selected before, choose within unselected ones
                random = randi([1 N]);
                rand_selected = chosen(random);
            end
            chosen(random) = 1;    
            n = random;
            
            %stochastic gradient descent, consider one point
            gradient = (-1)*( (y(n)*data_vector(:,n)) ./ (1+exp(y(n)*transpose(w_old)*data_vector(:,n))) );
            w_new = w_old - eta * gradient;
            w_old = w_new;          
         end
         %1 epoch ends%
         
         w_afterEpoc = w_new;
         condition = norm(w_afterEpoc-w_beforeEpoc); %norm
     
     end
     
    totalEpochs = totalEpochs + numOfEpoch;
    
    %*******************Eout calculation*******************%
    freshPoints = 1000;
    freshData = zeros(N,3);
    freshy = zeros(N,1);
    
    %generate random data between [-1 1]
     for i = 1:freshPoints
         r = range1 + (range2-range1).*rand(2,1);
         freshData(i,1) = 1;
         freshData(i,2) = r(1,1); %x
         freshData(i,3) = r(2,1); %y
         
         freshData_vector = transpose(freshData);
         freshData_gResult = transpose(w_new)*freshData_vector(:,i);
         
         %label fresh data points
         if freshData(i,3)-a*(freshData(i,2))-b >= 0
            freshy(i) = 1;
            
         elseif freshData(i,3)-a*(freshData(i,2))-b < 0
             freshy(i) = -1;
         end
         
         %cross entropy formula
         sum = 0;
         for j=1:N
            sum = sum + log( 1 + ( exp(-freshy(n)*transpose(w_new)*freshData_vector(:,n)) ) );
         end
         cross_entropy = 1/N * sum;
         
     end
          
     totalEout = totalEout + cross_entropy;
     

end
averageTotalEout = totalEout / numberOfExperiments;
fprintf('averageTotalEout %i', averageTotalEout);
averageTotalEpochs = totalEpochs / numberOfExperiments;
fprintf(' averageTotalEpochs %i', averageTotalEpochs);