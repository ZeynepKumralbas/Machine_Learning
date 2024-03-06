%Linear Regression Algorithm%
%Q8 Q9
%Zeynep Kumralbaþ 150115051%

N = 100;
numberOfExperiments = 1000;

x = [-1,1];

point = zeros(2,2);

%generate random points between [-1 1] for line f
range1 = -1;
range2 = 1;

totalNumberOfIterations = 0;

totalEin = 0;
averageEin = 0;
averageTotalFreshMisclassified = 0;
totalFreshMisclassified = 0;

for iterations = 1:numberOfExperiments
    
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

    data = zeros(N,3);
    y = zeros(N,1);

    %generate random data between [-1 1] for line f
     for i = 1:N
         r = range1 + (range2-range1).*rand(2,1);
         data(i,1) = 1;
         data(i,2) = r(1,1); %x
         data(i,3) = r(2,1); %y
         %y-ax-b = 0
         if data(i,3)-a*(data(i,2))-b >=0
            y(i) = 1;
%              plot(data(i,2),data(i,3),'r+')
%              hold on;
         elseif data(i,3)-a*(data(i,2))-b <0
             y(i) = -1;
%              plot(data(i,2),data(i,3),'ro')
         end

     end

    w = zeros(3,1); %w vector, initially zero

    data_vector = transpose(data); %vector version of data
    
    X = transpose(data_vector);
    transpose_X = transpose(X);
    xDagger = (transpose_X*X)\transpose_X; %=inv(transpose_X*X)*transpose_X
    w = xDagger*y;
    
    g_a = -w(2,1)./w(3,1);
    g_b = -w(1,1)./w(3,1);
    y1 = g_a.*x + g_b;
%     plot(x,y1,'r');
%     subplot(2,1,2);
%     plot(x,y1,'r')
    
    Ein = 0;
    for data_i=1:N
        sign_result = transpose(w)*data_vector(:,data_i);
        if((sign_result>=0) && (y(data_i)<0))
            Ein = Ein + 1;
        elseif((sign_result<0) && (y(data_i)>=0))
            Ein = Ein + 1;
        end           
    end    
    totalEin = totalEin + (Ein/100);
    
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
         freshData_gResult = transpose(w)*freshData_vector(:,i);
         
         %+ to f and - to g
         if (freshData(i,3)-a*(freshData(i,2))-b >=0) && (freshData_gResult < 0)            
             freshMisclassified = freshMisclassified + 1;
%              plot(freshData(i,2),freshData(i,3),'g+');
%              hold on;

         %- to f and + to g       
         elseif (freshData(i,3)-a*(freshData(i,2))-b < 0) && (freshData_gResult >= 0)
             freshMisclassified = freshMisclassified + 1;
%              plot(freshData(i,2),freshData(i,3),'go')
%              hold on;
            
         end

     end
     
     totalFreshMisclassified = totalFreshMisclassified + freshMisclassified/freshPoints;
 
end

averageEin = totalEin / numberOfExperiments;
averageTotalFreshMisclassified = totalFreshMisclassified / numberOfExperiments;