%Linear Regression-Perceptron Algorithm%
%Q10
%Zeynep Kumralbaþ 150115051%

N = 10;
numberOfExperiments = 1000;

x = [-1,1];

point = zeros(2,2);

%generate random points between [-1 1] for line f
range1 = -1;
range2 = 1;

totalNumberOfIterations = 0;


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
%      subplot(2,1,1);
%      plot(x,line)
%      hold on;
%      axis([-1 1 -1 1])

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
%               plot(data(i,2),data(i,3),'r+')
%               hold on;
         elseif data(i,3)-a*(data(i,2))-b <0
             y(i) = -1;
%               plot(data(i,2),data(i,3),'ro')
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
%      plot(x,y1,'r');
%      subplot(2,1,2);
%      plot(x,y1,'r')
     
    misclassified = zeros(3,N);
    misclassified_y = zeros(N,1);
    data_vector = transpose(data); %vector version of data

    numberOfIterations = 0;
    converge = 0;
    
    %until it converges
    while converge ~=1

        if (w == 0) %if w == 0, all data are misclassified
            misclassified = data_vector;
            misclassified_y = y;
            n = N;
            random = randi([1 n]);
            if(y(random) == 1)
                w = w + misclassified(:,random);
            else
                w = w - misclassified(:,random);
            end

        else
            n = 0;
            for i=1:N
                scalar_product = dot(w,data_vector(:,i));
                %if data is labeled wrong, put it to the misclassified set
                if ((scalar_product < 0 && y(i) == 1) || (scalar_product >= 0 && y(i) == -1)) 
                    n = n+1;
                    misclassified(1,n) = data_vector(1,i);  
                    misclassified(2,n) = data_vector(2,i);   
                    misclassified(3,n) = data_vector(3,i);   
                    misclassified_y(n) = y(i); %corresponding label
                    converge = 0;
                end       
            end

            if n>0 %if there are misclassified data

                random = randi([1 n]); %pick a random integer value between [1, n]
                scalar_product = dot(w,misclassified(:,random));

                if (scalar_product < 0 && misclassified_y(random) == 1)
                  w = w + misclassified(:,random);

                elseif (scalar_product >= 0 && misclassified_y(random) == -1)
                    w = w - misclassified(:,random);

                end

                misclassified = zeros(3,N);
            end
        end
        

        if (n == 0)
           converge = 1;
        end

        numberOfIterations = numberOfIterations + 1;

    end

    totalNumberOfIterations = totalNumberOfIterations + numberOfIterations;
    
    y1 = (-w(2,1)./w(3,1)).*x - (w(1,1)./w(3,1));
%     plot(x,y1,'r');
%     subplot(2,1,2);
%     plot(x,y1,'r')

end    

averageIterations = totalNumberOfIterations/numberOfExperiments;
fprintf('%i averageIterations', averageIterations);