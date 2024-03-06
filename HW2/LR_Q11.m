%Linear Regression Algorithm%
%Q11
%Zeynep Kumralbaþ 150115051%

N = 1000;
numberOfExperiments = 1000;

x = [-1,1];

totalNumberOfIterations = 0;

all_w = zeros(3,N);
totalEin = 0;
total =0;
averageEin = 0;
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

    w = zeros(3,1); %w vector, initially zero

    data_vector = transpose(data); %vector version of data
    
    X = transpose(data_vector);
    transpose_X = transpose(X);
    xDagger = (transpose_X*X)\transpose_X; %=inv(transpose_X*X)*transpose_X
    w = xDagger*y;
    all_w(:,iterations) = w;
    
    g_a = -w(2,1)./w(3,1);
    g_b = -w(1,1)./w(3,1);
    y1 = g_a.*x + g_b;
%       plot(x,y1,'g');
%       subplot(2,1,2);
    
    Ein = 0;
    for data_i=1:N
        sign_result = transpose(w)*data_vector(:,data_i);
        if((sign_result>=0) && (y(data_i)<0))
            Ein = Ein + 1;
        elseif((sign_result<0) && (y(data_i)>=0))
            Ein = Ein + 1;
        end           
    end    
    
    totalEin = totalEin + (Ein/N);

end

averageEin = totalEin / numberOfExperiments;
avgtot = total/ numberOfExperiments;