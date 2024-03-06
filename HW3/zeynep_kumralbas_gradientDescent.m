%Zeynep Kumralbaþ 150115051
%Gradient Descent Questions 4-7

%functions
E_uv = @(u,v) (u.*exp(v)-2.*v.*exp(-u)).^2;  %Error
df_u = @(u,v) 2.*(exp(v)+2.*v.*exp(-u)).*(u.*exp(v)-2.*v.*exp(-u)); %partial derivative respect to u
df_v = @(u,v) 2.*(u.*exp(v)-2.*v.*exp(-u)).*(u.*exp(v)-2.*exp(-u)); %partial derivative respect to v
eta = 0.1;

u = 1;
v = 1;
df = [df_u(u,v), df_v(u,v)];        %gradient descent for u=1,v=1
old = [u,v];                        %old w
E_uv(u,v);
i = 0;
while E_uv(u,v) > 10^(-14)          %while error > 10^(-14)
    delta_w = (-1) * eta * df;      %delta_w = -1 * eta * gradient descent
    new = delta_w + old;            %delta_w = w(i) - w(i-1) , new w = delta_w + old w 
    u = new(1);
    v = new(2);
    df = [df_u(u,v), df_v(u,v)];    %calculate gradient descent with updated w
    old = new;
    i = i + 1;   
end
numOfIterations = i;
fprintf('Q5: number of iterations %i\n', numOfIterations);
fprintf('Q6: (u,v):(%d,%d)\n', new(1), new(2));

last_error = E_uv(u,v);

%Question 7
 old_u = 1;
 old_v = 1;
 df = [df_u(old_u,old_v), df_v(old_u,old_v)];
 j = 0;
 while j<15
      delta_w = (-1) * eta * df;
      new_u = delta_w(1) + old_u;                   %update u
      df = [df_u(new_u,old_v), df_v(new_u,old_v)];
      delta_w = (-1) * eta * df;
      new_v = delta_w(2) + old_v;                   %update v
      df = [df_u(new_u,new_v), df_v(new_u,new_v)];
      old_u = new_u;
      old_v = new_v;
      j = j + 1;
 end
 Q7_errorAfter15Episode = E_uv(new_u,new_v);
fprintf('Q7: error after 15 iterations %i\n', Q7_errorAfter15Episode);

