%% GP_Moosung_Lee - Update in 2021-01-16
%% Reference: Gaussian processes by Chuong B. Do (2019-07-19) - http://cs229.stanford.edu/summer2020/gaussian_processes.pdf

f_x = @(x) 10*x.*sin(x); % Function to be fitted.

n_train = 50;
x = sort((rand(n_train,1)-0.5)*10); %Training input x
y0 = f_x(x); % Noiseless output y0
y = y0 + 0.5*randn(size(x)); % Training output y 

% Test dataset
x_ = (-10:0.1:10)';
f_ = f_x(x_);

% compute posterior predictive
tau = 1;
beta = 10^3;
K = struct;
K.xx = Kfn(x,x,tau);
K.x_x_ = Kfn(x_,x_,tau);
K.xx_ = Kfn(x,x_,tau);
K.invK = inv(K.xx+eye(length(x))/beta);
postMu = K.xx_.'*K.invK*y;
postCov = K.x_x_ + (1/beta)*eye(length(x_)) - K.xx_.'*K.invK*K.xx_;


S2 = diag(postCov);
figure(1),
plot(x,y,'k*'),hold on;
plot(x_, f_x(x_),'r','linewidth',3),hold on
plot(x_, postMu ,'b','linewidth',3),hold on;
f__ = [postMu+2*sqrt(S2);flip(postMu-2*sqrt(S2),1)];
fill([x_; flip(x_,1)], f__, [4 4 7]/8, 'EdgeColor', [4 4 7]/8);
set(gcf,'color','w')

title('Plot result')
legend('Training data (*)', 'Test data (red)', 'Predicted by GPR (blue)')


function xz = Kfn(x,x_,tau,Q)
    if nargin == 3
        Q = 2;
    elseif nargin < 3
        error('Input numbers should be at least 2 vectors of the same vector dimension & tau')
    end
    x = x.'; x_ = x_.';
    [D, n] = size(x); 
    [d, m] = size(x_);
    if D ~= d
      error('Error: column lengths must agree.');
    end

    xz = repmat(sum(x.*x,1)',1,m)+repmat(sum(x_.*x_,1),n,1)-Q*x.'*x_;
    xz = exp(-xz ./ 2 ./ tau^2);
    
end
