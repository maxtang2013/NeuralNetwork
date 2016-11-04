function y = BP_SINCOS_HTAN()
% z = sin(x)cos(y)
pattern = [];
for x=(-pi):(pi/8):(pi)
    for y=(-pi):(pi/8):(pi)
        z = sin(x)*cos(y) * 0.45 + 0.5;
        pattern = [pattern' [x y z]']';
    end
end

Q = 17*17;             % Total number of the pattern to be input
eta = 0.9;             % Learning rate
alpha = 0.6;           % Momentum
tol = 0.5;            % Error tolerance
n = 2; r = 10; q = 10; p = 1;  % Architecture

Wih = 2 * rand(n+1, r) - 1; % Input-hidden weight matrix
Whh = 2 * rand(r+1, q) - 1; % First hidden layer weight matrix 
Whj = 2 * rand(q+1, p) - 1; % Hidden-output weight matrix

DeltaWih = zeros(n+1,r);
DeltaWhh = zeros(r+1,q);
DeltaWhj = zeros(q+1, p);
DeltaWihOld = zeros(n+1, r);
DeltaWhhOld = zeros(r+1, q);
DeltaWhjOld = zeros(q+1, p);

Si = [ones(Q,1) pattern(:, 1:2)]; % Input Signals
D = pattern(:,3);                 % Desired output
Sh1 = [1 zeros(1,r)];  % First hidden layer neuron signals
Sh = [1 zeros(1,q)];   % Second hidden layer neuron signals
Sy = zeros(1,p);       % Output neuron signals

deltaO = zeros(1,p);
deltaH = zeros(1,q+1);
deltaH1 = zeros(1,r+1);
sumerror = 2*tol;      

it = 0;
while (sumerror > tol) % Iterate
    sumerror = 0;
    for k = 1:Q
        Zh1 = Si(k,:) * Wih;              % First hidden layer activations
        Sh1 = [1 f(Zh1)];
        
        Zh = Sh1 * Whh;                   % Second hidden layer activations
        Sh = [1 f(Zh)];
        
        Yj = Sh * Whj;                    % Output activations 1x(q+1) * (q+1)xp = 1xp
        Sy = f(Yj);
        
        Ek = D(k) - Sy;                   % 1xp
        deltaO = Ek .* df(Yj);
        
        for h = 1:q+1
            DeltaWhj(h,:) = deltaO * Sh(h);
        end
        for h = 2:q+1
            deltaH(h) = deltaO*Whj(h,:)'*df(Zh(h-1));
        end
        for i = 1:r+1
            DeltaWhh(i,:)  = deltaH(2:q+1) * Sh1(i);
        end
        for h1 = 2:r+1
            deltaH1(h1) = deltaH(2:q+1) * Whh(h1,:)' * df(Zh1(h1-1));
        end 
        for i = 1:n+1
            DeltaWih(i,:) = deltaH1(2:r+1) * Si(k,i);
        end
        
        Wih = Wih + eta * DeltaWih + alpha * DeltaWihOld;
        Whh = Whh + eta * DeltaWhh + alpha * DeltaWhhOld;
        Whj = Whj + eta * DeltaWhj + alpha * DeltaWhjOld;
        DeltaWihOld = DeltaWih;
        DeltaWhhOld = DeltaWhh;
        DeltaWhjOld = DeltaWhj;
        sumerror = sumerror + sum(Ek.^2);
    end
    
    sumerror
    it = it + 1;
    if it > 50
        it = 0;
    else
        continue;
    end
    t = zeros(25,25);
    
    % visualize the neural network
    [u,v] = meshgrid(-pi:pi/12:pi);
    for i=1:25
        for j=1:25
            a = u(i,j);
            b = v(i, j);
            p = [1, a, b];
            
            Zh1 = p * Wih;              % First hidden layer activations
            Sh1 = [1 f(Zh1)];

            Zh = Sh1 * Whh;                   % Second hidden layer activations
            Sh = [1 f(Zh)];

            Yj = Sh * Whj;                    % Output activations 1x(q+1) * (q+1)xp = 1xp
            Sy = f(Yj);

            t(i,j) = Sy;
        end
    end
    surf(u,v,t);
end

function y = f(x) 
%y = 1.7159.*tanh(x*2/3);
y = 1./(1+exp(-x));

function y = df(x)
%lamda=2/3;
%alpha=1.7159;
%t = tanh(lamda.*x);
%y = alpha.*lamda.*(1-t.*t);
%y = exp(-x)./(1+exp(-x))./(1+exp(-x));
t = exp(-x);
y = t ./(1+t)./(1+t);