clc; clear; close all;

% Parameters
n = 21;  % Number of RC elements
t = 0:0.01:100;  % Time vector
dt = t(2) - t(1);

% Synthetic current data (sum of sine waves)
A = 1; 
T1 = 1;
T2 = 5;
T3 = 20;
I1 = A * sin(2 * pi * t / T1);
I2 = A * sin(2 * pi * t / T2);
I3 = A * sin(2 * pi * t / T3);
ik = I1 + I2 + I3;  % Total current

% Parameters for the true DRT (R_discrete)
mu = 10;
sigma = 5;
tau_discrete = linspace(0.01, 20, n);  % Discrete tau values

% Calculate true R_discrete using a normal distribution
R_discrete = normpdf(tau_discrete, mu, sigma);
R_discrete = R_discrete / max(R_discrete);  % Normalize to max value of 1

% Initialize voltage
V_est = zeros(1, length(t));  % Estimated voltage
R0 = 0.1;  % Internal resistance
OCV = 0;   % Open circuit voltage
V_RC = zeros(n, length(t));  % RC voltages for each element

%% Initial voltage calculation (first time step)
for i = 1:n
    V_RC(i, 1) = ik(1) * R_discrete(i) * (1 - exp(-dt / tau_discrete(i))); % ik(1) = 0 
end
V_est(1) = OCV + R0 * ik(1) + sum(V_RC(:, 1));

%% Discrete-time voltage calculation for subsequent time steps
for k = 2:length(t)
    for i = 1:n
        % Calculate RC voltages based on previous time step
        V_RC(i, k) = exp(-dt / tau_discrete(i)) * V_RC(i, k-1) + R_discrete(i) * (1 - exp(-dt / tau_discrete(i))) * ik(k);       
    end
    V_est(k) = OCV + R0 * ik(k) + sum(V_RC(:, k));
end

%% Add noise to the voltage
noise_level = 0.01;
V_noisy = V_est + noise_level * randn(size(V_est));

%% Optimization using fmincon to find the best R values with regularization
initial_R = ones(1, n);  % Initial guess for R_discrete
lb = zeros(1, n);  % Lower bound (R cannot be negative)
ub = [];  % No upper bound

% Regularization parameters
lambda = 0.1;  % Regularization parameter (adjust as needed)

% Construct the first derivative matrix L
L = zeros(n-1, n);
for i = 1:n-1
    L(i, i) = -1;
    L(i, i+1) = 1;
end

% Alternatively, using vectorized operations:
% L = diag(-ones(n-1,1), 0) + diag(ones(n-1,1), 1);
% L = L(1:n-1, :);  % Adjust size to (n-1) x n

% Optimization options
options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'interior-point', 'MaxFunctionEvaluations', 1e5);

% Define anonymous function for cost
cost_func = @(R) cost_function(R, tau_discrete, ik, V_noisy, dt, n, R0, OCV, t, lambda, L);

% Perform optimization
R_optimized = fmincon(cost_func, initial_R, [], [], [], [], lb, ub, [], options);

%% Plot the DRT comparison (True DRT vs Optimized DRT)
figure;
plot(tau_discrete, R_discrete, 'b-', 'LineWidth', 2);  % True DRT (blue solid line)
hold on;
stem(tau_discrete, R_discrete, 'bo', 'LineWidth', 1.5);  % True DRT points (blue circles)

plot(tau_discrete, R_optimized, 'r-', 'LineWidth', 2);  % Optimized DRT (red solid line)
stem(tau_discrete, R_optimized, 'ro', 'LineWidth', 1.5);  % Optimized DRT points (red circles)

xlabel('\tau (Time Constant)');
ylabel('R (Resistance)');
legend('True DRT','', 'Optimized DRT');
title('True vs Optimized Distribution of Relaxation Times (DRT)');
grid on;

%% Plot the estimated voltage (V_est) and the synthetic current (ik)
figure;
subplot(2, 1, 1);
plot(t, V_est, 'r-', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Estimated Voltage (V)');
title('Estimated Voltage (V_{est})');
grid on;

subplot(2, 1, 2);
plot(t, ik, 'b-', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Current (A)');
title('Synthetic Current (i_k)');
grid on;

%% Functions

% Residuals function for calculating V_est with given R_discrete
function V_est = calculate_voltage(R_discrete, tau_discrete, ik, dt, n, R0, OCV, t)
    V_est = zeros(1, length(t));  % Initialize estimated voltage
    V_RC = zeros(n, length(t));  % RC voltages for each element

    % Initial voltage calculation (first time step)
    for i = 1:n
        V_RC(i, 1) = ik(1) * R_discrete(i) * (1 - exp(-dt / tau_discrete(i)));
    end
    V_est(1) = OCV + R0 * ik(1) + sum(V_RC(:, 1));

    % Discrete-time voltage calculation for subsequent time steps
    for k = 2:length(t)
        for i = 1:n
            V_RC(i, k) = exp(-dt / tau_discrete(i)) * V_RC(i, k-1) + R_discrete(i) * (1 - exp(-dt / tau_discrete(i))) * ik(k);
        end
        V_est(k) = OCV + R0 * ik(k) + sum(V_RC(:, k));
    end
end

% Cost function for fmincon (sum of squared residuals with regularization)
function cost = cost_function(R_discrete, tau_discrete, ik, V_noisy, dt, n, R0, OCV, t, lambda, L)
    % Calculate the estimated voltage for the current R_discrete
    V_est = calculate_voltage(R_discrete, tau_discrete, ik, dt, n, R0, OCV, t);
    
    % Compute the sum of squared differences (residuals)
    residuals = V_noisy - V_est;
    data_fidelity = sum(residuals.^2);
    
    % Regularization term (first derivative penalty)
    regularization = lambda * norm(L * R_discrete', 2)^2;
    
    % Total cost
    cost = data_fidelity + regularization;
end
