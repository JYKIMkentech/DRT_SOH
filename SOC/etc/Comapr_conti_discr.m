clc; clear; close all;

%% Parameters for synthetic data generation
n = 2;  % Number of RC elements
time_data = 0:0.1:100;  % Time from 0 to 100 seconds with a step of 0.1

% Synthetic current data (sum of sine waves)
current_data = sin(2 * pi * time_data / 20) + 0.5 * sin(2 * pi * time_data / 50);  

% True parameters for synthetic data
tau_true = [5, 10];  % Time constants for 2 RC elements (true values)
R_true = [1, 0.5];   % Resistances for 2 RC elements (true values)
R0 = 0.1;  % Internal resistance
OCV = 4.0; % Open circuit voltage

% Initialize U and V_est for the discrete and continuous methods
U = zeros(n, length(time_data));  % U for each RC element in discrete calculation
V_est_discrete = zeros(1, length(time_data));  % Discrete-time voltage
V_est_continuous = zeros(1, length(time_data));  % Continuous-time voltage
V_true = zeros(1, length(time_data));  % True voltage

% Time step for discrete calculation (assuming uniform time steps)
dt = time_data(2) - time_data(1);

%% Generate true voltage using the same parameters as continuous-time equation
for k = 1:length(time_data)
    V_true(k) = OCV + R0 * current_data(k);
    for i = 1:n
        V_true(k) = V_true(k) + R_true(i) * current_data(k) * (1 - exp(-time_data(k) / tau_true(i)));
    end
end

%% Discrete-time calculation
for k = 1:length(time_data)-1
    for i = 1:n
        U(i, k+1) = exp(-dt / tau_true(i)) * U(i, k) + R_true(i) * (1 - exp(-dt / tau_true(i))) * current_data(k);
    end
    V_est_discrete(k+1) = OCV + R0 * current_data(k) + sum(U(:, k));
end

%% Continuous-time calculation
for k = 1:length(time_data)
    V_est_continuous(k) = OCV + R0 * current_data(k);
    for i = 1:n
        V_est_continuous(k) = V_est_continuous(k) + R_true(i) * current_data(k) * (1 - exp(-time_data(k) / tau_true(i)));
    end
end

%% Plot the comparison results
figure;

% Plot true voltage
plot(time_data, V_true, 'b', 'LineWidth', 2); hold on;

% Plot discrete estimated voltage
plot(time_data, V_est_discrete, 'g--', 'LineWidth', 1.5);

% Plot continuous estimated voltage
plot(time_data, V_est_continuous, 'r:', 'LineWidth', 1.5);

% Label the plot
xlabel('Time (s)');
ylabel('Voltage (V)');
title('Comparison of True Voltage, Discrete Estimated Voltage, and Continuous Estimated Voltage');
legend('True Voltage', 'Discrete Estimated Voltage', 'Continuous Estimated Voltage');
grid on;

