clc; clear; close all;

%% Parameters
n = 5;  % Number of RC elements (extendable to any order)
t = 0:0.01:100;  % Time vector (continuous time)
dt = t(2) - t(1);  % Time step (delta t)

% Synthetic current data (sum of sine waves)
A = 1; 
T1 = 1;
T2 = 5;
T3 = 20;
I1 = A * sin(2 * pi * t / T1);
I2 = A * sin(2 * pi * t / T2);
I3 = A * sin(2 * pi * t / T3);
ik = I1 + I2 + I3;  % Total current

% Parameters for the normal distribution of tau
mu = 10;
sigma = 5;
tau = 0.1:0.1:20;  % Time constants
pdf_tau = (1/(sigma * sqrt(2*pi))) * exp(-(tau - mu).^2 / (2 * sigma^2));
pdf_tau = pdf_tau / max(pdf_tau);  % Normalize the PDF
tau_discrete = linspace(min(tau), max(tau), n);  % Discrete tau values for RC elements
R_discrete = interp1(tau, pdf_tau, tau_discrete);  % Interpolated R values

% Initialize voltage
V_est = zeros(1, length(t));  % Estimated voltage
R0 = 0.1;  % Internal resistance
OCV = 0;   % Open circuit voltage

%% Continuous-time voltage calculation (n-RC system)
for k = 1:length(t)
    % Calculate total voltage for each time step (continuous form)
    V_est(k) = OCV + R0 * ik(k);
    for i = 1:n
        V_est(k) = V_est(k) + R_discrete(i) * ik(k) * (1 - exp(-t(k) / tau_discrete(i)));
    end
end

%% Add noise to the voltage
noise_level = 0.01;
V_noisy = V_est + noise_level * randn(size(V_est));

%% Plot the results
figure;

% Subplot for current
subplot(2,1,1);
plot(t, ik, 'k-', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Current (A)');
title('Current');
grid on;

% Subplot for voltage
subplot(2,1,2);
plot(t, V_est, 'b-', 'LineWidth', 1.5); hold on;
plot(t, V_noisy, 'r--', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Voltage (V)');
legend('V_{est}', 'V_{noisy}');
title('Estimated Voltage and Noisy Voltage (n-RC System, Continuous)');
grid on;
