clc; clear; close all;

%% Parameters
n = 21;  % Number of RC elements
t = 0:0.01:100;  % Time vector
dt = t(2) - t(1);
num_scenarios = 10;  % Total number of current scenarios

% Synthetic current data (sum of sine waves for multiple scenarios)
Amp = linspace(1, 10, num_scenarios);  % Amplitude values from 1 to 10
T = [1, 2, 5, 10, 20, 25, 30, 35, 40, 50];  % Different period values for each scenario
ik_scenarios = zeros(num_scenarios, length(t));  % Initialize current for each scenario

% Generate the 10 sine waves
for k = 1:num_scenarios
    ik_scenarios(k, :) = Amp(k) * sin(2 * pi * t / T(k));
end

% Parameters for the true DRT (R_discrete)
mu = 10;
sigma = 5;
tau_discrete = linspace(0.01, 20, n);  % Discrete tau values

% Calculate true R_discrete using a normal distribution
R_discrete_true = normpdf(tau_discrete, mu, sigma);
R_discrete_true = R_discrete_true / max(R_discrete_true);  % Normalize to max value of 1

%% Cross-Validation Setup
% Define cross-validation splits
validation_splits = {[1,2], [3,4], [5,6], [7,8]};  % 4개의 검증 분할
num_folds = length(validation_splits);

% Define lambda candidates (log-spaced)
lambda_candidates = logspace(-4, 0, 50);  % From 1e-4 to 1 with 50 points
num_lambdas = length(lambda_candidates);

% Initialize CVE array
CVE = zeros(num_lambdas,1);  % Cross-Validation Error for each lambda

% Construct the first derivative matrix L for regularization
L = zeros(n-1, n);
for i = 1:n-1
    L(i, i) = -1;
    L(i, i+1) = 1;
end

%% Cross-Validation Loop
for l = 1:num_lambdas
    lambda = lambda_candidates(l);
    total_error = 0;  % Initialize total CVE for current lambda
    
    for fold = 1:num_folds
        % Define validation scenarios for current fold
        val_scenarios = validation_splits{fold};
        
        % Define training scenarios (1-8 excluding current validation pair)
        training_scenarios = setdiff(1:8, val_scenarios);
        
        % Initialize training data
        W_train = [];
        V_train = [];
        
        for s = training_scenarios
            ik_train = ik_scenarios(s, :);
            
            % Construct W matrix for the training scenario
            W = zeros(length(t), n);
            for k_idx = 1:length(t)
                for i = 1:n
                    if k_idx == 1
                        W(k_idx, i) = ik_train(k_idx) * (1 - exp(-dt / tau_discrete(i)));
                    else
                        W(k_idx, i) = exp(-dt / tau_discrete(i)) * W(k_idx-1, i) + ...
                                      ik_train(k_idx) * (1 - exp(-dt / tau_discrete(i)));
                    end
                end
            end
            
            % Calculate voltage for training scenario
            V_est = zeros(1, length(t));
            V_RC = zeros(n, length(t));
            for i = 1:n
                V_RC(i,1) = ik_train(1) * R_discrete_true(i) * (1 - exp(-dt / tau_discrete(i)));
            end
            V_est(1) = 0 + 0.1 * ik_train(1) + sum(V_RC(:,1));
            for k_idx = 2:length(t)
                for i = 1:n
                    V_RC(i,k_idx) = exp(-dt / tau_discrete(i)) * V_RC(i,k_idx-1) + ...
                                    R_discrete_true(i) * (1 - exp(-dt / tau_discrete(i))) * ik_train(k_idx);
                end
                V_est(k_idx) = 0 + 0.1 * ik_train(k_idx) + sum(V_RC(:,k_idx));
            end
            
            % Add noise to training voltage
            noise_level = 0.01;
            V_noisy_train = V_est + noise_level * randn(size(V_est));
            
            % Append to training data
            W_train = [W_train; W];
            V_train = [V_train; V_noisy_train'];
        end
        
        % Analytical solution to estimate R_discrete with current lambda
        R_discrete_est = (W_train' * W_train + lambda * (L' * L)) \ (W_train' * V_train);
        R_discrete_est(R_discrete_est < 0) = 0;  % Enforce non-negativity
        
        % Validate on current validation scenarios
        for v = 1:length(val_scenarios)
            s_val = val_scenarios(v);
            ik_val = ik_scenarios(s_val, :);
            
            % Construct W matrix for validation scenario
            W_val = zeros(length(t), n);
            for k_idx = 1:length(t)
                for i = 1:n
                    if k_idx == 1
                        W_val(k_idx, i) = ik_val(k_idx) * (1 - exp(-dt / tau_discrete(i)));
                    else
                        W_val(k_idx, i) = exp(-dt / tau_discrete(i)) * W_val(k_idx-1, i) + ...
                                          ik_val(k_idx) * (1 - exp(-dt / tau_discrete(i)));
                    end
                end
            end
            
            % Calculate voltage for validation scenario
            V_est_val = zeros(1, length(t));
            V_RC_val = zeros(n, length(t));
            for i = 1:n
                V_RC_val(i,1) = ik_val(1) * R_discrete_true(i) * (1 - exp(-dt / tau_discrete(i)));
            end
            V_est_val(1) = 0 + 0.1 * ik_val(1) + sum(V_RC_val(:,1));
            for k_idx = 2:length(t)
                for i = 1:n
                    V_RC_val(i,k_idx) = exp(-dt / tau_discrete(i)) * V_RC_val(i,k_idx-1) + ...
                                        R_discrete_true(i) * (1 - exp(-dt / tau_discrete(i))) * ik_val(k_idx);
                end
                V_est_val(k_idx) = 0 + 0.1 * ik_val(k_idx) + sum(V_RC_val(:,k_idx));
            end
            
            % Add noise to validation voltage
            V_noisy_val = V_est_val + noise_level * randn(size(V_est_val));
            
            % Predict voltage using estimated R_discrete
            V_pred_val = W_val * R_discrete_est + 0.1 * ik_val' + 0;  % ik_val' to match dimensions
            
            % Compute squared error
            squared_error = sum((V_noisy_val' - V_pred_val).^2, 'all');  % Ensure scalar
            
            % Accumulate CVE
            total_error = total_error + squared_error;
        end
    end
    
    % Store CVE for current lambda
    CVE(l) = total_error;
end

%% Plot CVE vs Lambda
figure;
plot(lambda_candidates, CVE, 'b-o', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('Lambda (\lambda)');
ylabel('Cross-Validation Error (CVE)');
title('CVE vs Lambda for Analytical DRT Optimization');
set(gca, 'XScale', 'log');  % Use logarithmic scale for lambda
grid on;

% Find the optimal lambda (minimum CVE)
[~, idx_min] = min(CVE);
optimal_lambda = lambda_candidates(idx_min);
hold on;
plot(optimal_lambda, CVE(idx_min), 'rp', 'MarkerSize', 12, 'MarkerFaceColor', 'r');
legend('CVE', ['Optimal \lambda = ', num2str(optimal_lambda)]);
hold off;

%% After Optimizing Lambda, Fit R_discrete on All Training Data (Scenarios 1-6)
% 학습 시나리오: 1-6
% 검증 시나리오: 7-8
% 테스트 시나리오: 9-10 (현재 코드에서는 사용하지 않음)

% Initialize training data
training_scenarios_final = 1:6;
W_train_final = [];
V_train_final = [];

for s = training_scenarios_final
    ik_train = ik_scenarios(s, :);
    
    % Construct W matrix for the training scenario
    W = zeros(length(t), n);
    for k_idx = 1:length(t)
        for i = 1:n
            if k_idx == 1
                W(k_idx, i) = ik_train(k_idx) * (1 - exp(-dt / tau_discrete(i)));
            else
                W(k_idx, i) = exp(-dt / tau_discrete(i)) * W(k_idx-1, i) + ...
                              ik_train(k_idx) * (1 - exp(-dt / tau_discrete(i)));
            end
        end
    end
    
    % Calculate voltage for training scenario
    V_est = zeros(1, length(t));
    V_RC = zeros(n, length(t));
    for i = 1:n
        V_RC(i,1) = ik_train(1) * R_discrete_true(i) * (1 - exp(-dt / tau_discrete(i)));
    end
    V_est(1) = 0 + 0.1 * ik_train(1) + sum(V_RC(:,1));
    for k_idx = 2:length(t)
        for i = 1:n
            V_RC(i,k_idx) = exp(-dt / tau_discrete(i)) * V_RC(i,k_idx-1) + ...
                            R_discrete_true(i) * (1 - exp(-dt / tau_discrete(i))) * ik_train(k_idx);
        end
        V_est(k_idx) = 0 + 0.1 * ik_train(k_idx) + sum(V_RC(:,k_idx));
    end
    
    % Add noise to training voltage
    noise_level = 0.01;
    V_noisy_train = V_est + noise_level * randn(size(V_est));
    
    % Append to training data
    W_train_final = [W_train_final; W];
    V_train_final = [V_train_final; V_noisy_train'];
end

% Fit R_discrete using the optimal lambda on all training data
Rtrain = (W_train_final' * W_train_final + optimal_lambda * (L' * L)) \ (W_train_final' * V_train_final);
Rtrain(Rtrain < 0) = 0;  % Enforce non-negativity

%% Plot Current and Voltage for Each Scenario (10 Subplots)
figure(1);  % Current and Voltage Figure
for s = 1:num_scenarios
    subplot(5, 2, s);
    yyaxis left
    plot(t, ik_scenarios(s, :), 'b-', 'LineWidth', 1.5);
    ylabel('Current (A)');
    ylim([min(ik_scenarios(s, :))-1, max(ik_scenarios(s, :))+1]);
    
    yyaxis right
    % Calculate voltage with Rtrain
    ik = ik_scenarios(s, :);
    W = zeros(length(t), n);
    for k_idx = 1:length(t)
        for i = 1:n
            if k_idx == 1
                W(k_idx, i) = ik(k_idx) * (1 - exp(-dt / tau_discrete(i)));
            else
                W(k_idx, i) = exp(-dt / tau_discrete(i)) * W(k_idx-1, i) + ...
                              ik(k_idx) * (1 - exp(-dt / tau_discrete(i)));
            end
        end
    end
    V_pred = W * Rtrain + 0.1 * ik' + 0;  % Transpose ik for correct dimensions
    
    plot(t, V_pred, 'r-', 'LineWidth', 1.5);
    ylabel('Voltage (V)');
    ylim([min(V_pred)-0.1, max(V_pred)+0.1]);
    
    title(['Scenario ', num2str(s), ': A=', num2str(Amp(s)), ', T=', num2str(T(s))]);
    xlabel('Time (s)');
    grid on;
end

%% Enhance the Current and Voltage Figure
figure(1);
sgtitle('Current and Voltage for Each Scenario');
% Optionally, save the figure
% saveas(gcf, 'Current_Voltage_Scenarios.png');

%% Plot the DRT comparison for each scenario in separate figures
for s = 1:num_scenarios
    figure(2 + s);  % DRT Comparison Figure for each scenario
    hold on;
    
    % Plot True DRT
    plot(tau_discrete, R_discrete_true, 'k-', 'LineWidth', 1.5, 'DisplayName', 'True DRT');
    
    % Plot Analytical DRT
    plot(tau_discrete, Rtrain, 'g--', 'LineWidth', 1.5, 'DisplayName', 'Analytical DRT');
    
    hold off;
    xlabel('\tau (Time Constant)');
    ylabel('R (Resistance)');
    title(['DRT Comparison for Scenario ', num2str(s), ' (\lambda = ', num2str(optimal_lambda), ')']);
    legend('True DRT', 'Analytical DRT', 'Location', 'BestOutside');
    grid on;
    
    % Optionally, save each DRT figure
    % saveas(gcf, ['DRT_Comparison_Scenario_', num2str(s), '.png']);
end

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
    for k_idx = 2:length(t)
        for i = 1:n
            V_RC(i, k_idx) = exp(-dt / tau_discrete(i)) * V_RC(i, k_idx-1) + ...
                             R_discrete(i) * (1 - exp(-dt / tau_discrete(i))) * ik(k_idx);
        end
        V_est(k_idx) = OCV + R0 * ik(k_idx) + sum(V_RC(:, k_idx));
    end
end

% Cost function for fmincon (sum of squared residuals with regularization)
function cost = cost_function(R_discrete, tau_discrete, ik, V_noisy, dt, n, R0, OCV, t, lambda, L)
    % Ensure R_discrete is a column vector
    R_discrete = R_discrete(:);
    
    % Calculate the estimated voltage for the current R_discrete
    V_est = calculate_voltage(R_discrete, tau_discrete, ik, dt, n, R0, OCV, t);
    
    % Compute the sum of squared differences (residuals)
    residuals = V_noisy - V_est;
    data_fidelity = sum(residuals.^2, 'all');  % Ensure scalar
    
    % Regularization term (first derivative penalty)
    regularization = lambda * norm(L * R_discrete, 2)^2;  % Removed R_discrete' as per correction
    
    % Total cost
    cost = data_fidelity + regularization;
end

