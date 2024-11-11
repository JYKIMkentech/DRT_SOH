clc; clear; close all;

% Parameters
n = 21;  % Number of RC elements
t = 0:0.01:100;  % Time vector
dt = t(2) - t(1);
R0 = 0.1;  % Internal resistance
OCV = 0;   % Open circuit voltage
tau_discrete = linspace(0.01, 20, n);  % Discrete tau values
mu = 10;
sigma = 5;

% Regularization parameters
lambda_values = [0, 1e-3, 1e-2, 1e-1, 1];  % Lambda values
num_scenarios = 10;  % Number of current scenarios
k_fold = 5;  % Number of folds for cross-validation

% Preallocate arrays
R_true_all = zeros(num_scenarios, n);
V_noisy_all = zeros(num_scenarios, length(t));
ik_all = zeros(num_scenarios, length(t));

% Generate 10 different current scenarios
for s = 1:num_scenarios
    % Synthetic current data (sum of sine waves with random parameters)
    A1 = rand * 2; % Random value between 0 and 2
    A2 = rand * 2;
    A3 = rand * 2;
    T1 = rand * 10 + 1; % Random value between 1 and 11
    T2 = rand * 10 + 1;
    T3 = rand * 10 + 1;
    I1 = A1 * sin(2 * pi * t / T1);  
    I2 = A2 * sin(2 * pi * t / T2);
    I3 = A3 * sin(2 * pi * t / T3);
    ik = I1 + I2 + I3;  % Total current
    ik_all(s, :) = ik;
    
    % Calculate true R_discrete using a normal distribution
    R_discrete = normpdf(tau_discrete, mu, sigma);
    R_discrete = R_discrete / max(R_discrete);  % Normalize to max value of 1
    R_true_all(s, :) = R_discrete;
    
    % Calculate V_est and add noise
    V_est = calculate_voltage(R_discrete, tau_discrete, ik, dt, n, R0, OCV, t);
    noise_level = 0.01;
    V_noisy = V_est + noise_level * randn(size(V_est));
    V_noisy_all(s, :) = V_noisy;
end

% Generate k-fold indices
rng(0);  % Set seed for reproducibility
scenario_indices = randperm(num_scenarios);
fold_sizes = floor(num_scenarios / k_fold) * ones(1, k_fold);
remaining = num_scenarios - sum(fold_sizes);
fold_sizes(1:remaining) = fold_sizes(1:remaining) + 1;

indices = zeros(num_scenarios, 1);
start_idx = 1;
for k = 1:k_fold
    end_idx = start_idx + fold_sizes(k) - 1;
    indices(scenario_indices(start_idx:end_idx)) = k;
    start_idx = end_idx + 1;
end

% Optimization options
options_qp = optimoptions('quadprog', 'Display', 'off');
options_fmincon = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp');

% Construct the first derivative matrix L
L = zeros(n-1, n);
for i = 1:n-1
    L(i, i) = -1;
    L(i, i+1) = 1;
end

methods = {'quadprog', 'fmincon', 'Analytical'};
num_methods = length(methods);
total_computation_time = zeros(num_methods, 1);  % To store computation time per method

% Initialize arrays to store cross-validation errors
errors = zeros(length(lambda_values), k_fold, num_methods);

% Cross-validation to find the optimal lambda for each method
for m = 1:num_methods
    method = methods{m};
    computation_time = 0;
    for idx = 1:length(lambda_values)
        lambda = lambda_values(idx);
        for k = 1:k_fold
            % Training and validation indices
            test_idx = (indices == k);
            train_idx = ~test_idx;
            
            % Prepare training data
            W_train = [];
            y_train = [];
            for s = find(train_idx)'
                ik = ik_all(s, :);
                V_noisy = V_noisy_all(s, :)';
                y = V_noisy - OCV - R0 * ik';
                W = construct_W(ik, tau_discrete, dt, n, t);
                W_train = [W_train; W];
                y_train = [y_train; y];
            end
            
            % Solve using the selected method
            tic;
            switch method
                case 'quadprog'
                    H = 2 * (W_train' * W_train + lambda * L' * L);
                    f = -2 * (W_train' * y_train);
                    lb = zeros(n, 1);
                    R_est = quadprog(H, f, [], [], [], [], lb, [], [], options_qp);
                case 'fmincon'
                    cost_func = @(R) sum((W_train * R' - y_train).^2) + lambda * norm(L * R', 2)^2;
                    initial_R = ones(1, n);
                    lb = zeros(1, n);
                    R_est = fmincon(cost_func, initial_R, [], [], [], [], lb, [], [], options_fmincon);
                    R_est = R_est';
                case 'Analytical'
                    A = W_train' * W_train + lambda * L' * L;
                    b = W_train' * y_train;
                    R_est = A \ b;
                    R_est(R_est < 0) = 0;
                otherwise
                    error('Unknown method');
            end
            computation_time = computation_time + toc;
            
            % Calculate validation error
            validation_error = 0;
            for s = find(test_idx)'
                ik = ik_all(s, :);
                V_noisy = V_noisy_all(s, :)';
                y = V_noisy - OCV - R0 * ik';
                W = construct_W(ik, tau_discrete, dt, n, t);
                y_pred = W * R_est;
                residuals = y - y_pred;
                validation_error = validation_error + sum(residuals.^2);
            end
            errors(idx, k, m) = validation_error;
        end
    end
    total_computation_time(m) = computation_time;
end

% Calculate mean validation error for each lambda and method
mean_errors = squeeze(mean(errors, 2));

% Find optimal lambda for each method
optimal_lambda_indices = zeros(num_methods, 1);
for m = 1:num_methods
    [~, idx] = min(mean_errors(:, m));
    optimal_lambda_indices(m) = idx;
end
optimal_lambdas = lambda_values(optimal_lambda_indices);

% Display total computation time per method
for m = 1:num_methods
    disp(['Total computation time for ', methods{m}, ': ', num2str(total_computation_time(m)), ' seconds']);
    disp(['Optimal lambda for ', methods{m}, ': ', num2str(optimal_lambdas(m))]);
end

% Plotting results for each method and lambda
figure;
for m = 1:num_methods
    method = methods{m};
    R_estimates = zeros(length(lambda_values), n);
    R_std = zeros(length(lambda_values), n);
    for idx = 1:length(lambda_values)
        lambda = lambda_values(idx);
        R_est_all = zeros(k_fold, n);  % Store R estimates for each fold
        
        % Cross-validation to get R estimates for plotting
        for k = 1:k_fold
            % Training indices
            train_idx = (indices ~= k);
            
            % Prepare training data
            W_train = [];
            y_train = [];
            for s = find(train_idx)'
                ik = ik_all(s, :);
                V_noisy = V_noisy_all(s, :)';
                y = V_noisy - OCV - R0 * ik';
                W = construct_W(ik, tau_discrete, dt, n, t);
                W_train = [W_train; W];
                y_train = [y_train; y];
            end
            
            % Solve using the selected method
            switch method
                case 'quadprog'
                    H = 2 * (W_train' * W_train + lambda * L' * L);
                    f = -2 * (W_train' * y_train);
                    lb = zeros(n, 1);
                    R_est = quadprog(H, f, [], [], [], [], lb, [], [], options_qp);
                case 'fmincon'
                    cost_func = @(R) sum((W_train * R' - y_train).^2) + lambda * norm(L * R', 2)^2;
                    initial_R = ones(1, n);
                    lb = zeros(1, n);
                    R_est = fmincon(cost_func, initial_R, [], [], [], [], lb, [], [], options_fmincon);
                    R_est = R_est';
                case 'Analytical'
                    A = W_train' * W_train + lambda * L' * L;
                    b = W_train' * y_train;
                    R_est = A \ b;
                    R_est(R_est < 0) = 0;
                otherwise
                    error('Unknown method');
            end
            R_est_all(k, :) = R_est';
        end
        
        % Calculate mean and standard deviation over folds
        R_mean = mean(R_est_all, 1);
        R_std(idx, :) = std(R_est_all, 0, 1);
        R_estimates(idx, :) = R_mean;
        
        % Plotting
        subplot(num_methods, length(lambda_values), (m-1)*length(lambda_values) + idx);
        errorbar(tau_discrete, R_mean, R_std(idx, :), 'b-', 'LineWidth', 1.5); hold on;
        plot(tau_discrete, mean(R_true_all, 1), 'k--', 'LineWidth', 2);
        xlabel('\tau (Time Constant)');
        ylabel('R (Resistance)');
        title([method, ', \lambda = ', num2str(lambda)]);
        legend('Estimated R with Error Bars', 'True R');
        grid on;
        hold off;
    end
end

% Functions

% Function to calculate V_est given R_discrete
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
            V_RC(i, k) = exp(-dt / tau_discrete(i)) * V_RC(i, k-1) + ...
                R_discrete(i) * (1 - exp(-dt / tau_discrete(i))) * ik(k);
        end
        V_est(k) = OCV + R0 * ik(k) + sum(V_RC(:, k));
    end
end

% Function to construct W matrix
function W = construct_W(ik, tau_discrete, dt, n, t)
    W = zeros(length(t), n);
    for i = 1:n
        for k = 1:length(t)
            if k == 1
                W(k, i) = ik(k) * (1 - exp(-dt / tau_discrete(i)));
            else
                W(k, i) = exp(-dt / tau_discrete(i)) * W(k-1, i) + ...
                    ik(k) * (1 - exp(-dt / tau_discrete(i)));
            end
        end
    end
end

