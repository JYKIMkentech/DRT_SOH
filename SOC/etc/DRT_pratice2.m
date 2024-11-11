clc; clear; close all;

%% Parameters
n = 21;  % Number of RC elements
t = 0:0.01:100;  % Time vector
dt = t(2) - t(1);
num_scenarios = 10;  % Number of current scenarios

% Synthetic current data (sum of sine waves for multiple scenarios)
Amp = linspace(1, 10, num_scenarios);  % Amplitude values from 1 to 10 (arithmetic progression)
T = [1, 2, 5, 10, 20, 25, 30, 35, 40, 50];  % Different period values for each signal
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

%% Initialize arrays for storing results
R_optimized_fmincon_all = zeros(num_scenarios, n);  % fmincon DRT estimates
R_optimized_quad_all = zeros(num_scenarios, n);      % quadprog DRT estimates
R_analytical_all = zeros(num_scenarios, n);         % Analytical DRT estimates

% Regularization parameter
lambda = 0.1;  % 정규화 파라미터 설정

% Construct the first derivative matrix L for regularization
L = zeros(n-1, n);
for i = 1:n-1
    L(i, i) = -1;
    L(i, i+1) = 1;
end

%% Loop over each scenario
figure(1);  % Current and Voltage Figure
for s = 1:num_scenarios
    fprintf('Processing Scenario %d/%d...\n', s, num_scenarios);
    
    % Current for the scenario
    ik = ik_scenarios(s, :);
    
    %% Initialize voltage
    V_est = zeros(1, length(t));  % Estimated voltage
    R0 = 0.1;  % Internal resistance
    OCV = 0;   % Open circuit voltage
    V_RC = zeros(n, length(t));  % RC voltages for each element
    
    %% Initial voltage calculation (first time step)
    for i = 1:n
        V_RC(i, 1) = ik(1) * R_discrete_true(i) * (1 - exp(-dt / tau_discrete(i)));
    end
    V_est(1) = OCV + R0 * ik(1) + sum(V_RC(:, 1));
    
    %% Discrete-time voltage calculation for subsequent time steps
    for k_idx = 2:length(t)
        for i = 1:n
            % Calculate RC voltages based on previous time step
            V_RC(i, k_idx) = exp(-dt / tau_discrete(i)) * V_RC(i, k_idx-1) + ...
                             R_discrete_true(i) * (1 - exp(-dt / tau_discrete(i))) * ik(k_idx);       
        end
        V_est(k_idx) = OCV + R0 * ik(k_idx) + sum(V_RC(:, k_idx));
    end
    
    %% Add noise to the voltage
    rng(0);

    noise_level = 0.01;
    V_noisy = V_est + noise_level * randn(size(V_est));
    
    %% Construct the model matrix W for the current scenario
    W = zeros(length(t), n);  % Initialize W matrix
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
    
    %% Optimization using fmincon to find the best R values with regularization
    initial_R = ones(n, 1);  % Initial guess for R_discrete (column vector)
    lb = zeros(n,1);  % Lower bound (R cannot be negative)
    ub = [];  % No upper bound
    
    % Optimization options for fmincon
    options_fmincon = optimoptions('fmincon', 'Display', 'none', 'Algorithm', 'interior-point', 'MaxFunctionEvaluations', 1e5);
    
    % Define the cost function handle
    cost_func = @(R) cost_function(R, tau_discrete, ik, V_noisy, dt, n, R0, OCV, t, lambda, L);
    
    % Perform optimization with fmincon
    [R_optimized_fmincon, ~] = fmincon(cost_func, initial_R, [], [], [], [], lb, ub, [], options_fmincon);
    
    %% Quadratic Form Optimization using quadprog
    % Quadratic form solution
    H = (W' * W + lambda * (L' * L)); % Regularized Hessian matrix
    f = -W' * V_noisy';  % Linear term for optimization
    
    % Quadratic optimization options
    options_quadprog = optimoptions('quadprog','Display','none','Algorithm','interior-point-convex');
    
    % Perform quadratic optimization
    R_optimized_quad = quadprog(H, f, [], [], [], [], lb, ub, [], options_quadprog);
    
    %% Analytical solution using matrix method with regularization
    R_analytical = (W' * W + lambda * (L' * L)) \ (W' * V_noisy');
    R_analytical(R_analytical < 0) = 0;  % Enforce non-negativity
    
    %% Store the results
    R_optimized_fmincon_all(s, :) = R_optimized_fmincon';
    R_optimized_quad_all(s, :) = R_optimized_quad';
    R_analytical_all(s, :) = R_analytical';
    
    %% Plot Current and Voltage for the scenario as a subplot
    subplot(5, 2, s);
    yyaxis left
    plot(t, ik, 'b-', 'LineWidth', 1.5);
    ylabel('Current (A)');
    ylim([min(ik)-1, max(ik)+1]);
    
    yyaxis right
    plot(t, V_noisy, 'r-', 'LineWidth', 1.5);
    ylabel('Voltage (V)');
    ylim([min(V_noisy)-0.1, max(V_noisy)+0.1]);
    
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
    
    % Plot fmincon Optimized DRT
    plot(tau_discrete, R_optimized_fmincon_all(s, :), '-', 'Color', 'r', 'LineWidth', 1.5, 'DisplayName', 'fmincon DRT');
    
    % Plot quadprog Optimized DRT
    plot(tau_discrete, R_optimized_quad_all(s, :), '--', 'Color', 'm', 'LineWidth', 1.5, 'DisplayName', 'quadprog DRT');
    
    % Plot Analytical DRT
    plot(tau_discrete, R_analytical_all(s, :), ':', 'Color', 'g', 'LineWidth', 1.5, 'DisplayName', 'Analytical DRT');
    
    hold off;
    xlabel('\tau (Time Constant)');
    ylabel('R (Resistance)');
    title(['DRT Comparison for Scenario ', num2str(s), ' (\lambda = ', num2str(lambda), ')']);
    legend('True DRT', 'fmincon DRT', 'quadprog DRT', 'Analytical DRT', 'Location', 'BestOutside');
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
    data_fidelity = sum(residuals.^2);
    
    % Regularization term (first derivative penalty)
    regularization = lambda * norm(L * R_discrete, 2)^2;  % 수정: R_discrete' 제거
    
    % Total cost
    cost = data_fidelity + regularization;
end


