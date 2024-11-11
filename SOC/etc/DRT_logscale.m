% 클리어 및 초기 설정
clear; close all; clc;

%% 1. Parameters
n = 21;                      % Number of RC elements (increased for smoother DRT)
t = 0:0.01:100;              % Time vector (seconds)
dt = t(2) - t(1);            % Time step size
num_scenarios = 10;          % Number of current scenarios
lambda = 0.0409;             % Regularization parameter

%% 2. Define Amplitudes and Periods for Current Synthesis
A = [1, 1, 1;          % Scenario 1   
     1.7, 0.6, 0.7;    % Scenario 2
     0.2, 0.5, 2.3;    % Scenario 3
     1.3, 1.1, 0.6;    % Scenario 4
     1.7, 1.8, 0.5;    % Scenario 5
     1.27, 1.33, 0.4;  % Scenario 6
     1.2, 1.6, 0.2;    % Scenario 7
     0.9, 0.7, 2.4;    % Scenario 8
     1.1, 1.1, 0.8;    % Scenario 9
     0.1, 0.1, 2.8];   % Scenario 10

T = [1, 5, 20;         % Scenario 1
     2, 4, 20;         % Scenario 2
     1, 20, 25;        % Scenario 3
     1.5, 5.3, 19.8;   % Scenario 4
     2.5, 4.2, 20.5;   % Scenario 5
     1.5, 20.9, 24.2;  % Scenario 6
     1.3, 6, 19.3;     % Scenario 7
     2.2, 4.8, 20.2;   % Scenario 8
     2, 20.8, 26.1;    % Scenario 9
     1.1, 4.3, 20.1];  % Scenario 10

%% 3. Generate Synthetic Current Data (Multi-Sine Approach)
ik_scenarios = zeros(num_scenarios, length(t)); % Initialize current scenarios

for s = 1:num_scenarios
    % Sum of three sine waves for each scenario
    ik_scenarios(s, :) = A(s,1)*sin(2*pi*t / T(s,1)) + ...
                         A(s,2)*sin(2*pi*t / T(s,2)) + ...
                         A(s,3)*sin(2*pi*t / T(s,3));
end

%% 4. Define Time Constants on Logarithmic Scale (Natural Log)
a = 0.01;        % Minimum tau
b = 100;         % Maximum tau
theta_j = linspace(log(a), log(b), n); % Linearly spaced in log domain (ln(tau))
tau_discrete_log = exp(theta_j);        % Exponentiate to get tau_j

%% 5. True DRT Parameters (R_discrete_true_log)
mu_log = log(10);    % Set peak to be at tau = 10 (ln(tau) = log(10))
sigma_log = 0.5;     % Standard deviation of the true DRT distribution in log domain

% True Resistance Vector based on a Gaussian distribution over ln(tau)
R_discrete_true_log = normpdf(theta_j, mu_log, sigma_log);
R_discrete_true_log = R_discrete_true_log / max(R_discrete_true_log);  % Normalize to max value of 1

%% 6. Define Regularization Matrix L (1st Order Difference)
% 원래의 1차 차분 행렬 L 생성
L = zeros(n-1, n);
for i = 1:n-1
    L(i, i) = -1;
    L(i, i+1) = 1;
end

%% 7. Initialize Storage for Results
R_estimated_analytic = zeros(num_scenarios, n);   % Analytical DRT estimates
V_est_all = zeros(num_scenarios, length(t));      % Estimated voltages for all scenarios
V_sd_all = zeros(num_scenarios, length(t));       % Synthetic measured voltages for all scenarios

%% 8. Voltage Synthesis and Plotting Setup
figure(1);  
sgtitle('Current and Voltage for Each Scenario');

% Plot Current for each scenario
for s = 1:num_scenarios
    subplot(5, 2, s);
    yyaxis left
    plot(t, ik_scenarios(s, :), 'b-', 'LineWidth', 1.5);
    ylabel('Current (A)');
    xlabel('Time (s)');
    grid on;
end

%% 9. Processing Each Scenario (Logarithmic Time Scale & Analytical Solution)
for s = 1:num_scenarios
    fprintf('Processing Scenario %d/%d...\n', s, num_scenarios);
    
    % Current for the scenario (m x 1)
    ik = ik_scenarios(s, :)';  % Transpose to column vector (m x 1)
    
    %% 9.1. Initialize Voltage
    V_est = zeros(length(t),1);      % Estimated voltage via n-RC model (m x 1)
    R0 = 0.1;                         % Ohmic resistance (Ohms)
    OCV = 0;                        % Open Circuit Voltage (V)
    V_RC = zeros(n, length(t));       % RC voltages for each element (n x m)
    
    %% 9.2. Initial Voltage Calculation (First Time Step)
    for i = 1:n
        V_RC(i, 1) = ik(1) * R_discrete_true_log(i) * (1 - exp(-dt / tau_discrete_log(i)));
    end
    V_est(1) = OCV + R0 * ik(1) + sum(V_RC(:, 1));
    
    %% 9.3. Voltage Calculation for t > 1
    for k_idx = 2:length(t)
        for i = 1:n
            % Calculate RC voltage using previous time step data
            V_RC(i, k_idx) = exp(-dt / tau_discrete_log(i)) * V_RC(i, k_idx-1) + ...
                              R_discrete_true_log(i) * (1 - exp(-dt / tau_discrete_log(i))) * ik(k_idx);       
        end
        V_est(k_idx) = OCV + R0 * ik(k_idx) + sum(V_RC(:, k_idx));
    end
    
    % Store V_est for the current scenario
    V_est_all(s, :) = V_est';  % Store as row vector (1 x m)
    
    %% 9.4. Add Noise to the Voltage
    rng(s);  % Ensure reproducibility of noise for each scenario
    noise_level = 0.01;
    V_sd = V_est + noise_level * randn(size(V_est));  % V_sd = synthetic measured voltage (m x 1)
    
    % Store V_sd for the current scenario
    V_sd_all(s, :) = V_sd';  % Store as row vector (1 x m)
    
    %% 9.5. Construct System Matrix W for Logarithmic Time Scale
    W = zeros(length(t), n);  % Initialize W matrix (m x n)
    for j = 1:n
        for k_idx = 1:length(t)
            if k_idx == 1
                W(k_idx,j) = ik(k_idx) * (1 - exp(-dt / tau_discrete_log(j)));
            else
                W(k_idx,j) = exp(-dt / tau_discrete_log(j)) * W(k_idx-1,j) + ...
                            ik(k_idx) * (1 - exp(-dt / tau_discrete_log(j)));
            end
        end
    end
    
    %% 9.6. Analytical Solution with Regularization using L
    % Compute (W^T W + lambda * L^T L)
    A_matrix = W' * W + lambda * (L' * L);  % (n x n)
    
    % Compute W^T (V_sd - OCV - I * R0)
    b_vector = W' * (V_sd - OCV - ik * R0);  % (n x 1)
    
    % Solve for R using the analytical solution
    R_analytic = A_matrix \ b_vector;         % (n x 1)
    
    % Enforce non-negativity
    R_analytic(R_analytic < 0) = 0;
    
    % Store Analytical DRT
    R_estimated_analytic(s, :) = R_analytic';  % Store as row vector (1 x n)
    
    %% 9.7. Plot Voltage on Existing Subplots
    subplot(5, 2, s);
    yyaxis right
    plot(t, V_sd, 'r-', 'LineWidth', 1.5);
    ylabel('Voltage (V)');
    ylim([min(V_sd)-0.1, max(V_sd)+0.1]);
    
    % Update title with correct amplitudes and periods
    title(['Scenario ', num2str(s), ...
           ': A1=', num2str(A(s,1)), ', A2=', num2str(A(s,2)), ', A3=', num2str(A(s,3)), ...
           ', T1=', num2str(T(s,1)), ', T2=', num2str(T(s,2)), ', T3=', num2str(T(s,3))]);
    
    % Add legend
    legend({'Current (A)', 'Voltage (V)'}, 'Location', 'best');
end

%% 10. Plot the DRT Comparison for Each Scenario
for s = 1:num_scenarios
    figure(1 + s);  % DRT Comparison Figure for each scenario
    hold on;
    
    % Plot True DRT vs ln(tau)
    plot(theta_j, R_discrete_true_log, 'k-', 'LineWidth', 2, 'DisplayName', 'True DRT');
    
    % Plot Analytical DRT vs ln(tau)
    plot(theta_j, R_estimated_analytic(s, :), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Analytical DRT');
    
    hold off;
    xlabel('ln(\tau) (Logarithmic Time Constant)');
    ylabel('R (Resistance)');
    title(['DRT Comparison for Scenario ', num2str(s), ' (\lambda = ', num2str(lambda), ')']);
    legend('Location', 'BestOutside');
    grid on;
end

%% 11. Summary Plot Comparing Analytical Solution Across All Scenarios
figure;
hold on;
colors = lines(num_scenarios);

for s = 1:num_scenarios
    plot(theta_j, R_estimated_analytic(s, :), '--', 'Color', colors(s,:), 'LineWidth', 1, 'DisplayName', ['Analytical S', num2str(s)]);
end

% Plot True DRT vs ln(tau)
plot(theta_j, R_discrete_true_log, 'k-', 'LineWidth', 2, 'DisplayName', 'True DRT');

xlabel('ln(\tau) (Logarithmic Time Constant)');
ylabel('R (Resistance)');
title('DRT Estimation Comparison Across All Scenarios (Analytical Solution)');
legend('Location', 'BestOutside');
grid on;
hold off;

%% 12. Verify Dimensions (Optional)
disp('Dimensions of matrices:');
disp(['W: ', mat2str(size(W))]);
disp(['R_estimated_analytic: ', mat2str(size(R_estimated_analytic))]);
disp(['L: ', mat2str(size(L))]);

