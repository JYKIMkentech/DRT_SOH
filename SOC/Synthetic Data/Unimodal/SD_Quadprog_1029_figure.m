clc; clear; close all;

% Load AS1.mat File
load('AS1.mat');  % Load variables A, T, ik_scenarios, t

% Parameters
n = 201;  % Number of discrete elements
dt = t(2) - t(1);  % Time step based on loaded time vector
num_scenarios = 10;  % Number of current scenarios
lambda = 3.2375;  % Regularization parameter

% DRT
mu_theta = log(10);  % Mean value
sigma_theta = 1;     % Standard deviation
theta_min = mu_theta - 3*sigma_theta;
theta_max = mu_theta + 3*sigma_theta;
theta_discrete = linspace(theta_min, theta_max, n);
tau_discrete = exp(theta_discrete);
delta_theta = theta_discrete(2) - theta_discrete(1);
gamma_discrete_true = (1/(sigma_theta * sqrt(2*pi))) * exp(- (theta_discrete - mu_theta).^2 / (2 * sigma_theta^2));
gamma_discrete_true = gamma_discrete_true / max(gamma_discrete_true);

% Initialize storage variables
gamma_quadprog_all = zeros(num_scenarios, n);
V_est_all = zeros(num_scenarios, length(t));
V_sd_all = zeros(num_scenarios, length(t));

% First-order difference matrix L
L = zeros(n-1, n);
for i = 1:n-1
    L(i, i) = -1;
    L(i, i+1) = 1;
end

% Define Color Matrix Using lines(9)
c_mat = lines(9);  % Define a color matrix with 9 distinct colors

% Font size settings
axisFontSize = 14;
titleFontSize = 12;
legendFontSize = 12;
labelFontSize = 12;

% Voltage Synthesis and DRT Estimation using quadprog
for s = 1:num_scenarios
    fprintf('Processing Scenario %d/%d...\n', s, num_scenarios);
    ik = ik_scenarios(s, :);
    
    % Initialize Voltage
    V_est = zeros(1, length(t));
    R0 = 0.1;
    OCV = 0;
    V_RC = zeros(n, length(t));
    
    % Voltage Calculation
    for k_idx = 1:length(t)
        if k_idx == 1
            for i = 1:n
                V_RC(i, k_idx) = gamma_discrete_true(i) * delta_theta * ik(k_idx) * (1 - exp(-dt / tau_discrete(i)));
            end
        else
            for i = 1:n
                V_RC(i, k_idx) = V_RC(i, k_idx-1) * exp(-dt / tau_discrete(i)) + ...
                                 gamma_discrete_true(i) * delta_theta * ik(k_idx) * (1 - exp(-dt / tau_discrete(i)));
            end
        end
        V_est(k_idx) = OCV + R0 * ik(k_idx) + sum(V_RC(:, k_idx));
    end
    
    % Store V_est for the current scenario
    V_est_all(s, :) = V_est;
    
    % Add Noise to the Voltage
    rng(0);
    noise_level = 0.01;
    V_sd = V_est + noise_level * randn(size(V_est));
    V_sd_all(s, :) = V_sd;
    
    % Construct W Matrix
    W = zeros(length(t), n);
    for k_idx = 1:length(t)
        if k_idx == 1
            for i = 1:n
                W(k_idx, i) = ik(k_idx) * (1 - exp(-dt / tau_discrete(i))) * delta_theta;
            end
        else
            for i = 1:n
                W(k_idx, i) = W(k_idx-1, i) * exp(-dt / tau_discrete(i)) + ...
                              ik(k_idx) * (1 - exp(-dt / tau_discrete(i))) * delta_theta;
            end
        end
    end
    
    % DRT Estimation using quadprog
    y_adjusted = V_sd' - OCV - R0 * ik';
    H = 2 * (W' * W + lambda * (L' * L));
    f = -2 * W' * y_adjusted;
    lb = zeros(n,1);
    ub = [];
    options = optimoptions('quadprog','Display','off');
    gamma_quadprog = quadprog(H, f, [], [], [], [], lb, ub, [], options);
    gamma_quadprog_all(s, :) = gamma_quadprog';
    
    % Plot Voltage and DRT Comparison
    figure(1);
    subplot(5, 2, s);
    yyaxis left
    plot(t, ik, 'Color', c_mat(1, :), 'LineWidth', 1.5);
    ylabel('Current (A)', 'FontSize', labelFontSize);
    xlabel('Time (s)', 'FontSize', labelFontSize);
    
    
    yyaxis right
    plot(t, V_sd, 'Color', c_mat(2, :), 'LineWidth', 1.5);
    ylabel('Voltage (V)', 'FontSize', labelFontSize);
    ylim([min(V_sd)-0.1, max(V_sd)+0.1]);
    
%     title(['Scenario ', num2str(s), ...
%            ': A1=', num2str(A(s,1)), ', A2=', num2str(A(s,2)), ', A3=', num2str(A(s,3)), ...
%            ', T1=', num2str(T(s,1)), ', T2=', num2str(T(s,2)), ', T3=', num2str(T(s,3))], 'FontSize', titleFontSize);
      title(['Scenario ', num2str(s)], 'FontSize', titleFontSize);


    legend({'Current (A)', 'Voltage (V)'}, 'Location', 'best', 'FontSize', legendFontSize);
end

% Additional Figures for Scenarios 6, 7, 8, 9
selected_scenarios = [6, 7, 8, 9];

% Figure 2: I(t) and V(t) for Scenarios 6,7,8,9 as Subplots
figure(2);
for idx = 1:length(selected_scenarios)
    s = selected_scenarios(idx);
    subplot(2, 2, idx);
    yyaxis left
    plot(t, ik_scenarios(s, :), 'Color', c_mat(1, :), 'LineWidth', 1.5);
    ylabel('Current (A)', 'FontSize', labelFontSize);
    yyaxis right
    plot(t, V_sd_all(s, :), 'Color', c_mat(2, :), 'LineWidth', 1.5);
    ylabel('Voltage (V)', 'FontSize', labelFontSize);
    xlabel('Time (s)', 'FontSize', labelFontSize);
%     title(['Scenario ', num2str(s), ...
%            ': A1=', num2str(A(s,1)), ', A2=', num2str(A(s,2)), ', A3=', num2str(A(s,3)), ...
%            ', T1=', num2str(T(s,1)), ', T2=', num2str(T(s,2)), ', T3=', num2str(T(s,3))], 'FontSize', titleFontSize);
    title(['Scenario ', num2str(s)], 'FontSize', titleFontSize);



    legend({'Current (A)', 'Voltage (V)'}, 'Location', 'best', 'FontSize', legendFontSize);
   
end
sgtitle('Unimodal: Current and Voltage vs Time', 'FontSize', titleFontSize);

% Figure 3: gamma vs theta for Scenarios 6,7,8,9
figure(3);
hold on;
for idx = 1:length(selected_scenarios)
    s = selected_scenarios(idx);
    plot(theta_discrete, gamma_quadprog_all(s, :), '--', 'LineWidth', 1.5, ...
        'Color', c_mat(s, :), 'DisplayName', ['Scenario ', num2str(s)]);
end
plot(theta_discrete, gamma_discrete_true, 'k-', 'LineWidth', 2, 'DisplayName', 'True \gamma');
hold off;
xlabel('$\theta = \ln(\tau \, [s])$', 'Interpreter', 'latex', 'FontSize', labelFontSize)
ylabel('\gamma', 'FontSize', labelFontSize);
title('Unimodal : Estimated \gamma', 'FontSize', titleFontSize);
legend('Location', 'Best', 'FontSize', legendFontSize);



