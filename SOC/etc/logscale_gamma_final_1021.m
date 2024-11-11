clc; clear; close all;

%% AS1.mat 파일 로드
load('AS1.mat');  % 첫 번째 코드에서 저장한 A, T, ik_scenarios, t 변수를 불러옵니다.

%% Parameters 
n = 201;  % Number of discrete elements
dt = t(2) - t(1);  % Time step based on loaded time vector
num_scenarios = 10;  % Number of current scenarios
lambda = 0.51795;  % Regularization parameter

%% DRT 

% 세타 (ln (tau)) 가 정규분포를 따른다 ---> tau는 로그정규분포를 따른다. 
% (tau 평균 : 16.4697, 표준편차 : 21.635)


% Theta = ln(tau) (x축)
% gamma(theta) = [ R(exp(theta)) * exp(theta) ] = [ R(tau) * tau ] (y축)
% R_i = gamma_i * delta theta % 면적은 저항 = gamma (세로) * delta (가로, 일정하게)

% True DRT Parameters (gamma_discrete)
% mu_theta = -0.3404;       % 계산된 평균 값
% sigma_theta = 0.4991;     % 계산된 표준편차 값

mu_theta = log(10);       % 계산된 평균 값
sigma_theta = 1;     % 계산된 표준편차 값

% Discrete theta values (from -3sigma to +3sigma)
theta_min = mu_theta - 3*sigma_theta;
theta_max = mu_theta + 3*sigma_theta;
theta_discrete = linspace(theta_min, theta_max, n);

% Corresponding tau values
tau_discrete = exp(theta_discrete);

% Delta theta
delta_theta = theta_discrete(2) - theta_discrete(1);

% True gamma distribution
gamma_discrete_true = (1/(sigma_theta * sqrt(2*pi))) * exp(- (theta_discrete - mu_theta).^2 / (2 * sigma_theta^2));

% Normalize gamma to have a maximum value of 1
gamma_discrete_true = gamma_discrete_true / max(gamma_discrete_true);

% Analytical gamma estimates
gamma_analytical_all = zeros(num_scenarios, n);  % Analytical gamma estimates

% Voltage storage variables (V_est and V_sd for each scenario)
V_est_all = zeros(num_scenarios, length(t));  % For storing V_est for all scenarios
V_sd_all = zeros(num_scenarios, length(t));   % For storing V_sd for all scenarios

%% First-order difference matrix L
L = zeros(n-1, n);
for i = 1:n-1
    L(i, i) = -1;
    L(i, i+1) = 1;
end

%% Voltage Synthesis and DRT Estimation
for s = 1:num_scenarios
    fprintf('Processing Scenario %d/%d...\n', s, num_scenarios);
    
    % Current for the scenario
    ik = ik_scenarios(s, :);  % 로드된 전류 시나리오 사용
    
    %% Initialize Voltage
    V_est = zeros(1, length(t));  % Model voltage calculated via n-element model
    R0 = 0.1;  % Ohmic resistance
    OCV = 0;   % Open Circuit Voltage
    V_RC = zeros(n, length(t));  % Voltages for each element
    
    %% Voltage Calculation
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
    V_est_all(s, :) = V_est;  % Save the calculated V_est for this scenario
    
    %% Add Noise to the Voltage
    rng(0);  % Ensure reproducibility of noise
    noise_level = 0.01;
    V_sd = V_est + noise_level * randn(size(V_est));  % V_sd = synthetic measured voltage
    
    % Store V_sd for the current scenario
    V_sd_all(s, :) = V_sd;  % Save the noisy V_sd for this scenario
    
    %% Construct W Matrix
    W = zeros(length(t), n);  % Initialize W matrix
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
    
    %% Analytical Solution with Regularization
    % Remove constants: Subtract OCV and R0*ik
    y_adjusted = V_sd' - OCV - R0 * ik';
    
    % Regularized least squares solution
    gamma_analytical = (W' * W + lambda * (L' * L)) \ (W' * y_adjusted);
    
    %% Store Analytical gamma
    gamma_analytical_all(s, :) = gamma_analytical';
    
    %% Plot Voltage and DRT Comparison
    figure(1);  
    subplot(5, 2, s);
    yyaxis left
    plot(t, ik, 'b-', 'LineWidth', 1.5);
    ylabel('Current (A)');
    xlabel('Time (s)');
    grid on;
    
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
    
    % DRT Comparison Plot
    figure(1 + s);  % DRT Comparison Figure for each scenario
    hold on;
    
    % Plot True gamma
    plot(theta_discrete, gamma_discrete_true, 'k-', 'LineWidth', 1.5, 'DisplayName', 'True \gamma');
    
    % Plot Analytical gamma
    plot(theta_discrete, gamma_analytical_all(s, :), ':', 'Color', 'g', 'LineWidth', 1.5, 'DisplayName', 'Analytical \gamma');
    
    hold off;
    xlabel('\theta = ln(\tau)');
    ylabel('\gamma');
    title(['DRT Comparison for Scenario ', num2str(s), ' (\lambda = ', num2str(lambda), ')']);
    legend('Location', 'Best');
    grid on;
end
