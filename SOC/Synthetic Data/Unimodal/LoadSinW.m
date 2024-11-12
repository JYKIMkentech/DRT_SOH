clc; clear; close all;

% Set the random seed to ensure reproducibility
rng(0);  

%% Parameters
num_scenarios = 10;  % 시나리오 수
num_waves = 3;       % 각 시나리오당 사인파 수
t = linspace(0, 1000, 10000)';  % 시간 벡터 (0~1000초, 샘플링 포인트 10000개)
dt = t(2) - t(1);
n = 201; % true n

%% Constraints
T_min = 15;           % 최소 주기 (초)
T_max = 250;          % 최대 주기 (초)

%% Current calculation

A = zeros(num_scenarios, num_waves);   % 진폭 행렬
T = zeros(num_scenarios, num_waves);   % 주기 행렬
ik_scenarios = zeros(num_scenarios, length(t)); % 전류 시나리오 행렬

% Generate random amplitudes, periods, and current scenarios
for s = 1:num_scenarios
    % Random amplitudes that sum to 3
    temp_A = rand(1, num_waves);       % 3개의 랜덤 진폭 생성
    A(s, :) = 3 * temp_A / sum(temp_A);  % 진폭 합이 3이 되도록 정규화
    
    % Random periods between T_min and T_max on a linear scale
    T(s, :) = T_min + (T_max - T_min) * rand(1, num_waves);  % 선형 스케일에서 주기 생성
    
    % Generate the current scenario as the sum of three sine waves
    ik_scenarios(s, :) = A(s,1)*sin(2*pi*t / T(s,1)) + ...
                         A(s,2)*sin(2*pi*t / T(s,2)) + ...
                         A(s,3)*sin(2*pi*t / T(s,3));
end

%% DRT parameters for Unimodal (AS1)

% Unimodal gamma distribution parameters
mu_theta = log(10);       % 평균 값
sigma_theta = 1;          % 표준편차 값

% Discretized theta values (-3sigma to +3sigma)
theta_min = mu_theta - 3*sigma_theta;
theta_max = mu_theta + 3*sigma_theta;
theta_discrete = linspace(theta_min, theta_max, n);

% Corresponding tau values
tau_discrete = exp(theta_discrete);

% Delta theta
delta_theta = theta_discrete(2) - theta_discrete(1);

% Unimodal gamma distribution
gamma_discrete_true_unimodal = (1/(sigma_theta * sqrt(2*pi))) * exp(- (theta_discrete - mu_theta).^2 / (2 * sigma_theta^2));

% Normalize gamma to have a maximum of 1
gamma_discrete_true_unimodal = gamma_discrete_true_unimodal / max(gamma_discrete_true_unimodal);

%% DRT parameters for Bimodal (AS2)

% Bimodal gamma distribution parameters
% First peak
mu_theta1 = log(10);     % 첫 번째 피크의 중심 위치
sigma_theta1 = 1;        % 첫 번째 피크의 폭

% Second peak
mu_theta2 = log(120);    % 두 번째 피크의 중심 위치
sigma_theta2 = 0.7;      % 두 번째 피크의 폭

theta_min_bi = min([mu_theta1, mu_theta2]) - 3 * max([sigma_theta1, sigma_theta2]);
theta_max_bi = max([mu_theta1, mu_theta2]) + 3 * max([sigma_theta1, sigma_theta2]);
theta_discrete_bi = linspace(theta_min_bi, theta_max_bi, n);

% Corresponding tau values
tau_discrete_bi = exp(theta_discrete_bi);

% Delta theta
delta_theta_bi = theta_discrete_bi(2) - theta_discrete_bi(1);

% Calculate individual Gaussian distributions
gamma1 = (1 / (sigma_theta1 * sqrt(2 * pi))) * exp(- (theta_discrete_bi - mu_theta1).^2 / (2 * sigma_theta1^2));
gamma2 = (1 / (sigma_theta2 * sqrt(2 * pi))) * exp(- (theta_discrete_bi - mu_theta2).^2 / (2 * sigma_theta2^2));

% Combine
gamma_discrete_true_bimodal = gamma1 + gamma2;

% Normalize
gamma_discrete_true_bimodal = gamma_discrete_true_bimodal / max(gamma_discrete_true_bimodal);

%% Initialize structs
AS1_1per = struct('V', {}, 'I', {}, 't', {});
AS1_2per = struct('V', {}, 'I', {}, 't', {});
AS2_1per = struct('V', {}, 'I', {}, 't', {});
AS2_2per = struct('V', {}, 'I', {}, 't', {});

%% Voltage calculation and Struct storage for Unimodal (AS1)

% Parameters
gamma_discrete_true = gamma_discrete_true_unimodal;
tau_discrete_current = tau_discrete;
delta_theta_current = delta_theta;

for s = 1:num_scenarios
    fprintf('Processing Scenario %d/%d for Unimodal...\n', s, num_scenarios);
    
    % Current scenario's current
    ik = ik_scenarios(s, :)';  % Transpose for column vector
    
    %% Initialize voltage
    V_est = zeros(length(t), 1);  % Estimated voltage
    R0 = 0.1;  % Resistance (Ohms)
    OCV = 0;   % Open Circuit Voltage
    V_RC = zeros(n, length(t));  % Voltage across RC elements
    
    %% Voltage calculation
    for k_idx = 1:length(t)
        if k_idx == 1
            for i = 1:n
                V_RC(i, k_idx) = gamma_discrete_true(i) * delta_theta_current * ik(k_idx) * (1 - exp(-dt / tau_discrete_current(i)));
            end
        else
            for i = 1:n
                V_RC(i, k_idx) = V_RC(i, k_idx-1) * exp(-dt / tau_discrete_current(i)) + ...
                                 gamma_discrete_true(i) * delta_theta_current * ik(k_idx) * (1 - exp(-dt / tau_discrete_current(i)));
            end
        end
        V_est(k_idx) = OCV + R0 * ik(k_idx) + sum(V_RC(:, k_idx));
    end
    
    % Add 1% Gaussian noise
    noise_std_1per = 0.01 * max(V_est);  % 1% of the maximum voltage
    V_sd_1per = V_est + noise_std_1per * randn(size(V_est));
    
    % Store in AS1_1per struct
    AS1_1per(s).V = V_sd_1per;
    AS1_1per(s).I = ik;
    AS1_1per(s).t = t;
    
    % Add 2% Gaussian noise
    noise_std_2per = 0.02 * max(V_est);  % 2% of the maximum voltage
    V_sd_2per = V_est + noise_std_2per * randn(size(V_est));
    
    % Store in AS1_2per struct
    AS1_2per(s).V = V_sd_2per;
    AS1_2per(s).I = ik;
    AS1_2per(s).t = t;
end

%% Voltage calculation and Struct storage for Bimodal (AS2)

% Parameters
gamma_discrete_true = gamma_discrete_true_bimodal;
tau_discrete_current = tau_discrete_bi;
delta_theta_current = delta_theta_bi;

for s = 1:num_scenarios
    fprintf('Processing Scenario %d/%d for Bimodal...\n', s, num_scenarios);
    
    % Current scenario's current
    ik = ik_scenarios(s, :)';  % Transpose for column vector
    
    %% Initialize voltage
    V_est = zeros(length(t), 1);  % Estimated voltage
    R0 = 0.1;  % Resistance (Ohms)
    OCV = 0;   % Open Circuit Voltage
    V_RC = zeros(n, length(t));  % Voltage across RC elements
    
    %% Voltage calculation
    for k_idx = 1:length(t)
        if k_idx == 1
            for i = 1:n
                V_RC(i, k_idx) = gamma_discrete_true(i) * delta_theta_current * ik(k_idx) * (1 - exp(-dt / tau_discrete_current(i)));
            end
        else
            for i = 1:n
                V_RC(i, k_idx) = V_RC(i, k_idx-1) * exp(-dt / tau_discrete_current(i)) + ...
                                 gamma_discrete_true(i) * delta_theta_current * ik(k_idx) * (1 - exp(-dt / tau_discrete_current(i)));
            end
        end
        V_est(k_idx) = OCV + R0 * ik(k_idx) + sum(V_RC(:, k_idx));
    end
    
    % Add 1% Gaussian noise
    noise_std_1per = 0.01 ;  
    V_sd_1per = V_est + noise_std_1per * randn(size(V_est));
    
    % Store in AS2_1per struct
    AS2_1per(s).V = V_sd_1per;
    AS2_1per(s).I = ik;
    AS2_1per(s).t = t;
    
    % Add 2% Gaussian noise
    noise_std_2per = 0.02 * mean(V_est);  
    V_sd_2per = V_est + noise_std_2per * randn(size(V_est));
    
    % Store in AS2_2per struct
    AS2_2per(s).V = V_sd_2per;
    AS2_2per(s).I = ik;
    AS2_2per(s).t = t;
end

%% Plot


plot_scenarios(AS1_1per, 'AS1 Unimodal with 1% Noise');
plot_scenarios(AS1_2per, 'AS1 Unimodal with 2% Noise');
plot_scenarios(AS2_1per, 'AS2 Bimodal with 1% Noise');
plot_scenarios(AS2_2per, 'AS2 Bimodal with 2% Noise');


%% Save structs

save('AS1_1per.mat', 'AS1_1per');
save('AS1_2per.mat', 'AS1_2per');
save('AS2_1per.mat', 'AS2_1per');
save('AS2_2per.mat', 'AS2_2per');

%% function
function plot_scenarios(scenarios, title_text)
    figure;
    num_scenarios = length(scenarios);
    for s = 1:num_scenarios
        subplot(5, 2, s);  % 5x2 grid of subplots
        yyaxis left;
        plot(scenarios(s).t, scenarios(s).I, 'b', 'LineWidth', 1.5);
        ylabel('Current (A)');
        yyaxis right;
        plot(scenarios(s).t, scenarios(s).V, 'r', 'LineWidth', 1.5);
        ylabel('Voltage (V)');
        title(['Scenario ', num2str(s)]);
        xlabel('Time (s)');
        grid on;
        legend('Current', 'Voltage');
    end
    sgtitle(title_text);
end



