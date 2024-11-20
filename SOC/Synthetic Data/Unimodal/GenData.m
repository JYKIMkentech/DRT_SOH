clc; clear; close all;

% Set random seed for reproducibility
rng(0);

%% Path
save_path = 'G:\공유 드라이브\Battery Software Lab\Projects\DRT\SD\';

%% Parameters
num_scenarios = 10;  % Number of scenarios
num_waves = 3;       % Number of sine waves per scenario
t = linspace(0, 1000, 10001)';  % Time vector (0~1000 seconds, 10000 sample points)
dt = t(2) - t(1);
n = 201; % Number of RC elements

%% Setting T (Period)
T_min = 15;           % Minimum period (seconds)
T_max = 250;          % Maximum period (seconds)

%% Current Calculation
A = zeros(num_scenarios, num_waves);        % Amplitude matrix
T = zeros(num_scenarios, num_waves);        % Period matrix
ik_scenarios = zeros(num_scenarios, length(t)); % Current scenarios matrix
R0 = 0.1;
OCV = 0;

% Generate random amplitudes and periods, and create current scenarios
for s = 1:num_scenarios
    % Generate amplitudes (normalized to sum to 3)
    temp_A = rand(1, num_waves);
    A(s, :) = 3 * temp_A / sum(temp_A);
    
    % Generate periods (random between T_min and T_max)
    T(s, :) = T_min + (T_max - T_min) * rand(1, num_waves);
    
    % Create current scenario (sum of three sine waves)
    ik_scenarios(s, :) = A(s,1)*sin(2*pi*t / T(s,1)) + ...
                         A(s,2)*sin(2*pi*t / T(s,2)) + ...
                         A(s,3)*sin(2*pi*t / T(s,3));
end

%% True DRT Parameter Setting

% Unimodal
mu_theta_uni = log(10);
sigma_theta_uni = 1.536;
theta_min_uni = mu_theta_uni - 3*sigma_theta_uni;
theta_max_uni = mu_theta_uni + 3*sigma_theta_uni;
theta_discrete_uni = linspace(theta_min_uni, theta_max_uni, n);
tau_discrete_uni = exp(theta_discrete_uni);
delta_theta_uni = theta_discrete_uni(2) - theta_discrete_uni(1);

gamma_discrete_true_unimodal = (1/(sigma_theta_uni * sqrt(2*pi))) * exp(- (theta_discrete_uni - mu_theta_uni).^2 / (2 * sigma_theta_uni^2));
gamma_discrete_true_unimodal = gamma_discrete_true_unimodal / max(gamma_discrete_true_unimodal);

% Bimodal

% First peak
mu_theta1 = log(10);
sigma_theta1 = 1;

% Second peak
mu_theta2 = log(120);
sigma_theta2 = 0.7;

theta_min_bi = min([mu_theta1, mu_theta2]) - 3 * max([sigma_theta1, sigma_theta2]);
theta_max_bi = max([mu_theta1, mu_theta2]) + 3 * max([sigma_theta1, sigma_theta2]);
theta_discrete_bi = linspace(theta_min_bi, theta_max_bi, n);
tau_discrete_bi = exp(theta_discrete_bi);
delta_theta_bi = theta_discrete_bi(2) - theta_discrete_bi(1);

gamma1 = (1 / (sigma_theta1 * sqrt(2 * pi))) * exp(- (theta_discrete_bi - mu_theta1).^2 / (2 * sigma_theta1^2));
gamma2 = (1 / (sigma_theta2 * sqrt(2 * pi))) * exp(- (theta_discrete_bi - mu_theta2).^2 / (2 * sigma_theta2^2));

% Combine
gamma_discrete_true_bimodal = gamma1 + gamma2;
gamma_discrete_true_bimodal = gamma_discrete_true_bimodal / max(gamma_discrete_true_bimodal);

%% Initialize Structs
% `SN` 필드를 가장 앞에 정의하여 구조체의 첫 번째 필드로 설정
AS1_1per = struct('SN', {}, 'V', {}, 'I', {}, 't', {});
AS1_2per = struct('SN', {}, 'V', {}, 'I', {}, 't', {});
AS2_1per = struct('SN', {}, 'V', {}, 'I', {}, 't', {});
AS2_2per = struct('SN', {}, 'V', {}, 'I', {}, 't', {});

%% Voltage Calculation (Unimodal)
gamma_discrete_true = gamma_discrete_true_unimodal;
tau_discrete_current = tau_discrete_uni;
delta_theta_current = delta_theta_uni;

for s = 1:num_scenarios
    fprintf('Processing Unimodal Scenario %d/%d...\n', s, num_scenarios);
    
    ik = ik_scenarios(s, :)';  % Current data (column vector)
    
    % Initialize voltage
    V_est = zeros(length(t), 1);   
    V_RC = zeros(n, length(t));
    
    % Voltage calculation
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
    
    % Add noise (1%)
    noise_level = 0.01;
    V_sd_1per = V_est + noise_level .* V_est .* randn(size(V_est));

    % Add noise (2%)
    noise_level = 0.02;
    V_sd_2per = V_est + noise_level .* V_est .* randn(size(V_est));

    % Save data with SN field
    AS1_1per(s).SN = s;          % 시나리오 번호 추가 (가장 앞에 위치)
    AS1_1per(s).V = V_sd_1per;
    AS1_1per(s).I = ik;
    AS1_1per(s).t = t;

    AS1_2per(s).SN = s;          % 시나리오 번호 추가 (가장 앞에 위치)
    AS1_2per(s).V = V_sd_2per;
    AS1_2per(s).I = ik;
    AS1_2per(s).t = t;
end

%% Voltage Calculation (Bimodal)
gamma_discrete_true = gamma_discrete_true_bimodal;
tau_discrete_current = tau_discrete_bi;
delta_theta_current = delta_theta_bi;
theta_discrete_current = theta_discrete_bi;

for s = 1:num_scenarios
    fprintf('Processing Bimodal Scenario %d/%d...\n', s, num_scenarios);
    
    ik = ik_scenarios(s, :)';  % Current data (column vector)
    
    % Initialize voltage
    V_est = zeros(length(t), 1);
    V_RC = zeros(n, length(t));
    
    % Voltage calculation
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
    
    % Add noise (1%)
    noise_level = 0.01;
    V_sd_1per = V_est + noise_level .* V_est .* randn(size(V_est));
    
    % Add noise (2%)
    noise_level = 0.02;
    V_sd_2per = V_est + noise_level .* V_est .* randn(size(V_est));
    
    % Save data with SN field
    AS2_1per(s).SN = s;          % 시나리오 번호 추가 (가장 앞에 위치)
    AS2_1per(s).V = V_sd_1per;
    AS2_1per(s).I = ik;
    AS2_1per(s).t = t;

    AS2_2per(s).SN = s;          % 시나리오 번호 추가 (가장 앞에 위치)
    AS2_2per(s).V = V_sd_2per;
    AS2_2per(s).I = ik;
    AS2_2per(s).t = t;
end


%% Plot current and voltage

% Define figure names for each case
figure_names = {'AS1 Unimodal with 1% Noise', 'AS1 Unimodal with 2% Noise', ...
                'AS2 Bimodal with 1% Noise', 'AS2 Bimodal with 2% Noise'};
            
% Define the structures corresponding to each case
struct_cases = {AS1_1per, AS1_2per, AS2_1per, AS2_2per};

% Define colors for different scenarios
c_mat = lines(num_scenarios);  % MATLAB's default color matrix

% Loop through each case
for case_idx = 1:length(struct_cases)
    current_case = struct_cases{case_idx};
    figure('Name', figure_names{case_idx}, 'NumberTitle', 'off');
    
    % Create a grid of subplots (e.g., 5 rows x 2 columns for 10 scenarios)
    num_rows = 5;
    num_cols = 2;
    
    for s = 1:num_scenarios
        subplot(num_rows, num_cols, s);
        
        yyaxis left
        plot(current_case(s).t, current_case(s).I, 'b-', 'LineWidth', 1.5);
        ylabel('Current (A)', 'FontSize', 12);
        hold on;
        
        yyaxis right
        plot(current_case(s).t, current_case(s).V, 'r-', 'LineWidth', 1.5);
        ylabel('Voltage (V)', 'FontSize', 12);
        
        xlabel('Time (s)', 'FontSize', 12);
        title(['Scenario ', num2str(current_case(s).SN)], 'FontSize', 14);  % 시나리오 번호 사용
        legend('Current', 'Voltage', 'Location', 'best');
        hold off;
    end
    
    sgtitle(figure_names{case_idx}, 'FontSize', 16);
end


% Create structs for gamma vs theta
Gamma_unimodal.theta = theta_discrete_uni';
Gamma_unimodal.gamma = gamma_discrete_true_unimodal';

Gamma_bimodal.theta = theta_discrete_bi';
Gamma_bimodal.gamma = gamma_discrete_true_bimodal';

% Unimodal
figure;
plot(Gamma_unimodal.theta, Gamma_unimodal.gamma, 'b-', 'LineWidth', 1.5);
xlabel('\theta');
ylabel('\gamma');
title('Unimodal \gamma vs \theta');

% Bimodal
figure;
plot(Gamma_bimodal.theta, Gamma_bimodal.gamma, 'r-', 'LineWidth', 1.5);
xlabel('\theta');
ylabel('\gamma');
title('Bimodal \gamma vs \theta');


%% Save

% % Save the current, voltage scenarios
% save(fullfile(save_path, 'AS1_1per.mat'), 'AS1_1per');
% save(fullfile(save_path, 'AS1_2per.mat'), 'AS1_2per');
% save(fullfile(save_path, 'AS2_1per.mat'), 'AS2_1per');
% save(fullfile(save_path, 'AS2_2per.mat'), 'AS2_2per');
% 
% % Save the True Gamma vs theta
% save(fullfile(save_path, 'Gamma_unimodal.mat'), 'Gamma_unimodal');
% save(fullfile(save_path, 'Gamma_bimodal.mat'), 'Gamma_bimodal');


