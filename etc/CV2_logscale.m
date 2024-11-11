clc; clear; close all;

%% ------------------------- Parameters -------------------------
% Number of RC elements
n = 21;  % RC 요소의 수

% Time vector: 0 to 100 seconds with 10001 points
t = 0:0.01:100;
dt = t(2) - t(1); % 시간 간격

% Number of current scenarios
num_scenarios = 10;  % 전류 시나리오의 수

% Regularization parameter
lambda = 0.01;  % 정규화 매개변수

%% ------------------- Define Amplitudes and Periods for Current Synthesis -------------------
% 각 시나리오에 대한 세 개의 사인파 진폭
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

% 진폭에 대응하는 주기
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

%% ------------------- Generate Synthetic Current Data (Multi-Sine Approach) -------------------
% 전류 시나리오를 저장할 행렬 초기화
ik_scenarios = zeros(num_scenarios, length(t));

for s = 1:num_scenarios
    % 각 시나리오에 대한 세 개의 사인파 합성
    ik_scenarios(s, :) = A(s,1)*sin(2*pi*t / T(s,1)) + ...
                         A(s,2)*sin(2*pi*t / T(s,2)) + ...
                         A(s,3)*sin(2*pi*t / T(s,3));
end

%% ------------------- DRT Parameters -------------------
% ln(τ)가 정규분포를 따르도록 파라미터 정의
mu_theta = log(10);  % ln(τ)의 평균
sigma_theta = 0.2;   % ln(τ)의 표준편차

% θ (ln(τ))의 범위 정의
theta_min = mu_theta - 3*sigma_theta;
theta_max = mu_theta + 3*sigma_theta;
theta_discrete = linspace(theta_min, theta_max, n);  % θ의 이산화

% θ로부터 τ 계산
tau_discrete = exp(theta_discrete);  % τ = e^θ

% Δθ (θ가 균등 간격이므로)
delta_theta = theta_discrete(2) - theta_discrete(1);  % Δθ = θ_{n+1} - θ_n

% True DRT [θ, γ]
gamma_discrete_true = normpdf(theta_discrete, mu_theta, sigma_theta);

% R_n_true 계산 (스케일링 적용)
R_n_true = gamma_discrete_true .* tau_discrete .* delta_theta;  % R_n = γ_n * τ_n * Δθ

%% ------------------- Regularization Matrix L (First-order derivative) -------------------
% 1차 도함수를 위한 차분 행렬 D 생성
D = zeros(n-1, n);
for i = 1:n-1
    D(i, i) = -1;
    D(i, i+1) = 1;
end

% Δθ로 정규화된 정규화 행렬 L
L = (1 / delta_theta) * D;

%% ------------------- Initialize Storage Variables -------------------
W_all = cell(num_scenarios, 1);          % 모든 시나리오의 W 행렬 저장
V_sd_all = zeros(num_scenarios, length(t));   % 모든 시나리오의 노이즈가 추가된 전압 데이터 저장
V_est_all = zeros(num_scenarios, length(t));  % 추정된 전압 데이터 저장

%% ------------------- Plot Synthesized Currents and Voltages -------------------
figure;
sgtitle('Synthesized Current and Voltage for Each Scenario');

% 각 시나리오의 전류 플롯 초기화
for s = 1:num_scenarios
    subplot(5, 2, s);
    yyaxis left
    plot(t, ik_scenarios(s, :), 'b-', 'LineWidth', 1.5);
    ylabel('Current (A)');
    xlabel('Time (s)');
    grid on;
end

%% ------------------- Voltage Synthesis -------------------
for s = 1:num_scenarios
    fprintf('Processing Scenario %d/%d...\n', s, num_scenarios);
    
    % 현재 시나리오의 전류
    ik = ik_scenarios(s, :);  % 전류 시나리오 입력
    
    %% Initialize Voltage
    V_est = zeros(1, length(t));        % n-RC 모델을 통한 모델 전압 계산
    R0 = 0.1;                            % 오믹 저항 (R0)
    OCV = 0;                             % 개방 회로 전압 (OCV)
    V_RC = zeros(n, length(t));         % 각 요소에 대한 RC 전압
    
    %% Initial Voltage Calculation (첫 번째 시간 단계)
    for i = 1:n
        V_RC(i, 1) = R_n_true(i) * (1 - exp(-dt / tau_discrete(i))) * ik(1);
    end
    V_est(1) = OCV + R0 * ik(1) + sum(V_RC(:, 1));
    
    %% Voltage Calculation for t > 1
    for k_idx = 2:length(t)
        for i = 1:n
            V_RC(i, k_idx) = exp(-dt / tau_discrete(i)) * V_RC(i, k_idx-1) + ...
                              R_n_true(i) * (1 - exp(-dt / tau_discrete(i))) * ik(k_idx);       
        end
        V_est(k_idx) = OCV + R0 * ik(k_idx) + sum(V_RC(:, k_idx));
    end
    
    % 현재 시나리오의 V_est 저장
    V_est_all(s, :) = V_est;  % 계산된 V_est 저장
    
    %% Add Noise to the Voltage
    rng(s);  % 각 시나리오에 대한 노이즈의 재현성 보장
    noise_level = 0.01;
    V_sd = V_est + noise_level * randn(size(V_est));  % V_sd = 합성된 측정 전압
    
    % 현재 시나리오의 V_sd 저장
    V_sd_all(s, :) = V_sd;  % 노이즈가 추가된 V_sd 저장
    
    %% Plot Voltage on Existing Subplots
    subplot(5, 2, s);
    yyaxis right
    plot(t, V_sd, 'r-', 'LineWidth', 1.5);
    ylabel('Voltage (V)');
    ylim([min(V_sd)-0.1, max(V_sd)+0.1]);
    
    % 진폭 및 주기를 포함한 제목 업데이트
    title(['Scenario ', num2str(s), ...
           ': A1=', num2str(A(s,1)), ', A2=', num2str(A(s,2)), ', A3=', num2str(A(s,3)), ...
           ', T1=', num2str(T(s,1)), ', T2=', num2str(T(s,2)), ', T3=', num2str(T(s,3))]);
    
    % 범례 추가
    legend({'Current (A)', 'Voltage (V)'}, 'Location', 'best');
    
    %% Construct W Matrix for Scenario s
    W = zeros(length(t), n);  % W 행렬 초기화
    for k_idx = 1:length(t)
        for i = 1:n
            if k_idx == 1
                W(k_idx, i) = (1 - exp(-dt / tau_discrete(i))) * ik(k_idx);
            else
                W(k_idx, i) = exp(-dt / tau_discrete(i)) * W(k_idx-1, i) + ...
                              (1 - exp(-dt / tau_discrete(i))) * ik(k_idx);
            end
        end
    end
    W_all{s} = W; % W 행렬 저장
end

%% ------------------- DRT Estimation for Scenario 1 -------------------
% 시나리오 1의 W와 y 추출
W_s1 = W_all{1};  % 시나리오 1의 W 행렬
y_s1 = V_sd_all(1, :)' - OCV - R0 * ik_scenarios(1, :)';  % y = V_sd - OCV - R0*I

% τ와 Δθ로 W를 스케일링하여 W_gamma 생성
W_gamma = W_s1 .* (tau_discrete .* delta_theta);

% 정규화된 최소 자승법을 사용하여 Gamma 추정
gamma_estimated_s1 = (W_gamma' * W_gamma + lambda * (L' * L)) \ (W_gamma' * y_s1);
gamma_estimated_s1(gamma_estimated_s1 < 0) = 0;  % 비음수 조건 강제

% 비교를 위한 gamma_estimated 정규화
gamma_estimated_s1_normalized = gamma_estimated_s1 / max(gamma_estimated_s1);

%% ------------------- Plot the DRT Comparison for Scenario 1 -------------------
figure;
plot(theta_discrete, gamma_discrete_true / max(gamma_discrete_true), 'k-', 'LineWidth', 2, 'DisplayName', 'True DRT');
hold on;
plot(theta_discrete, gamma_estimated_s1_normalized, 'b--', 'LineWidth', 2, 'DisplayName', 'Estimated DRT');
xlabel('\theta = ln(\tau)');
ylabel('\gamma(\theta)');
title(['DRT Comparison (\lambda = ', num2str(lambda), ')']);
legend('Location', 'Best');
grid on;
hold off;

% 그래프 레이아웃 조정
set(gcf, 'Units', 'normalized', 'Position', [0.1, 0.1, 0.8, 0.6]);  % 화면 비율

%% ------------------- Display Total Squared Voltage Error for Each Scenario -------------------
total_test_error = 0;
for s = 1:num_scenarios
    % 추정된 gamma를 사용하여 시나리오의 V_pred 예측
    R_n_estimated = gamma_estimated_s1 .* tau_discrete .* delta_theta;
    V_pred = W_all{s} * R_n_estimated + R0 * ik_scenarios(s, :)' + OCV;
    
    % 실제 노이즈가 추가된 전압
    V_actual = V_sd_all(s, :)';
    
    % 제곱 오차 계산
    error = sum((V_actual - V_pred).^2);
    total_test_error = total_test_error + error;
    
    fprintf('Scenario %d: Squared Voltage Error = %.4f\n', s, error);
end
fprintf('Total Squared Voltage Error across all scenarios: %.4f\n', total_test_error);


