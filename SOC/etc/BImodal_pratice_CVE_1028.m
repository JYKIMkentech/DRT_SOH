clc; clear; close all;

%% 1. DRT 추정 및 교차 검증 (Bimodal Case)

% Load the generated AS1.mat 파일
load('AS1.mat');  % Load A, T, ik_scenarios, t variables

%% Parameters 
n = 201;  % Number of discrete elements 
dt = t(2) - t(1);  % Time step based on loaded time vector
num_scenarios = 10;  % Number of current scenarios
lambda_values = logspace(-4, 9, 50);  % 람다 값 범위 설정

%% DRT 설정 (Bimodal Gamma Distribution)

% 이봉형(bimodal) 감마 분포 생성
% 첫 번째 피크의 평균과 표준편차
mu_theta1 = log(10);     % 첫 번째 피크의 중심 위치
sigma_theta1 = 1;        % 첫 번째 피크의 폭

% 두 번째 피크의 평균과 표준편차
mu_theta2 = log(120);    % 두 번째 피크의 중심 위치
sigma_theta2 = 0.7;      % 두 번째 피크의 폭

% Theta 범위 설정 (두 피크를 모두 포함하도록)
theta_min = min([mu_theta1, mu_theta2]) - 3 * max([sigma_theta1, sigma_theta2]);
theta_max = max([mu_theta1, mu_theta2]) + 3 * max([sigma_theta1, sigma_theta2]);
theta_discrete = linspace(theta_min, theta_max, n);

% Corresponding tau values
tau_discrete = exp(theta_discrete);

% Delta theta
delta_theta = theta_discrete(2) - theta_discrete(1);

% 각 가우시안 분포 계산
gamma1 = (1 / (sigma_theta1 * sqrt(2 * pi))) * exp(- (theta_discrete - mu_theta1).^2 / (2 * sigma_theta1^2));
gamma2 = (1 / (sigma_theta2 * sqrt(2 * pi))) * exp(- (theta_discrete - mu_theta2).^2 / (2 * sigma_theta2^2));

% 두 가우시안 분포를 합산하여 이봉형 감마 분포 생성
gamma_discrete_true = gamma1 + gamma2;

% 감마 분포를 최대값이 1이 되도록 정규화
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

%% Voltage Synthesis
R0 = 0.1;  % Ohmic resistance
OCV = 0;   % Open Circuit Voltage

rng(0);  % Ensure reproducibility of noise

for s = 1:num_scenarios
    fprintf('Processing Scenario %d/%d...\n', s, num_scenarios);
    
    % Current for the scenario
    ik = ik_scenarios(s, :);  % 로드된 전류 시나리오 사용
    
    %% Initialize Voltage
    V_est = zeros(1, length(t));  % Model voltage calculated via n-element model
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
    noise_level = 0.01;
    V_sd = V_est + noise_level * randn(size(V_est));  % V_sd = synthetic measured voltage
    
    % Store V_sd for the current scenario
    V_sd_all(s, :) = V_sd;  % Save the noisy V_sd for this scenario
end

%% 교차 검증을 통한 람다 최적화

% 테스트 시나리오를 [9 10]으로 고정
test_scenarios = [9, 10];
train_scenarios_full = setdiff(1:num_scenarios, test_scenarios);  % [1 2 3 4 5 6 7 8]

% 학습 시나리오에서 2개를 검증 세트로 선택하는 조합 생성
validation_indices = nchoosek(train_scenarios_full, 2);  % 8C2 = 28개의 조합
num_folds = size(validation_indices, 1);  % 28개의 폴드

cve_lambda = zeros(length(lambda_values), 1);  % 각 람다에 대한 CVE 저장

for l_idx = 1:length(lambda_values)
    lambda = lambda_values(l_idx);
    cve_total = 0;
    
    for fold = 1:num_folds
        % 검증 세트와 학습 세트 분리
        val_scenarios = validation_indices(fold, :);
        train_scenarios = setdiff(train_scenarios_full, val_scenarios);
        
        % 학습 데이터로 gamma 추정
        gamma_estimated = estimate_gamma(lambda, train_scenarios, ik_scenarios, V_sd_all, tau_discrete, delta_theta, L, OCV, R0, dt);
        
        % 검증 데이터로 전압 예측 및 에러 계산
        error_fold = calculate_error(gamma_estimated, val_scenarios, ik_scenarios, V_sd_all, tau_discrete, delta_theta, OCV, R0, dt);
        
        % 폴드의 에러 합산
        cve_total = cve_total + error_fold;
    end
    
    % 평균 CVE 계산
    cve_lambda(l_idx) = cve_total / num_folds;
    fprintf('Lambda %e, CVE: %f\n', lambda, cve_lambda(l_idx));
end

%% CVE vs 람다 그래프 그리기
figure;
semilogx(lambda_values, cve_lambda, 'b-', 'LineWidth', 1.5); % CVE vs Lambda plot
hold on;

% 최적 \(\lambda\) 포인트 찾기
[~, min_idx] = min(cve_lambda);
optimal_lambda = lambda_values(min_idx);

% 최적 \(\lambda\) 포인트 표시
semilogx(optimal_lambda, cve_lambda(min_idx), 'ro', 'MarkerSize', 10, 'LineWidth', 2);

% 최적 \(\lambda\) 텍스트 추가
optimal_lambda_str = ['Optimal \lambda = ', num2str(optimal_lambda, '%.2e')];
ylim([1.98322, 1.98324]);

% 레이블 및 제목
xlabel('\lambda (Regularization Parameter)');
ylabel('Cross-Validation Error (CVE)');
title('CVE vs. \lambda (Bimodal Case)');

% 그리드 및 범례
grid on;
% set(gca, 'YScale', 'log');  % Y축 로그 스케일 설정 (필요에 따라 사용)
% ylim([1.9830, 1.9835]); % Y축 한계 조정 (데이터에 맞게 조정 가능)
legend({'CVE', optimal_lambda_str}, 'Location', 'best');

hold off;

%% 최적의 람다로 전체 학습 데이터로 gamma 추정
gamma_optimal = estimate_gamma(optimal_lambda, train_scenarios_full, ik_scenarios, V_sd_all, tau_discrete, delta_theta, L, OCV, R0, dt);

%% 테스트 데이터로 전압 예측 및 에러 계산
test_error = calculate_error(gamma_optimal, test_scenarios, ik_scenarios, V_sd_all, tau_discrete, delta_theta, OCV, R0, dt);
disp(['Test Error with optimal lambda: ', num2str(test_error)]);

%% 결과 플롯
% True gamma와 추정된 gamma 비교
figure;
hold on;
plot(theta_discrete, gamma_discrete_true, 'k-', 'LineWidth', 1.5, 'DisplayName', 'True \gamma');
plot(theta_discrete, gamma_optimal, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Estimated \gamma');
xlabel('\theta = ln(\tau)');
ylabel('\gamma');
title(['True vs. Estimated \gamma with Optimal \lambda = ', num2str(optimal_lambda)]);
legend('Location', 'Best');
grid on;
hold off;

% 테스트 시나리오의 전압 비교
for s = test_scenarios
    ik = ik_scenarios(s, :);
    V_predicted = predict_voltage(gamma_optimal, ik, tau_discrete, delta_theta, OCV, R0, dt);
    V_actual = V_sd_all(s, :);
    
    figure;
    plot(t, V_actual, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Actual Voltage');
    hold on;
    plot(t, V_predicted, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Predicted Voltage');
    xlabel('Time (s)');
    ylabel('Voltage (V)');
    title(['Voltage Comparison for Test Scenario ', num2str(s)]);
    legend('Location', 'Best');
    grid on;
    hold off;
end

%% 함수 정의

% Gamma 추정 함수
function [gamma_estimated] = estimate_gamma(lambda, train_scenarios, ik_scenarios, V_sd_all, tau_discrete, delta_theta, L, OCV, R0, dt)
    % 학습 시나리오에서 W와 y_adjusted를 누적하여 구성
    W_total = [];
    y_total = [];
    
    for s = train_scenarios
        ik = ik_scenarios(s, :);
        V_sd = V_sd_all(s, :)';
        
        W_s = compute_W(ik, tau_discrete, delta_theta, dt);
        y_s = V_sd - OCV - R0 * ik';
        
        % 누적
        W_total = [W_total; W_s];
        y_total = [y_total; y_s];
    end
    
    % 정규화된 최소자승법으로 gamma 추정
    gamma_estimated = (W_total' * W_total + lambda * (L' * L)) \ (W_total' * y_total);
    
end

% 에러 계산 함수
function error_total = calculate_error(gamma_estimated, val_scenarios, ik_scenarios, V_sd_all, tau_discrete, delta_theta, OCV, R0, dt)
    error_total = 0;
    
    for s = val_scenarios
        ik = ik_scenarios(s, :);
        V_predicted = predict_voltage(gamma_estimated, ik, tau_discrete, delta_theta, OCV, R0, dt);
        V_actual = V_sd_all(s, :);
        
        % 전압 차이의 제곱 합산
        error_total = error_total + sum((V_predicted - V_actual).^2);
    end
end

% W 행렬 계산 함수
function W = compute_W(ik, tau_discrete, delta_theta, dt)
    n = length(tau_discrete);
    len_t = length(ik);
    W = zeros(len_t, n);  % Initialize W matrix
    
    for k_idx = 1:len_t
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
end

% 전압 예측 함수
function V_predicted = predict_voltage(gamma_estimated, ik, tau_discrete, delta_theta, OCV, R0, dt)
    len_t = length(ik);
    n = length(gamma_estimated);
    V_predicted = zeros(1, len_t);
    V_RC = zeros(n, len_t);
    
    for k_idx = 1:len_t
        if k_idx == 1
            for i = 1:n
                V_RC(i, k_idx) = gamma_estimated(i) * ik(k_idx) * (1 - exp(-dt / tau_discrete(i))) * delta_theta;
            end
        else
            for i = 1:n
                V_RC(i, k_idx) = V_RC(i, k_idx-1) * exp(-dt / tau_discrete(i)) + ...
                                 gamma_estimated(i) * ik(k_idx) * (1 - exp(-dt / tau_discrete(i))) * delta_theta;
            end
        end
        V_predicted(k_idx) = OCV + R0 * ik(k_idx) + sum(V_RC(:, k_idx));
    end
end
