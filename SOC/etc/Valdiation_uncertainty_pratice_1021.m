clc; clear; close all;

%% 2. DRT 추정 및 교차 검증

% Load the generated AS1.mat 파일
load('AS1.mat');  % Load A, T, ik_scenarios, t variables

%% Parameters 
n = 40;  % Number of discrete elements
dt = t(2) - t(1);  % Time step based on loaded time vector
num_scenarios = 10;  % Number of current scenarios
lambda_values = logspace(-4, 9, 50);  % 람다 값 범위 설정

%% DRT 설정

% Theta = ln(tau) (x축)
% gamma(theta) = [ R(exp(theta)) * exp(theta) ] = [ R(tau) * tau ] (y축)

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
% cve_fold = zeros(length(lambda_values), num_folds);  % 각 람다 및 폴드에 대한 CVE 저장 (삭제)
gamma_estimates_all = cell(length(lambda_values), num_folds);  % 각 람다 및 폴드에 대한 gamma 저장

for l_idx = 1:length(lambda_values)
    lambda = lambda_values(l_idx);
    cve_total = 0;
    
    for fold = 1:num_folds
        % 검증 세트와 학습 세트 분리
        val_scenarios = validation_indices(fold, :);
        train_scenarios = setdiff(train_scenarios_full, val_scenarios);
        
        % 학습 데이터로 gamma 추정
        gamma_estimated = estimate_gamma(lambda, train_scenarios, ik_scenarios, V_sd_all, tau_discrete, delta_theta, L, OCV, R0, dt);
        
        % γ 추정치 저장
        gamma_estimates_all{l_idx, fold} = gamma_estimated;
        
        % 검증 데이터로 전압 예측 및 에러 계산
        error_fold = calculate_error(gamma_estimated, val_scenarios, ik_scenarios, V_sd_all, tau_discrete, delta_theta, OCV, R0, dt);
        
        % 폴드의 에러 합산
        cve_total = cve_total + error_fold;
    end
    
    % 평균 CVE 계산
    cve_lambda(l_idx) = cve_total / num_folds;
    fprintf('Lambda %e, CVE: %f\n', lambda, cve_lambda(l_idx));
end

% % CVE의 표준 편차 계산 (삭제)
% cve_std = std(cve_fold, 0, 2);  % 각 람다에 대한 표준 편차

%% CVE vs 람다 그래프 그리기
figure;
plot(lambda_values, cve_lambda, 'b-', 'LineWidth', 1.5); % 에러바 제거
hold on;

% 최적 \(\lambda\) 포인트 찾기
[~, min_idx] = min(cve_lambda);
optimal_lambda = lambda_values(min_idx);

% 최적 \(\lambda\) 포인트 표시
semilogx(optimal_lambda, cve_lambda(min_idx), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
%ylim([1.983*28 1.9835*28]);

% 레이블 및 제목
xlabel('\lambda (정규화 파라미터)');
ylabel('교차 검증 오류 (CVE)');
title('로그 스케일 \lambda 에 따른 CVE 그래프');

% 그리드 및 범례
grid on;
set(gca, 'XScale', 'log');  % X축 로그 스케일 설정
legend({'CVE', '최적 \lambda'}, 'Location', 'best');

hold off;

%% 최적의 람다에 대한 γ의 평균과 95% 신뢰 구간 계산
num_optimal_folds = num_folds;
optimal_gamma_estimates = zeros(n, num_optimal_folds);

for fold = 1:num_optimal_folds
    optimal_gamma_estimates(:, fold) = gamma_estimates_all{min_idx, fold}; % 최적의 람다 인덱스에 해당하는 28개 fold - (세타-gamma) 가져오기 
end

% γ의 평균 및 표준 편차 계산
gamma_mean = mean(optimal_gamma_estimates, 2);
gamma_std = std(optimal_gamma_estimates, 0, 2);

% 95% 신뢰 구간 계산
gamma_se = gamma_std / sqrt(num_optimal_folds);  % 표준 오차
t_value = tinv(0.975, num_optimal_folds - 1);    % 95% 신뢰 구간에 대한 t-값
gamma_ci_lower = gamma_mean - t_value * gamma_se;
gamma_ci_upper = gamma_mean + t_value * gamma_se;

%% 최적의 람다로 전체 학습 데이터로 gamma 추정
gamma_optimal = estimate_gamma(optimal_lambda, train_scenarios_full, ik_scenarios, V_sd_all, tau_discrete, delta_theta, L, OCV, R0, dt);

%% 테스트 데이터로 전압 예측 및 에러 계산
test_error = calculate_error(gamma_optimal, test_scenarios, ik_scenarios, V_sd_all, tau_discrete, delta_theta, OCV, R0, dt);
disp(['Test Error with optimal lambda: ', num2str(test_error)]);

%% 결과 플롯 (γ의 평균 및 95% 신뢰 구간)
figure;
hold on;
plot(theta_discrete, gamma_discrete_true, 'k-', 'LineWidth', 1.5, 'DisplayName', 'True \gamma');
errorbar(theta_discrete, gamma_mean, t_value * gamma_se, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Estimated \gamma with 95% CI');
xlabel('\theta = ln(\tau)');
ylabel('\gamma');
title(['True vs. Estimated \gamma with Optimal \lambda = ', num2str(optimal_lambda)]);
legend('Location', 'Best');
grid on;
hold off;

%% 테스트 시나리오의 전압 비교
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
