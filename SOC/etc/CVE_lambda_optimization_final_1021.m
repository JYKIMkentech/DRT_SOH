clc; clear; close all;

%% AS1.mat 파일 로드
% 첫 번째 코드에서 저장한 A, T, ik_scenarios, t 변수를 불러옵니다.
load('AS1.mat');  % A: Amplitudes, T: Periods, ik_scenarios: Current Scenarios, t: Time Vector

%% Parameters
n = 40;                      % RC 요소의 수
dt = t(2) - t(1);            % 시간 간격
num_scenarios = 10;          % 총 시나리오 수
test_scenarios = [9,10];     % 테스트 세트
candidate_lambdas = logspace(-4, 4, 50); % \(\lambda\) 후보 (1e-4부터 1e4까지, 50개)

%% Generate Validation Folds: All possible combinations of 2 out of 8 (C(8,2)=28)
train_validation_scenarios = 1:8; % 시나리오 1-8을 교차 검증에 사용
validation_combinations = nchoosek(train_validation_scenarios, 2); % 28개의 검증 세트 생성
num_folds = size(validation_combinations, 1); % 28
validation_folds = mat2cell(validation_combinations, ones(num_folds,1), 2); % 셀 배열로 변환

% 확인: C(8,2) = 28
if num_folds ~= 28
    error('검증 폴드 수가 예상과 다릅니다. C(8,2) = 28이어야 합니다.');
end

%% True DRT Parameters [theta, gamma]
mu_theta = log(10);  % 평균 ln(τ)
sigma_theta = 1;     % ln(τ)의 표준편차

% θ (ln(τ)) 값 설정
theta_min = mu_theta - 3*sigma_theta;
theta_max = mu_theta + 3*sigma_theta;
theta_discrete = linspace(theta_min, theta_max, n);  % ln(τ)를 균등 간격으로 분할

% τ 값은 θ의 지수 함수
tau_discrete = exp(theta_discrete);  % τ = exp(θ)

% Δθ 계산
delta_theta = theta_discrete(2) - theta_discrete(1);  % Δθ = θ_{n+1} - θ_n

%% Regularization Matrix L (First-order difference, no scaling)
% 차분 행렬 D 생성
D = zeros(n-1, n);
for i = 1:n-1
    D(i, i) = -1;
    D(i, i+1) = 1;
end
L = D;  % 스케일링 적용하지 않음

%% True DRT [theta, gamma]
g_discrete_true = normpdf(theta_discrete, mu_theta, sigma_theta);
gamma_discrete_true = g_discrete_true;  % γ(θ) = g(θ)
gamma_discrete_true = gamma_discrete_true / max(gamma_discrete_true);  % 최대값으로 정규화

%% Generate Voltage Data for Each Scenario
V_est_all = zeros(num_scenarios, length(t)); % 추정 전압 저장
V_sd_all = cell(num_scenarios, 1);          % 노이즈가 추가된 전압 저장
W_all = cell(num_scenarios,1);               % W 행렬 저장

rng(0); % 동일한 노이즈 패턴 적용
noise_level = 0.01; % 노이즈 수준

for s = 1:num_scenarios
    fprintf('시나리오 %d/%d 데이터 생성 중...\n', s, num_scenarios);
    
    ik = ik_scenarios(s, :); % 첫 번째 코드에서 로드한 전류 시나리오 사용
    
    %% 전압 초기화
    V_est = zeros(1, length(t)); % 추정 전압
    R0 = 0.1; % 오믹 저항
    OCV = 0;  % 개방 회로 전압
    V_RC = zeros(n, length(t)); % 각 RC 요소의 전압
    
    %% 초기 전압 계산 (k=1)
    for i = 1:n
        V_RC(i, 1) = gamma_discrete_true(i) * delta_theta * ik(1) * (1 - exp(-dt / tau_discrete(i)));
    end
    V_est(1) = OCV + R0 * ik(1) + sum(V_RC(:, 1));
    
    %% 시간 t > 1에 대한 전압 계산
    for k_idx = 2:length(t)
        for i = 1:n
            V_RC(i, k_idx) = V_RC(i, k_idx-1) * exp(-dt / tau_discrete(i)) + ...
                              gamma_discrete_true(i) * delta_theta * ik(k_idx) * (1 - exp(-dt / tau_discrete(i)));
        end
        V_est(k_idx) = OCV + R0 * ik(k_idx) + sum(V_RC(:, k_idx));
    end
    
    % 추정 전압 저장 및 노이즈 추가
    V_est_all(s, :) = V_est;
    V_sd = V_est + noise_level * randn(size(V_est));
    V_sd_all{s} = V_sd;
    
    %% 시나리오별 W 행렬 구성
    W = zeros(length(t), n); % W 행렬 초기화
    for k_idx = 1:length(t)
        for i = 1:n
            if k_idx == 1
                W(k_idx, i) = ik(k_idx) * (1 - exp(-dt / tau_discrete(i))) * delta_theta;
            else
                W(k_idx, i) = W(k_idx-1, i) * exp(-dt / tau_discrete(i)) + ...
                              ik(k_idx) * (1 - exp(-dt / tau_discrete(i))) * delta_theta;
            end
        end
    end
    W_all{s} = W; % W 행렬 저장
end

%% Cross-Validation to Optimize Lambda
CVE = zeros(length(candidate_lambdas),1); % 각 \(\lambda\)에 대한 교차 검증 오류 초기화

for l = 1:length(candidate_lambdas)
    current_lambda = candidate_lambdas(l); % 현재 \(\lambda\)
    total_error = 0; % 총 교차 검증 오류 초기화
    
    for fold = 1:num_folds
        val_set = validation_folds{fold}; % 현재 검증 세트
        train_set = setdiff(train_validation_scenarios, val_set); % 훈련 세트 (6개)
        
        % 훈련 세트에서 W_train과 y_train 구성
        W_train = [];
        y_train = [];
        
        for s = train_set
            W_train = [W_train; W_all{s}];
            y_train = [y_train; V_sd_all{s}' - 0.1 * ik_scenarios(s, :)' - 0]; % OCV=0
        end
        
        %% 정규화된 최소 제곱 해 계산
        gamma_analytical = (W_train' * W_train + current_lambda * (L' * L)) \ (W_train' * y_train);
        gamma_analytical(gamma_analytical < 0) = 0; % 음수 값 제거
        
        %% 검증 세트에서 오류 계산
        for s_val = val_set
            W_val = W_all{s_val};
            V_val = V_sd_all{s_val};
            ik_val = ik_scenarios(s_val, :)';
            y_val = V_val' - 0.1 * ik_val - 0; % OCV=0
            
            % 예측 전압 계산
            V_pred = W_val * gamma_analytical + 0.1 * ik_val + 0;
            
            % 제곱 오차 합산
            error = sum((V_val' - V_pred).^2);
            total_error = total_error + error;
        end
    end
    
    % 현재 \(\lambda\)에 대한 CVE 저장
    CVE(l) = total_error;
end

%% Find Optimal Lambda
[~, optimal_idx] = min(CVE);
optimal_lambda = candidate_lambdas(optimal_idx);
fprintf('최적 \lambda: %e\n', optimal_lambda);

%% Plot CVE vs Lambda with Logarithmic Scale and Aggressive Y-axis Zoom
figure;
semilogx(candidate_lambdas, CVE, 'b-', 'LineWidth', 1.5); % CVE vs Lambda plot
hold on;

% 최적 \(\lambda\) 포인트 표시
semilogx(optimal_lambda, CVE(optimal_idx), 'ro', 'MarkerSize', 10, 'LineWidth', 2);

% 최적 \(\lambda\) 텍스트 추가
optimal_lambda_str = ['최적 \lambda = ', num2str(optimal_lambda, '%.2e')];

% 레이블 및 제목
xlabel('\lambda (정규화 파라미터)');
ylabel('교차 검증 오류 (CVE)');
title('로그 스케일 \lambda 에 따른 CVE 그래프');

% 그리드 및 범례
grid on;
set(gca, 'YScale', 'log');  % Y축 로그 스케일 설정
ylim([min(CVE)/10, max(CVE)*10]); % Y축 한계 조정 (데이터에 맞게 조정 가능)
legend({'CVE', optimal_lambda_str}, 'Location', 'best');
hold off;

%% Retrain on All Training Data with Optimal Lambda and Evaluate on Test Set
% 훈련 세트: 시나리오 [1-8] 중 테스트 세트 [9,10] 제외
train_set_final = setdiff(train_validation_scenarios, test_scenarios); % 시나리오 [1-8]
sum_WtW_final = zeros(n, n);   % W^T * W 합 초기화
sum_WtV_final = zeros(n, 1);   % W^T * y 합 초기화

for s = train_set_final
    W_s = W_all{s};
    V_s = V_sd_all{s};
    ik_s = ik_scenarios(s, :)';
    y_s = V_s' - 0.1 * ik_s - 0;  % OCV=0
    sum_WtW_final = sum_WtW_final + W_s' * W_s;
    sum_WtV_final = sum_WtV_final + W_s' * y_s;
end

% 정규화 항
regularization_term = optimal_lambda * (L' * L);

% 최적 \(\gamma\) 계산
gamma_final = (sum_WtW_final + regularization_term) \ sum_WtV_final;
%gamma_final(gamma_final < 0) = 0; % 음수 값 제거

%% Test on Test Scenarios [9,10]
test_set = test_scenarios;
total_test_error = 0;

for s_test = test_set
    W_test = W_all{s_test};
    V_test = V_sd_all{s_test};
    ik_test = ik_scenarios(s_test, :)';
    y_test = V_test' - 0.1 * ik_test - 0; % OCV=0
    
    % 예측 전압 계산
    V_pred_test = W_test * gamma_final + 0.1 * ik_test + 0;
    
    % 제곱 오차 합산
    error_test = sum((V_test' - V_pred_test).^2);
    total_test_error = total_test_error + error_test;
end

fprintf('테스트 세트에서의 총 제곱 전압 오류: %f\n', total_test_error);

%% Plot DRT Comparison: True vs Optimized Gamma
figure;
plot(theta_discrete, gamma_discrete_true, 'k-', 'LineWidth', 2, 'DisplayName', 'True \gamma(\theta)');
hold on;
plot(theta_discrete, gamma_final, 'r--', 'LineWidth', 2, 'DisplayName', ['Optimized \gamma(\theta) (\lambda = ', num2str(optimal_lambda, '%.2e'), ')']);
xlabel('ln(\tau) = \theta');
ylabel('\gamma(\theta)');
title('True DRT와 최적화된 DRT 비교');
legend('Location', 'Best');
grid on;
hold off;

% 그래프 레이아웃 정규화 설정
set(gcf, 'Units', 'normalized', 'Position', [0.1, 0.1, 0.8, 0.8]);  % 화면 비율로 설정

