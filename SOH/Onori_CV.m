clear; clc; close all;

%% 1. trips 데이터 로드
% trips 데이터를 로드합니다.
load('processed_10_trips_cycle14.mat', 'trips');  % 'trips' 구조체를 로드합니다.

%% 2. SOC-OCV 데이터 로드
% RPT_10_soc_ocv_cap.mat 파일에서 soc_ocv_cap 데이터를 로드합니다.
load('RPT_10_soc_ocv_cap.mat', 'soc_ocv_cap');

% SOC와 OCV 값 추출
soc_values = soc_ocv_cap(:, 1);  % SOC 값 (0 ~ 1)
ocv_values = soc_ocv_cap(:, 2);  % OCV 값 (V)

% 배터리 용량 추출 (Capacity 열의 최대값)
Q_batt = max(soc_ocv_cap(:, 3));  % Ah 단위

%% 3. DRT 추정에 필요한 파라미터 설정
n = 401;  % 이산 요소의 개수
tau_min = 0.1;     % 최소 시간 상수 (초)
tau_max = 2600;    % 최대 시간 상수 (초)

% Theta 및 tau 값 계산
theta_min = log(tau_min);
theta_max = log(tau_max);
theta_discrete = linspace(theta_min, theta_max, n);
tau_discrete = exp(theta_discrete);

% Delta theta 계산
delta_theta = theta_discrete(2) - theta_discrete(1);

% 정규화 파라미터 람다 값 범위 설정 (로그 스케일로 10개의 값)
lambda_values = logspace(-4, 1, 50);  % 필요에 따라 조정 가능

% Gamma에 대한 1차 차분 행렬 L_gamma 생성
L_gamma = zeros(n-1, n);
for i = 1:n-1
    L_gamma(i, i) = -1;
    L_gamma(i, i+1) = 1;
end

% R0에 대한 정규화를 피하기 위해 L_aug 생성
L_aug = [L_gamma, zeros(n-1, 1)];

%% 4. 각 트립의 데이터 준비
num_trips = length(trips);

% 트립별 데이터 저장을 위한 셀 배열 초기화
t_all = cell(num_trips, 1);
ik_all = cell(num_trips, 1);
V_sd_all = cell(num_trips, 1);
SOC_all = cell(num_trips, 1);

SOC0 = 0.8;  % 전체 시뮬레이션의 초기 SOC 설정

for s = 1:num_trips
    % 현재 트립의 데이터 추출
    t = trips(s).time_reset;
    ik = trips(s).I;
    V_sd = trips(s).V;
    
    % 시간 간격 계산
    delta_t = [0; diff(t)];  % 시간 간격 계산 (초)
  
    delta_t(1) = delta_t(2);  % 첫 번째 값이 0이면 두 번째 값으로 대체
  
    
    % SOC 계산 (전류 적분)
    Ah_consumed = cumsum(ik .* delta_t) / 3600;  % Ah로 변환
    SOC = SOC0 + Ah_consumed / Q_batt;  % SOC 계산
    
    % 데이터 저장
    t_all{s} = t;
    ik_all{s} = ik;
    V_sd_all{s} = V_sd;
    SOC_all{s} = SOC;
    
    % 다음 트립을 위한 SOC0 업데이트
    SOC0 = SOC(end);
end


%% 5. 교차 검증을 통한 람다 최적화

% K-폴드 교차 검증 (K는 트립 수)
K = num_trips;

% 각 람다에 대한 CVE 저장
cve_lambda = zeros(length(lambda_values), 1);

for l_idx = 1:length(lambda_values)
    lambda = lambda_values(l_idx);
    cve_total = 0;
    
    fprintf('Processing lambda %e (%d/%d)...\n', lambda, l_idx, length(lambda_values));
    
    for fold = 1:K
        % 검증 세트와 학습 세트 분리
        val_indices = fold;
        train_indices = setdiff(1:num_trips, val_indices);
        
        % 학습 데이터로 gamma 및 R0 추정
        [gamma_estimated, R0_estimated] = estimate_gamma(lambda, train_indices, ik_all, V_sd_all, SOC_all, soc_values, ocv_values, tau_discrete, delta_theta, L_aug);
        
        % 검증 데이터로 전압 예측 및 에러 계산
        error_fold = calculate_error(gamma_estimated, R0_estimated, val_indices, ik_all, V_sd_all, SOC_all, soc_values, ocv_values, tau_discrete, delta_theta);
        
        % 폴드의 에러 합산
        cve_total = cve_total + error_fold;
    end
    
    % 평균 CVE 계산
    cve_lambda(l_idx) = cve_total / K;
    fprintf('Lambda %e, CVE: %f\n', lambda, cve_lambda(l_idx));
end

%% 6. CVE vs 람다 그래프 그리기
figure;
semilogx(lambda_values, cve_lambda, 'b-', 'LineWidth', 1.5); % CVE vs Lambda plot
hold on;

% 최적 람다 포인트 찾기
[~, min_idx] = min(cve_lambda);
optimal_lambda = lambda_values(min_idx);

% 최적 람다 포인트 표시
semilogx(optimal_lambda, cve_lambda(min_idx), 'ro', 'MarkerSize', 10, 'LineWidth', 2);

% 최적 람다 텍스트 추가
optimal_lambda_str = ['Optimal \lambda = ', num2str(optimal_lambda, '%.2e')];
%ylim([2.31, 10]);

% 레이블 및 제목
xlabel('\lambda (Regularization Parameter)', 'FontSize', 14);
ylabel('Cross-Validation Error (CVE)', 'FontSize', 14);
title('CVE vs. \lambda', 'FontSize', 16);

% 그리드 및 범례
grid on;
legend({'CVE', optimal_lambda_str}, 'Location', 'best', 'FontSize', 12);

hold off;

%% 7. 최적의 람다로 전체 데이터로 gamma 및 R0 추정
[gamma_optimal, R0_optimal] = estimate_gamma(optimal_lambda, 1:num_trips, ik_all, V_sd_all, SOC_all, soc_values, ocv_values, tau_discrete, delta_theta, L_aug);

% 최종 결과를 저장하거나 추가 분석을 수행할 수 있습니다.

%% 8. 함수 정의

% Gamma 및 R0 추정 함수
function [gamma_estimated, R0_estimated] = estimate_gamma(lambda, train_indices, ik_all, V_sd_all, SOC_all, soc_values, ocv_values, tau_discrete, delta_theta, L_aug)
    n = length(tau_discrete);
    num_train = length(train_indices);
    
    gamma_estimated = zeros(n, num_train);
    R0_estimated = zeros(num_train, 1);
    
    for idx = 1:num_train
        s = train_indices(idx);
        ik = ik_all{s};
        V_sd = V_sd_all{s};
        SOC = SOC_all{s};
        
        % 시간 벡터
        t = (0:length(ik)-1)';  % 가정: dt가 일정하거나 필요에 따라 수정
        
        % 시간 간격 dt 계산
        delta_t = [0; diff(t)];
        if length(delta_t) > 1
            delta_t(1) = delta_t(2);  % 첫 번째 값이 0이면 두 번째 값으로 대체
        else
            delta_t(1) = 0.1;  % 데이터가 하나뿐인 경우 예외 처리
        end
        
        % OCV 계산
        ocv_over_time = interp1(soc_values, ocv_values, SOC, 'linear', 'extrap');
        
        % W 행렬 생성
        W = compute_W(ik, tau_discrete, delta_theta, delta_t);
        
        % R0 추정을 위한 W 행렬 확장
        W_aug = [W, ik(:)];  % ik(:)는 ik를 열 벡터로 변환
        
        % y 벡터 생성
        y = V_sd - ocv_over_time;
        y = y(:);  % y를 열 벡터로 변환
        
        % 비용 함수: 0.5 * Theta' * H * Theta + f' * Theta
        H = (W_aug' * W_aug + lambda * (L_aug' * L_aug));
        f = -W_aug' * y;
        
        % 제약 조건: Theta >= 0 (gamma와 R0는 0 이상)
        A = -eye(n+1);
        b = zeros(n+1, 1);
        
        % quadprog 옵션 설정
        options = optimoptions('quadprog', 'Display', 'off', 'Algorithm', 'interior-point-convex');
        
        % quadprog 실행
        [Theta_est, ~, exitflag] = quadprog(H, f, A, b, [], [], [], [], [], options);
        
        if exitflag ~= 1
            warning('Optimization did not converge for trip %d.', s);
        end
        
        % gamma와 R0 추정값 추출
        gamma_est = Theta_est(1:n);
        R0_est = Theta_est(n+1);
        
        % 추정값 저장
        gamma_estimated(:, idx) = gamma_est;
        R0_estimated(idx) = R0_est;
    end
end

% 에러 계산 함수
function error_total = calculate_error(gamma_estimated, R0_estimated, val_indices, ik_all, V_sd_all, SOC_all, soc_values, ocv_values, tau_discrete, delta_theta)
    error_total = 0;
    num_val = length(val_indices);
    
    for idx = 1:num_val
        s = val_indices(idx);
        ik = ik_all{s};
        V_sd = V_sd_all{s};
        SOC = SOC_all{s};
        
        % OCV 계산
        ocv_over_time = interp1(soc_values, ocv_values, SOC, 'linear', 'extrap');
        
        % 추정된 전압 계산
        gamma_est = mean(gamma_estimated, 2);  % 학습 세트의 평균 gamma 사용
        R0_est = mean(R0_estimated);          % 학습 세트의 평균 R0 사용
        V_est = predict_voltage(gamma_est, R0_est, ik, SOC, soc_values, ocv_values, tau_discrete, delta_theta);
        
        % V_sd와 V_est를 열 벡터로 변환 (필요한 경우)
        V_sd = V_sd(:);
        V_est = V_est(:);
        
        % 전압 차이의 제곱 합산
        error_total = error_total + sum((V_est - V_sd).^2);
    end
end

% W 행렬 계산 함수
function W = compute_W(ik, tau_discrete, delta_theta, dt)
    n = length(tau_discrete);
    len_t = length(ik);
    W = zeros(len_t, n);  % W 행렬 초기화
    
    for k_idx = 1:len_t
        if k_idx == 1
            for i = 1:n
                W(k_idx, i) = ik(k_idx) * (1 - exp(-dt(k_idx) / tau_discrete(i))) * delta_theta;
            end
        else
            for i = 1:n
                W(k_idx, i) = W(k_idx-1, i) * exp(-dt(k_idx) / tau_discrete(i)) + ...
                              ik(k_idx) * (1 - exp(-dt(k_idx) / tau_discrete(i))) * delta_theta;
            end
        end
    end
end

% 전압 예측 함수
function V_est = predict_voltage(gamma_est, R0_est, ik, SOC, soc_values, ocv_values, tau_discrete, delta_theta)
    len_t = length(ik);
    n = length(gamma_est);
    V_RC = zeros(n, len_t);
    V_est = zeros(len_t, 1);
    
    % 시간 벡터
    t = (0:len_t-1)';  % 가정: dt가 일정하거나 필요에 따라 수정
    
    % 시간 간격 dt 계산
    delta_t = [0; diff(t)];
    if length(delta_t) > 1
        delta_t(1) = delta_t(2);  % 첫 번째 값이 0이면 두 번째 값으로 대체
    else
        delta_t(1) = 0.1;  % 데이터가 하나뿐인 경우 예외 처리
    end
    
    % OCV 계산
    ocv_over_time = interp1(soc_values, ocv_values, SOC, 'linear', 'extrap');
    
    for k_idx = 1:len_t
        if k_idx == 1
            for i = 1:n
                V_RC(i, k_idx) = gamma_est(i) * ik(k_idx) * (1 - exp(-delta_t(k_idx) / tau_discrete(i))) * delta_theta;
            end
        else
            for i = 1:n
                V_RC(i, k_idx) = V_RC(i, k_idx-1) * exp(-delta_t(k_idx) / tau_discrete(i)) + ...
                                 gamma_est(i) * ik(k_idx) * (1 - exp(-delta_t(k_idx) / tau_discrete(i))) * delta_theta;
            end
        end
        V_est(k_idx) = ocv_over_time(k_idx) + R0_est * ik(k_idx) + sum(V_RC(:, k_idx));
    end
end
