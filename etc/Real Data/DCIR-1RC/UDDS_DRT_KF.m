clc; clear; close all;

%% 1. 저장된 데이터 로드
% gamma_data.mat 및 soc_ocv_data.mat 파일 로드
load('gamma_data.mat', 'gamma_sorted', 'soc_sorted', 'theta_discrete', 'R0_est_all', 'soc_mid_all');
load('soc_ocv_data.mat', 'soc_values', 'ocv_values');

% tau_discrete 재정의
tau_discrete = exp(theta_discrete);

%% 2. 주행 데이터 로드
load('udds_data.mat');  % 'udds_data' 구조체를 로드합니다.

%% 3. 칼만 필터를 적용할 사이클 선택
cycle_idx = 1;  % 원하는 사이클 번호로 변경 가능
if cycle_idx > length(udds_data)
    error('선택한 사이클 번호가 데이터 범위를 벗어났습니다.');
end

% 선택한 사이클의 데이터 추출
time = udds_data(cycle_idx).t;
current_measured = udds_data(cycle_idx).I;  % 측정된 전류 (양수: 방전, 음수: 충전)
voltage_measured = udds_data(cycle_idx).V;  % 측정된 전압
SOC_true = udds_data(cycle_idx).SOC;  % 실제 SOC (있는 경우)

%% 4. 칼만 필터 설정
% 시간 간격 계산
dt = [0; diff(time)];
if dt(1) == 0
    dt(1) = dt(2);
end

% RC 소자의 개수
num_RC = length(theta_discrete);

% 초기화
num_samples = length(time);
voltage_estimated = zeros(num_samples, 1);
SOC_estimated = zeros(num_samples, 1);  % SOC 추정값 초기화

% 초기 SOC 설정 (실제 SOC가 없을 경우 0.9901로 설정)
if ~isempty(SOC_true)
    SOC_estimated(1) = SOC_true(1);
else
    SOC_estimated(1) = 0.9901;  % 초기 SOC를 99.01%로 가정
end

% 상태 벡터 차원
state_dimension = 1 + num_RC;  % 상태 벡터: [SOC; V_RC_1; V_RC_2; ... ; V_RC_n]

% 초기 상태 추정
X_est = zeros(state_dimension, 1);
X_est(1) = SOC_estimated(1);

% 초기 공분산 행렬 P0 설정
P = zeros(state_dimension);
P(1,1) = (0.0004)^2;  % SOC에 대한 초기 분산
% V_RC_1, V_RC_2, ..., V_RC_n에 대한 초기 분산은 모두 0

% 프로세스 잡음 공분산 행렬 Q 설정
Q = zeros(state_dimension);
Q(1,1) = 1; %.0e-14;  % SOC에 대한 프로세스 잡음 분산
for i = 2:state_dimension
    Q(i,i) = (0.0016)^2;  % 각 V_RC_i에 대한 프로세스 잡음 분산
end

% 측정 잡음 공분산 R 설정
R = 5.25e-16;  % 측정 잡음 분산

% 초기 V_RC_est 설정 (다중 RC 모델)
% 현재 SOC_estimated(1)에서 gamma_current_init 계산
gamma_current_init = interp1(soc_sorted, gamma_sorted, SOC_estimated(1), 'linear', 'extrap');  % (1 x num_RC)

% delta_theta 계산 (theta_discrete는 동일 간격 가정)
delta_theta = theta_discrete(2) - theta_discrete(1);  % 스칼라

% 각 RC 소자에 대한 R_i 초기값 계산
R_i_init = gamma_current_init * delta_theta;  % (1 x num_RC)

% 각 RC 소자에 대한 C_i 초기값 계산
C_i_init = tau_discrete ./ R_i_init;  % (1 x num_RC)

% 초기 V_RC_est 계산
V_RC_init = current_measured(1) * R_i_init .* (1 - exp(-dt(1) ./ (R_i_init .* C_i_init)));  % (1 x num_RC)

% 상태 추정치에 초기 V_RC_est 할당
X_est(2:end) = V_RC_init(:);  % 열 벡터로 변환하여 할당

%% 5. 칼만 필터 적용
for k = 1:num_samples-1
    % 현재 SOC에 대한 gamma 값 보간
    gamma_current = interp1(soc_sorted, gamma_sorted, SOC_estimated(k), 'linear', 'extrap');  % (1 x num_RC)
    
    % 각 RC 소자에 대한 R_i 계산
    R_i = gamma_current * delta_theta;  % (1 x num_RC)
    
    % 각 RC 소자에 대한 C_i 계산
    C_i = tau_discrete ./ R_i;  % (1 x num_RC)
    
    % 상태 전이 행렬 A 계산
    A = diag([1; exp(-dt(k) ./ tau_discrete(:))]);  % (1 + num_RC) x (1 + num_RC)
    
    % 예측 단계
    % SOC 예측 (쿨롱 카운팅)
    SOC_pred = X_est(1) + (dt(k) / 3600) * current_measured(k);  % 양의 전류는 방전을 의미
    
    % RC 전압 예측
    V_RC_pred = (exp(-dt(k) ./ (R_i .* C_i))') .* X_est(2:end) + (R_i' .* (1 - exp(-dt(k) ./ (R_i .* C_i))')) .* current_measured(k);
    
    % 상태 예측 값 결합
    X_pred = [SOC_pred; V_RC_pred];
    
    % 공분산 예측
    P_pred = A * P * A' + Q;
    
    % OCV 계산
    OCV_pred = interp1(soc_values, ocv_values, SOC_pred, 'linear', 'extrap');
    
    % 예측된 전압 계산
    V_pred = OCV_pred + R0_est_all(cycle_idx) * current_measured(k) + sum(V_RC_pred);
    voltage_estimated(k) = V_pred;
    
    % 관측 행렬 H 계산
    delta_SOC = 1e-5;
    OCV_plus = interp1(soc_values, ocv_values, SOC_pred + delta_SOC, 'linear', 'extrap');
    OCV_minus = interp1(soc_values, ocv_values, SOC_pred - delta_SOC, 'linear', 'extrap');
    dOCV_dSOC = (OCV_plus - OCV_minus) / (2 * delta_SOC);
    
    H = zeros(1, state_dimension);
    H(1) = dOCV_dSOC;  % SOC에 대한 OCV의 미분
    H(2:end) = 1;  % 각 RC 전압의 계수
    
    % 잔차 계산
    y_tilde = voltage_measured(k) - V_pred;
    
    % 칼만 이득 계산
    S = H * P_pred * H' + R;
    K = (P_pred * H') / S;
    
    % 상태 업데이트
    X_est = X_pred + K * y_tilde;
    
    % 공분산 업데이트
    P = P_pred - K * H * P_pred;
    
    % 상태 저장
    SOC_estimated(k+1) = X_est(1);
end

% 마지막 전압 추정
OCV_pred_final = interp1(soc_values, ocv_values, SOC_estimated(end), 'linear', 'extrap');
V_predicted_final = OCV_pred_final + R0_est_all(cycle_idx) * current_measured(end) + sum(X_est(2:end));
voltage_estimated(end) = V_predicted_final;

%% 6. 결과 시각화
figure;
subplot(3,1,1);
plot(time, SOC_true, 'b', 'LineWidth', 1.5); hold on;
plot(time, SOC_estimated, 'r--', 'LineWidth', 1.5);
xlabel('시간 (s)');
ylabel('SOC');
title('실제 SOC vs 추정 SOC');
legend('실제 SOC', '추정 SOC');
grid on;

subplot(3,1,2);
plot(time, voltage_measured, 'b', 'LineWidth', 1.5); hold on;
plot(time, voltage_estimated, 'r--', 'LineWidth', 1.5);
xlabel('시간 (s)');
ylabel('전압 (V)');
title('측정 전압 vs 추정 전압');
legend('측정 전압', '추정 전압');
grid on;

subplot(3,1,3);
plot(time, current_measured, 'k', 'LineWidth', 1.5);
xlabel('시간 (s)');
ylabel('전류 (A)');
title('전류 프로파일');
grid on;

%% 7. 추가 시각화 및 결과 저장 (필요 시)
% SOC 추정 오류 계산
if ~isempty(SOC_true)
    SOC_error = SOC_true - SOC_estimated;
    SOC_RMSE = sqrt(mean(SOC_error.^2));
    fprintf('SOC RMSE: %.4f%%\n', SOC_RMSE * 100);
    
    % SOC 추정 오류 시각화
    figure;
    plot(time, SOC_error, 'g', 'LineWidth', 1.5);
    xlabel('시간 (s)');
    ylabel('SOC 오류');
    title('SOC 추정 오류 시간에 따른 변화');
    legend(sprintf('SOC RMSE: %.4f%%', SOC_RMSE * 100));
    grid on;
end

% 결과 행렬 저장
result_matrix = [SOC_true, SOC_estimated];
column_names = {'True_SOC', 'Estimated_SOC'};
% save('kalman_filter_result.mat', 'result_matrix', 'column_names');

% 전압 잔차 계산 및 시각화
voltage_residuals = voltage_measured - voltage_estimated;
R_calculated = var(voltage_residuals);
fprintf('Calculated sensor noise covariance R: %.4e\n', R_calculated);

figure;
plot(time, voltage_residuals, 'k', 'LineWidth', 1.5);
xlabel('시간 (s)');
ylabel('잔차 (V)');
title('측정 전압과 추정 전압의 잔차');
grid on;

%%save

% DRT 기반 칼만 필터 코드에서 첫 번째 트립 처리 후에 추가

% SOC_estimated: 칼만 필터로 추정된 SOC (이미 코드에서 사용됨)
% time: 시간 벡터 (이미 코드에서 사용됨)

% 첫 번째 트립의 결과를 저장
DRT_SOC_est = SOC_estimated;  % 추정된 SOC
DRT_Time = time;              % 시간 벡터

% 결과 저장
save('DRT_SOC_trip1.mat', 'DRT_SOC_est', 'DRT_Time');






