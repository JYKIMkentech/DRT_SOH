clear; clc; close all;

%% 1. 저장된 데이터 로드
% gamma_data.mat 및 soc_ocv_data.mat 파일 로드
load('gamma_data.mat', 'gamma_sorted', 'soc_sorted', 'theta_discrete', 'R0_est_all', 'soc_mid_all');
load('soc_ocv_data.mat');  % 'soc_ocv_data'에 포함된 변수를 로드

% tau_discrete 재정의
tau_discrete = exp(theta_discrete);  % tau_discrete 재정의

% % OCV 데이터 로드 및 분리 (soc_ocv에서 추출)
% soc_values = soc_ocv_data(:, 1);  % 첫 번째 열이 SOC 값
% ocv_values = soc_ocv_data(:, 2);  % 두 번째 열이 OCV 값

%% 2. 주행 데이터 로드
load('udds_data.mat');  % 'udds_data' 구조체를 로드합니다.

%% 3. 칼만 필터를 적용할 사이클 선택
cycle_idx = 1;  % 원하는 사이클 번호로 변경 가능
if cycle_idx > length(udds_data)
    error('선택한 사이클 번호가 데이터 범위를 벗어났습니다.');
end

t = udds_data(cycle_idx).t;
I_meas = udds_data(cycle_idx).I;  % 측정된 전류 (양수: 방전, 음수: 충전)
V_meas = udds_data(cycle_idx).V;  % 측정된 전압
SOC_true = udds_data(cycle_idx).SOC;  % 실제 SOC (있는 경우)

%% 4. 칼만 필터 설정
% 시간 간격 계산
dt = [0; diff(t)];
if dt(1) == 0
    dt(1) = dt(2);
end

% RC 소자의 개수
n_RC = length(theta_discrete);

% 초기화
N = length(t);
V_est = zeros(N, 1);
SOC_est = zeros(N, 1);
x_est = zeros(n_RC, N);  % RC 전압들

% 초기 SOC 설정 (예: 100%)
SOC_est(1) = 0.9901;

% 초기 상태 변수
x_est(:, 1) = 0;

% 출력 행렬 C 정의
C = ones(1, n_RC);  % 1 x n_RC 행 벡터

% 칼만 필터를 위한 초기화
P = eye(n_RC) * 1e-6;  % 초기 오차 공분산 (n_RC x n_RC)
Q = eye(n_RC) * 1e-5;  % 프로세스 노이즈 공분산 (n_RC x n_RC)
R_noise = 1e-3;         % 측정 노이즈 공분산 (스칼라)

% 배터리 용량 (Ah) 설정 (예: 2.3Ah)
battery_capacity_Ah = 2.3;

%% 5. 칼만 필터 루프
for k = 2:N
    % 현재 SOC에 따른 gamma 및 R0 보간
    gamma_k = interp1(soc_sorted(:), gamma_sorted, SOC_est(k-1), 'linear', 'extrap');  % gamma 보간 (1 x n_RC)
    R0_k = interp1(soc_sorted, R0_est_all, SOC_est(k-1), 'linear', 'extrap');          % R0 보간 (스칼라)
    
    % 시간 간격에 따라 A_k, B_k 매트릭스 업데이트
    A_k = diag(exp(-dt(k) ./ tau_discrete));  % n_RC x n_RC
    B_k = gamma_k' .* (1 - exp(-dt(k) ./ tau_discrete));  % n_RC x 1
    
    % SOC에 따른 OCV 계산
    OCV_k = interp1(soc_values, ocv_values, SOC_est(k-1), 'linear', 'extrap');
    
    % 상태 예측
    x_pred = A_k * x_est(:, k-1) + B_k * I_meas(k-1);  % n_RC x 1
    P_pred = A_k * P * A_k' + Q;                       % n_RC x n_RC
    
    % 출력 예측
    V_pred = OCV_k + R0_k * I_meas(k-1) + C * x_pred;  % 스칼라
    
    % 혁신(innovation) 또는 잔차(residual)
    y_k = V_meas(k) - V_pred;  % 스칼라
    
    % 칼만 이득 계산
    S = C * P_pred * C' + R_noise;    % 스칼라
    K = (P_pred * C') / S;            % n_RC x 1
    
    % 상태 업데이트
    x_est(:, k) = x_pred + K * y_k;  % n_RC x 1
    
    % 공분산 업데이트
    P = (eye(n_RC) - K * C) * P_pred;  % n_RC x n_RC
    
    % SOC 업데이트 (Coulomb Counting)
    SOC_est(k) = SOC_est(k-1) - (I_meas(k-1) * dt(k)) / (3600 * battery_capacity_Ah);
    
    % SOC 범위 제한
    if SOC_est(k) > 1
        SOC_est(k) = 1;
    elseif SOC_est(k) < 0
        SOC_est(k) = 0;
    end
    
    % 추정된 전압 저장
    V_est(k) = OCV_k + R0_k * I_meas(k) + C * x_est(:, k);
    
end

%% 7. 결과 시각화
% SOC 추정 결과 시각화
figure;
plot(t, SOC_est, 'r', 'LineWidth', 1.5);
hold on;
if exist('SOC_true', 'var')
    plot(t, SOC_true, 'b--', 'LineWidth', 1.5);
    legend('Estimated SOC', 'True SOC');
else
    legend('Estimated SOC');
end
xlabel('Time (s)');
ylabel('SOC');
title('Estimated SOC over Time');
grid on;
hold off;

% 전압 비교
figure;
plot(t, V_meas, 'b', 'LineWidth', 1);
hold on;
plot(t, V_est, 'r--', 'LineWidth', 1);
xlabel('Time (s)');
ylabel('Voltage (V)');
title('Voltage Comparison');
legend('Measured Voltage', 'Estimated Voltage');
grid on;
hold off;

fprintf('code2 실행이 완료되었습니다. SOC 추정 결과와 전압 비교 그래프가 생성되었습니다.\n');
