
clc; clear; close all;

% UDDS 데이터 로드
load('C:\Users\deu04\OneDrive\바탕 화면\wykht8y7tg-1\Panasonic 18650PF Data\Panasonic 18650PF Data\25degC\Drive cycles\03-21-17_00.29 25degC_UDDS_Pan18650PF.mat');
udds_current = meas.Current; % UDDS 전류 데이터
udds_voltage = meas.Voltage; % UDDS 전압 데이터
udds_time = meas.Time; % UDDS 시간 데이터

% SOC-OCV 구조 로드
load('soc_ocv.mat');
soc_values = soc_ocv(:, 1);
ocv_values = soc_ocv(:, 2);

% 설정 파라미터
Config.dt = mean(diff(udds_time)); % 평균 시간 간격
Config.R0 = 0.001884314;
Config.R1 = 0.045801322;
Config.C1 = 4846.080679;
Config.cap = 2.99 ; % 명목 용량 [Ah]
Config.coulomb_efficient = 1;

% SOC 추정 초기화
SOC_est = zeros(length(udds_current), 1);
SOC_est(1) = 1; % 초기 SOC는 100%로 가정

% 진정한 SOC 초기화 (쿨롱 카운팅을 사용하여)
true_SOC = zeros(length(udds_current), 1);
true_SOC(1) = 1; % 초기 SOC는 100%로 가정

% EKF 초기화
P = eye(2); % 초기 추정 오차 공분산

% V1 및 Vt 배열 사전 할당
V1_est = zeros(length(udds_current), 1);
Vt_est = zeros(length(udds_current), 1);
V1_est(1) = udds_current(1) * Config.R1 * (1 - exp(-Config.dt / (Config.R1 * Config.C1))); % 초기 V1 값
Vt_est(1) = udds_voltage(1);

% 진정한 SOC 계산
true_SOC = calculate_true_soc(udds_current, diff([0; udds_time]), Config.cap, true_SOC(1));

% 시뮬레이션 루프
for k = 2:length(udds_current)
    % 샘플링 간격 동안 평균 전류 계산
    Config.ik = udds_current(k);

    % 배터리 모델로부터 true SOC 및 Vt 계산
    [true_SOC(k), Vt_true] = battery_model(true_SOC(k-1), V1_est(k-1), udds_current(k), Config, soc_values, ocv_values);

    % EKF를 사용한 SOC 추정
    [SOC_est(k), V1_est(k), Vt_est(k), P] = soc_estimation(SOC_est(k-1), V1_est(k-1), Vt_true, Config.ik, Config, P, soc_values, ocv_values);
end

% SOC 플롯
figure;
plot(udds_time, true_SOC, 'b', 'LineWidth', 1.5); hold on;
plot(udds_time, SOC_est, 'r--', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('SOC');
title('True SOC vs Estimated SOC during UDDS Cycle');
legend('True SOC', 'Estimated SOC');
grid on;

% Vt 플롯
figure;
plot(udds_time, udds_voltage, 'b', 'LineWidth', 1.5); hold on;
plot(udds_time, Vt_est, 'r--', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('V_t (V)');
title('Measured vs Estimated Terminal Voltage during UDDS Cycle');
legend('Measured V_t', 'Estimated V_t');
grid on;

% Coulomb 카운팅을 사용한 진정한 SOC 계산 함수
function soc = calculate_true_soc(current, dt, capacity, prev_soc)
    % SOC 배열 초기화
    soc = zeros(length(current), 1);
    soc(1) = prev_soc;
    
    % 전류 데이터를 통해 루프 실행
    for i = 2:length(current)
        delta_t = dt(i);
        soc(i) = soc(i-1) + (current(i) * delta_t) / (capacity * 3600);
    end
end

% OCV 함수
function v_ocv = ocv_soc(soc, soc_values, ocv_values)
    v_ocv = interp1(soc_values, ocv_values, soc, 'linear', 'extrap');
end

% 배터리 모델 함수
function [SOC_true, Vt_true] = battery_model(SOC_prev, V1_prev, ik, Config, soc_values, ocv_values)
    SOC_true = SOC_prev + (Config.dt / Config.cap * 3600) * Config.coulomb_efficient * ik;
    Vt_true = ocv_soc(SOC_true, soc_values, ocv_values) - V1_prev - Config.R0 * ik;
end

% SOC 추정 함수
function [SOC_est, V1_est, Vt_est, P] = soc_estimation(SOC_est, V1_est, Vt_true, ik, Config, P, soc_values, ocv_values)
    Q = [1e-5 0; 0 1e-5]; % 프로세스 노이즈 공분산
    R = 1e-2; % 측정 노이즈 공분산
    
    % 예측 단계
    SOC_pred = SOC_est + (Config.dt / Config.cap *3600 ) * Config.coulomb_efficient * ik;
    V1_pred = exp(-Config.dt / (Config.R1 * Config.C1)) * V1_est + (1 - exp(-Config.dt / (Config.R1 * Config.C1))) * ik * Config.R1;
    X_pred = [SOC_pred; V1_pred];
    
    % 상태 전이 행렬
    A = [1, 0; 0, exp(-Config.dt / (Config.R1 * Config.C1))];
    
    % 프로세스 공분산 업데이트
    P = A * P * A' + Q;
    
    % 측정 예측
    Vt_pred = interp1(soc_values, ocv_values, SOC_pred, 'linear', 'extrap') - Config.R1 * V1_pred - Config.R0 * ik;
    
    % 측정 업데이트
    K = P * [1; 0] / (P(1, 1) + R); % 칼만 이득
    z = Vt_true - Vt_pred; % 측정 잔차
    X_pred = X_pred + K * z;
    P = (eye(2) - K * [1, 0]) * P;
    
    % 추정값 저장
    SOC_est = X_pred(1);
    V1_est = X_pred(2);
    Vt_est = Vt_pred + K(1) * z; % 추정 Vt
end
