clc; clear; close all;

% Load optimized parameters
load('optimized_params_filtered.mat', 'optimized_params_filtered');

% Load UDDS data
load('G:\공유 드라이브\BSL_Data3\Driving cycles\03-21-17_00.29 25degC_UDDS_Pan18650PF.mat');
udds_current = meas.Current; % UDDS 전류 데이터
udds_voltage = meas.Voltage; % UDDS 전압 데이터
udds_time = meas.Time; % UDDS 시간 데이터

% Load SOC-OCV structure
load('soc_ocv.mat', 'soc_ocv'); %c/20에서 가져온 soc-ocv lookup table
soc_values = soc_ocv(:, 1); % soc
ocv_values = soc_ocv(:, 2); % ocv

% Configuration parameters
Config.dt = mean(diff(udds_time)); % 평균 시간 간격
Config.cap = 2.90; % nominal capacity [Ah] 
Config.coulomb_efficient = 1;

% Remove duplicate OCV values
[unique_ocv_values, unique_idx] = unique(ocv_values);
unique_soc_values = soc_values(unique_idx);

% Interpolate SOC for the first UDDS voltage value
initial_voltage = udds_voltage(1);
initial_soc = interp1(unique_ocv_values, unique_soc_values, initial_voltage, 'linear', 'extrap');

fprintf('Initial voltage: %.2f V corresponds to SOC: %.2f%%\n', initial_voltage, initial_soc * 100);

% Initialize SOC estimation
SOC_est = zeros(length(udds_current), 1);
SOC_est(1) = initial_soc; % (udds voltage 4.18v) 

% Initialize true SOC using coulomb counting
true_SOC = zeros(length(udds_current), 1);
true_SOC(1) = initial_soc; 

% EKF 초기화
P = [1 0;
    0 1]; % Initial estimation error covariance

% Preallocate arrays for V1 and Vt
V1_est = zeros(length(udds_current), 1);
Vt_est = zeros(length(udds_current), 1);
V1_est(1) = udds_current(1) * Config.R1 * (1 - exp(-Config.dt / (Config.R1 * Config.C1))); % 초기 V1 값 (전 상태의 v1값이 없기때문에 앞의 항이 zero)
Vt_est(1) = udds_voltage(1); % 초기 측정 전압 = estimated 전압 

% Simulation loop
for k = 2:length(udds_current)
    Config.ik = udds_current(k); % 샘플링 시간동안 흐르는 전류 

    % True SOC calculation using coulomb counting
    delta_t = Config.dt; % dt 써도 되는데, 정확한 값을 위하여... (비교해보니까 거의 비슷하긴 함) 
    true_SOC(k) = true_SOC(k-1) + (udds_current(k) * delta_t) / (Config.cap * 3600); % 실제 soc = 전 soc + i * dt/q_total (sampling 시간 동안 흐르는 전류)

    % 최적화된 매개변수에서 SOC에 따른 R0, R1, C 값 업데이트
    R0 = interp1([optimized_params_filtered.SOC], [optimized_params_filtered.R0], SOC_est(k-1), 'linear', 'extrap');
    R1 = interp1([optimized_params_filtered.SOC], [optimized_params_filtered.R1], SOC_est(k-1), 'linear', 'extrap');
    C1 = interp1([optimized_params_filtered.SOC], [optimized_params_filtered.C], SOC_est(k-1), 'linear', 'extrap');
    
    Config.R0 = R0;
    Config.R1 = R1;
    Config.C1 = C1;

    % SOC estimation using EKF
    [SOC_est(k), V1_est(k), Vt_est(k), P] = soc_estimation(SOC_est(k-1), V1_est(k-1), udds_voltage(k), udds_current(k), Config, P, unique_soc_values, unique_ocv_values); % 추정 코드
end

% Plot SOC
figure;
plot(udds_time, true_SOC, 'b', 'LineWidth', 1.5); hold on;
plot(udds_time, SOC_est, 'r--', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('SOC');
title('True SOC vs Estimated SOC during UDDS Cycle');
legend('True SOC', 'Estimated SOC');
grid on;

% Plot Vt
figure;
plot(udds_time, udds_voltage, 'b', 'LineWidth', 1.5); hold on;
plot(udds_time, Vt_est, 'r--', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('V_t (V)');
title('Measured vs Estimated Terminal Voltage during UDDS Cycle');
legend('Measured V_t', 'Estimated V_t');
grid on;

% SOC 추정 함수
function [SOC_est, V1_est, Vt_est, P] = soc_estimation(SOC_est, V1_est, Vt_true, ik, Config, P, soc_values, ocv_values)
    Q = [1e-5 0; 
         0 1e-5]; % Process noise covariance
    R = 1000; % Measurement noise covariance

    % Prediction step (상태방정식)
    SOC_pred = SOC_est + (Config.dt / (Config.cap * 3600)) * Config.coulomb_efficient * ik;
    V1_pred = exp(-Config.dt / (Config.R1 * Config.C1)) * V1_est + (1 - exp(-Config.dt / (Config.R1 * Config.C1))) * ik * Config.R1;
    X_pred = [SOC_pred; V1_pred];

   
    % State transition matrix
    A = [1, 0;
         0, exp(-Config.dt / (Config.R1 * Config.C1))];
    
    % Process covariance update
    P_predict = A * P * A' + Q; % p predict 예측 과정
    
    % Measurement prediction (OCV = 전류적산법으로 얻은 SOC로부터 lookup table 통과시켜 얻음)
    Vt_pred = interp1(soc_values, ocv_values, SOC_pred, 'linear', 'extrap') + V1_pred + Config.R0 * ik;

     % H 행렬 계산
    OCV_H = interp1(soc_values, ocv_values, SOC_pred, 'linear', 'extrap');
    OCV_H_before = interp1(soc_values, ocv_values, SOC_est, 'linear', 'extrap');
    
    if SOC_pred == SOC_est
        H_k = [0 -1];
    else
        H_k = [(OCV_H - OCV_H_before) / (SOC_pred - SOC_est), -1];
    end
    

    % Kalman gain
    S = H_k * P_predict * H_k' + R;
    K = P_predict * H_k' / S;

    % Measurement residual
    y_tilde = Vt_true - Vt_pred;

    % State update
    X_est = X_pred + K * y_tilde;
    SOC_est = X_est(1);
    V1_est = X_est(2);

    % Covariance update
    P = P_predict - K * H_k * P_predict;
    
    % Update the estimated terminal voltage
    Vt_est = interp1(soc_values, ocv_values, SOC_est, 'linear', 'extrap') + V1_est + Config.R0 * ik;
end


