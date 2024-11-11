clc;clear;close all

% Load optimized parameters
load('optimized_params_struct_final.mat')

I_1C = 2.8892; % 1C에 해당하는 전류 [A]

% Extract fields from the structure and flatten the arrays
SOC = cell2mat({optimized_params_struct.SOC});
R0 = cell2mat({optimized_params_struct.R0});
R1 = cell2mat({optimized_params_struct.R1});
C = cell2mat({optimized_params_struct.C});
Crate = cell2mat({optimized_params_struct.Crate});

% Crate를 전류로 변환
current = Crate * I_1C;

% Load UDDS data
load('G:\공유 드라이브\BSL_Data3\Driving cycles\03-21-17_00.29 25degC_UDDS_Pan18650PF.mat');
udds_current = meas.Current; % UDDS 전류 데이터
udds_voltage = meas.Voltage; % UDDS 전압 데이터
udds_time = meas.Time; % UDDS 시간 데이터

% Load SOC-OCV structure
load('soc_ocv.mat', 'soc_ocv'); % c/20에서 가져온 soc-ocv lookup table
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

% Initial Crate 계산
initial_current = udds_current(1); % 첫 전류

% 2D Interpolation for R0, R1, C using initial SOC and current 
initial_R0 = interp2(SOC, current, R0, initial_soc, initial_current, 'linear', 'extrap');
initial_R1 = interp2(SOC, current, R1, initial_soc, initial_current, 'linear', 'extrap');
initial_C1 = interp2(SOC, current, C, initial_soc, initial_current, 'linear', 'extrap');

% Config 구조체에 필드 추가
Config.R0 = initial_R0;
Config.R1 = initial_R1;
Config.C1 = initial_C1;

% Initialize SOC estimation
SOC_est = zeros(length(udds_current), 1);
SOC_est(1) = initial_soc; 

% Initialize true SOC using coulomb counting
true_SOC = zeros(length(udds_current), 1);
true_SOC(1) = initial_soc; 

% Preallocate arrays for V1, Vt, R0, R1, and C
V1_est = zeros(length(udds_current), 1);
Vt_est = zeros(length(udds_current), 1);
R0_used = zeros(length(udds_current), 1);
R1_used = zeros(length(udds_current), 1);
C_used = zeros(length(udds_current), 1);

% EKF 초기화
P = [1 0;
    0 1]; % Initial estimation error covariance

Q = [1e-4 0; 
         0 1e-4]; % Process noise covariance
R = 0.025; % Measurement noise covariance 

% Initial values
V1_est(1) = udds_current(1) * Config.R1 * (1 - exp(-Config.dt / (Config.R1 * Config.C1))); % 초기 V1 값
Vt_est(1) = udds_voltage(1); % 초기 측정 전압 = estimated 전압 
R0_used(1) = Config.R0;
R1_used(1) = Config.R1;
C_used(1) = Config.C1;

% Preallocate arrays for H vector components
H1 = zeros(length(udds_current), 1);
H2 = zeros(length(udds_current), 1);

% Simulation loop
for k = 2:length(udds_current)
    
    Config.ik = udds_current(k); % 샘플링 시간동안 흐르는 전류 

    % True SOC calculation using coulomb counting
    delta_t = Config.dt; 
    true_SOC(k) = true_SOC(k-1) + (udds_current(k) * delta_t) / (Config.cap * 3600);

    % 현재 전류 계산
    current_k = udds_current(k);

    % 2D Interpolation for R0, R1, C(SOC, current)
    R0 = interp2(SOC, current, R0, SOC_est(k-1), current_k, 'linear', 'extrap');
    R1 = interp2(SOC, current, R1, SOC_est(k-1), current_k, 'linear', 'extrap');
    C1 = interp2(SOC, current, C, SOC_est(k-1), current_k, 'linear', 'extrap');
    
    Config.R0 = R0;
    Config.R1 = R1;
    Config.C1 = C1;
    
    % Store the values used
    R0_used(k) = R0;
    R1_used(k) = R1;
    C_used(k) = C1;

    % SOC estimation using EKF
    [SOC_est(k), V1_est(k), Vt_est(k), P, H_k] = soc_estimation(SOC_est(k-1), V1_est(k-1), udds_voltage(k), udds_current(k), Config, P, unique_soc_values, unique_ocv_values,Q,R);

    % Store H vector components
    H1(k) = H_k(1);
    H2(k) = H_k(2);
end


% Calculate residuals between measured and estimated terminal voltage
residuals = udds_voltage - Vt_est;

% 잔차의 분산 계산 (센서 노이즈 공분산 R)
R_calculated = var(residuals);

fprintf('Calculated sensor noise covariance R: %.4f\n', R_calculated);

% 필터에 사용할 R 값 업데이트 가능
% SOC 추정 함수 호출 시 R_calculated를 사용하여 EKF에 전달 가능

% Calculate SOC RMSE
soc_rmse = sqrt(mean((true_SOC - SOC_est).^2));
fprintf('SOC RMSE: %.4f\n', soc_rmse);

% Plot SOC RMSE
figure;
plot(udds_time, true_SOC - SOC_est, 'g', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('SOC Error');
title('SOC Estimation Error Over Time');
grid on;
legend(sprintf('SOC RMSE: %.4f', soc_rmse));

% Create the result matrix as a Nx5 array
result_matrix = [true_SOC, SOC_est, R0_used, R1_used, C_used];

% Define column names
column_names = {'True_SOC', 'Estimated_SOC', 'R0', 'R1', 'C'};

% % Save the result matrix and column names to a .mat file
% save('result_matrix.mat', 'result_matrix', 'column_names');

% SOC 추정 함수
function [SOC_est, V1_est, Vt_est, P, H_k] = soc_estimation(SOC_est, V1_est, Vt_true, ik, Config, P, soc_values, ocv_values, Q , R)
    
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
    P =  P_predict - K * H_k * P_predict;
    
    % Update the estimated terminal voltage
    Vt_est = interp1(soc_values, ocv_values, SOC_est, 'linear', 'extrap') + V1_est + Config.R0 * ik;
end
