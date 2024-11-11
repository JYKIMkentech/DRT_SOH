clc; clear; close all;

% Load UDDS data
load('C:\Users\USER\Desktop\Panasonic 18650PF Data\Panasonic 18650PF Data\25degC\Drive cycles\03-21-17_00.29 25degC_UDDS_Pan18650PF.mat');
udds_current = meas.Current; % UDDS 전류 데이터
udds_voltage = meas.Voltage; % UDDS 전압 데이터
udds_time = meas.Time; % UDDS 시간 데이터

% Load SOC-OCV structure
load('soc_ocv.mat', 'soc_ocv');
soc_values = soc_ocv(:, 1);
ocv_values = soc_ocv(:, 2);

% Initialize configuration
Config = initialize_config(udds_time, udds_current);

% Initialize SOC estimation
SOC_est = zeros(length(udds_current), 1);
SOC_est(1) = 1; % 초기 SOC는 100%로 가정

% Initialize true SOC using coulomb counting
true_SOC = zeros(length(udds_current), 1);
true_SOC(1) = 1; % 초기 SOC는 100%로 가정

% EKF 초기화
P = eye(2); % Initial estimation error covariance

% Preallocate arrays for V1 and Vt
V1_est = zeros(length(udds_current), 1);
Vt_est = zeros(length(udds_current), 1);
V1_est(1) = udds_current(1) * Config.R1 * (1 - exp(-Config.dt / (Config.R1 * Config.C1))); % 초기 V1 값
Vt_est(1) = udds_voltage(1);

% Simulation loop
for k = 2:length(udds_current)
    Config.ik = udds_current(k);

    % True SOC calculation using coulomb counting
    delta_t = udds_time(k) - udds_time(k-1);
    true_SOC(k) = true_SOC(k-1) - (udds_current(k) * delta_t) / (Config.cap * 3600);

    % SOC estimation using EKF
    [SOC_est(k), V1_est(k), Vt_est(k), P] = kalman_filter(SOC_est(k-1), V1_est(k-1), udds_voltage(k), Config.ik, Config, P, soc_values, ocv_values);
end

% Plot SOC
figure;
%plot(udds_time, true_SOC, 'b', 'LineWidth', 1.5); hold on;
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
