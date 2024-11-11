clc; clear; close all;

% Load UDDS data
load('C:\Users\deu04\OneDrive\바탕 화면\wykht8y7tg-1\Panasonic 18650PF Data\Panasonic 18650PF Data\25degC\Drive cycles\03-21-17_00.29 25degC_UDDS_Pan18650PF.mat');
udds_current = meas.Current; % UDDS 전류 데이터
udds_voltage = meas.Voltage; % UDDS 전압 데이터
udds_time = meas.Time; % UDDS 시간 데이터

% Load SOC-OCV structure
load('soc_ocv.mat', 'soc_ocv'); %c/20에서 가져온 soc-ocv lookup table
soc_values = soc_ocv(:, 1); % soc
ocv_values = soc_ocv(:, 2); % ocv

% Configuration parameters
Config.dt = mean(diff(udds_time)); % 평균 시간 간격
Config.ik = udds_current(1); % 초기 전류
Config.R0 = 0.001884314;
Config.R1 = 0.045801322;
Config.C1 = 4846.080679 * 1e-6;
Config.cap = (2.99 * 3600) / 100; % nominal capacity [Ah] 
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
P = [3000 0;
    0 3000]; % Initial estimation error covariance

% Preallocate arrays for V1 and Vt
V1_est = zeros(length(udds_current), 1);
Vt_est = zeros(length(udds_current), 1);
V1_est(1) = udds_current(1) * Config.R1 * (1 - exp(-Config.dt / (Config.R1 * Config.C1))); % 초기 V1 값 (전 상태의 v1값이 없기때문에 앞의 항이 zero)
Vt_est(1) = udds_voltage(1); % 초기 측정 전압 = estimated 전압 

% State transition function
stateTransitionFcn = @(x, u) [x(1) + (Config.dt / Config.cap * 3600) * Config.coulomb_efficient * u; ...
                              exp(-Config.dt / (Config.R1 * Config.C1)) * x(2) + (1 - exp(-Config.dt / (Config.R1 * Config.C1))) * u * Config.R1];

% Measurement function
measurementFcn = @(x, u) interp1(unique_ocv_values, unique_soc_values, x(1), 'linear', 'extrap') - x(2) - Config.R0 * u;

% Create EKF object
ekf = extendedKalmanFilter(stateTransitionFcn, measurementFcn, [SOC_est(1); V1_est(1)], 'MeasurementNoise', 8000, 'ProcessNoise', [1e-5, 0; 0, 1e-5]);

% Simulation loop
for k = 2:length(udds_current)
    Config.ik = udds_current(k); % 샘플링 시간동안 흐르는 전류 

    % True SOC calculation using coulomb counting
    delta_t = Config.dt; % dt 써도 되는데, 정확한 값을 위하여... (비교해보니까 거의 비슷하긴 함) 
    true_SOC(k) = true_SOC(k-1) + (udds_current(k) * delta_t) / (Config.cap * 3600); % 실제 soc = 전 soc + i * dt/q_total (sampling 시간 동안 흐르는 전류)

    % Prediction and correction step
    [predictedState, P] = predict(ekf, udds_current(k));
    [correctedState, P] = correct(ekf, udds_voltage(k), udds_current(k));
    
    % Save estimates
    SOC_est(k) = correctedState(1);
    V1_est(k) = correctedState(2);
    Vt_est(k) = measurementFcn(predictedState, udds_current(k)); % Estimated Vt 업데이트
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

