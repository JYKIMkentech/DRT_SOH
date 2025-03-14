clc; clear; close all;

%% LOAD DATA

% Load ECM parameters from HPPC test
load('optimized_params_struct_final.mat'); % Contains fields: R0, R1, C, SOC, avgI, m, Crate

% Load parsed UDDS data (17 trips)
load('udds_data.mat'); % Contains struct array 'udds_data' with fields V, I, t, SOC

% Load SOC-OCV lookup table from C/20 test
load('soc_ocv.mat', 'soc_ocv'); % [SOC, OCV]
soc_values = soc_ocv(:, 1);     % SOC values
ocv_values = soc_ocv(:, 2);     % Corresponding OCV values [V]

%% PARAMETERS

I_1C = 2.8892;           % 1C current rate [A]
Config.cap = 2.90;       % Nominal capacity [Ah]
Config.coulomb_efficiency = 1; % Coulombic efficiency

% Remove duplicate OCV values for interpolation
[unique_ocv_values, unique_idx] = unique(ocv_values);
unique_soc_values = soc_values(unique_idx);

%% PREPARE ECM PARAMETERS

% Extract ECM parameters from the loaded structure
SOC_param = [optimized_params_struct.SOC];
R0_param = [optimized_params_struct.R0];
R1_param = [optimized_params_struct.R1];
C1_param = [optimized_params_struct.C];
Crate_param = [optimized_params_struct.Crate];

% Create scatteredInterpolant objects for R0, R1, C1
F_R0 = scatteredInterpolant(SOC_param', Crate_param', R0_param', 'linear', 'nearest');
F_R1 = scatteredInterpolant(SOC_param', Crate_param', R1_param', 'linear', 'nearest');
F_C1 = scatteredInterpolant(SOC_param', Crate_param', C1_param', 'linear', 'nearest');

%% INITIALIZE KALMAN FILTER PARAMETERS

% Initial covariance matrix
P_init = diag([1e-4, 1e-4]);

% Process noise covariance
Q = diag([1e-7, 1e-7]);

% Measurement noise covariance
R_cov = 1e-3; % Adjust based on measurement noise characteristics

%% LOOP OVER EACH TRIP

num_trips = length(udds_data);

for trip_num = 1:num_trips
    % Extract data for the current trip
    trip_current = udds_data(trip_num).I;      % Current [A]
    trip_voltage = udds_data(trip_num).V;      % Voltage [V]
    trip_time = udds_data(trip_num).t;         % Time [s]
    trip_SOC_true = udds_data(trip_num).SOC;   % True SOC
    
    % Initial SOC estimation
    initial_voltage = trip_voltage(1);
    initial_soc = interp1(unique_ocv_values, unique_soc_values, initial_voltage, 'linear', 'extrap');
    SOC_est = initial_soc; % Initial SOC estimate
    V1_est = 0;            % Initial V1 estimate [V]
    X_est = [SOC_est; V1_est];
    
    % Initialize covariance matrix
    P = P_init;
    
    % Preallocate arrays for saving results
    num_samples = length(trip_time);
    SOC_save = zeros(num_samples, 1);
    Vt_est_save = zeros(num_samples, 1);
    Vt_meas_save = trip_voltage;
    Time_save = trip_time;
    
    % Save initial SOC
    SOC_save(1) = SOC_est;
    OCV_initial = interp1(unique_soc_values, unique_ocv_values, SOC_est, 'linear', 'extrap');
    Vt_est_save(1) = OCV_initial + V1_est + F_R0(SOC_est, 0) * trip_current(1);
    
    %% MAIN LOOP FOR KALMAN FILTER FOR CURRENT TRIP
    
    for k = 2:num_samples
        % Time update
        dt = trip_time(k) - trip_time(k-1); % [s]
        ik = trip_current(k);               % Current at time k [A]
        
        % Predict SOC using Coulomb counting
        SOC_pred = SOC_est - (dt / (Config.cap * 3600)) * Config.coulomb_efficiency * ik;
        
        % Ensure SOC is within bounds [0, 1]
        SOC_pred = max(0, min(1, SOC_pred));
        
        % Current C-rate
        Crate_current = abs(ik) / Config.cap;
        
        % Interpolate ECM parameters at current SOC and C-rate
        R0_interp = F_R0(SOC_pred, Crate_current);
        R1_interp = F_R1(SOC_pred, Crate_current);
        C1_interp = F_C1(SOC_pred, Crate_current);
        
        % Prevent negative or zero values
        R0_interp = max(R0_interp, 1e-5);
        R1_interp = max(R1_interp, 1e-5);
        C1_interp = max(C1_interp, 1e-5);
        
        % State-space matrices for prediction
        A_k = [1, 0;
               0, exp(-dt / (R1_interp * C1_interp))];
        B_k = [-(dt / (Config.cap * 3600)) * Config.coulomb_efficiency;
               R1_interp * (1 - exp(-dt / (R1_interp * C1_interp)))];
        
        % Predict state
        X_pred = A_k * X_est + B_k * ik;
        SOC_pred = X_pred(1);
        V1_pred = X_pred(2);
        
        % Jacobian of the state transition function (for EKF)
        F_k = A_k;
        
        % Predict covariance
        P_predict = F_k * P * F_k' + Q;
        
        % Predict terminal voltage
        OCV_pred = interp1(unique_soc_values, unique_ocv_values, SOC_pred, 'linear', 'extrap');
        Vt_pred = OCV_pred + V1_pred + R0_interp * ik;
        
        % Compute Jacobian of the measurement function
        % H_k = [dV/dSOC, dV/dV1]
        delta_SOC = 1e-5;
        OCV_plus = interp1(unique_soc_values, unique_ocv_values, SOC_pred + delta_SOC, 'linear', 'extrap');
        OCV_minus = interp1(unique_soc_values, unique_ocv_values, SOC_pred - delta_SOC, 'linear', 'extrap');
        dOCV_dSOC = (OCV_plus - OCV_minus) / (2 * delta_SOC);
        H_k = [dOCV_dSOC, 1];
        
        % Measurement residual
        Vt_meas = trip_voltage(k); % Measured terminal voltage [V]
        y_tilde = Vt_meas - Vt_pred;
        
        % Kalman gain
        S_k = H_k * P_predict * H_k' + R_cov;
        K = (P_predict * H_k') / S_k;
        
        % Update state estimate
        X_est = X_pred + K * y_tilde;
        SOC_est = X_est(1);
        V1_est = X_est(2);
        
        % SOC 범위 제한
        SOC_est = max(0, min(1, SOC_est));
        
        % Update covariance estimate
        P = (eye(2) - K * H_k) * P_predict;
        
        % Update the estimated terminal voltage
        OCV_updated = interp1(unique_soc_values, unique_ocv_values, SOC_est, 'linear', 'extrap');
        Vt_est = OCV_updated + V1_est + R0_interp * ik;
        
        % Save results
        SOC_save(k) = SOC_est;
        Vt_est_save(k) = Vt_est;
    end
    
    %% PLOT RESULTS FOR CURRENT TRIP
    
    figure('Name', sprintf('Trip %d Results', trip_num), 'NumberTitle', 'off');
    
    % Subplot 1: SOC Comparison
    subplot(3,1,1);
    plot(Time_save , trip_SOC_true * 100, 'k', 'LineWidth', 1.5, 'DisplayName', 'True SOC');
    hold on;
    plot(Time_save , SOC_save * 100, 'b--', 'LineWidth', 1.5, 'DisplayName', 'Estimated SOC');
    xlabel('Time [hours]');
    ylabel('SOC [%]');
    title(sprintf('Trip %d: SOC Estimation using Extended Kalman Filter', trip_num));
    legend('Location', 'best');
    grid on;
    
    % Subplot 2: Terminal Voltage Comparison
    subplot(3,1,2);
    plot(Time_save , Vt_meas_save, 'k', 'LineWidth', 1.0, 'DisplayName', 'Measured Voltage');
    hold on;
    plot(Time_save , Vt_est_save, 'r--', 'LineWidth', 1.0, 'DisplayName', 'Estimated Voltage');
    xlabel('Time [hours]');
    ylabel('Terminal Voltage [V]');
    legend('Location', 'best');
    grid on;
    
    % Subplot 3: Current Profile
    subplot(3,1,3);
    plot(Time_save , trip_current, 'g', 'LineWidth', 1.0);
    xlabel('Time [hours]');
    ylabel('Current [A]');
    title('Current Profile');
    grid on;
    
    
end



%%save

% HPPC 기반 칼만 필터 코드에서 첫 번째 트립 처리 후에 추가

% SOC_save: 칼만 필터로 추정된 SOC (이미 코드에서 사용됨)
% Time_save: 시간 벡터 (이미 코드에서 사용됨)

% 첫 번째 트립의 결과를 저장
HPPC_SOC_est = SOC_save;      % 추정된 SOC
HPPC_Time = Time_save;        % 시간 벡터

% 결과 저장
save('HPPC_SOC_trip1.mat', 'HPPC_SOC_est', 'HPPC_Time');

