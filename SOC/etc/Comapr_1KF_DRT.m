clc; clear; close all;

%% 1. 데이터 로드

% ECM 파라미터 (HPPC 테스트로부터)
load('optimized_params_struct_final.mat'); % 필드: R0, R1, C, SOC, avgI, m, Crate

% DRT 파라미터 (gamma 및 tau 값)
load('gamma_data.mat', 'gamma_sorted', 'soc_sorted', 'theta_discrete', 'R0_est_all', 'soc_mid_all');
tau_discrete = exp(theta_discrete); % tau 값

% SOC-OCV 룩업 테이블 (C/20 테스트로부터)
load('soc_ocv.mat', 'soc_ocv'); % [SOC, OCV]
soc_values = soc_ocv(:, 1);     % SOC 값
ocv_values = soc_ocv(:, 2);     % 해당하는 OCV 값 [V]

% 주행 데이터 (17개의 트립)
load('udds_data.mat'); % 구조체 배열 'udds_data'로 V, I, t, Time_duration, SOC 필드 포함

%% 2. 공통 설정

I_1C = 2.8892;           % 1C 전류 [A]
Config.cap = 2.90;       % 명목 용량 [Ah]
Config.coulomb_efficiency = 1; % 쿨롱 효율

% OCV 중복 값 제거 (보간을 위해)
[unique_ocv_values, unique_idx] = unique(ocv_values);
unique_soc_values = soc_values(unique_idx);

%% 3. ECM 파라미터 준비 (HPPC 기반)

% HPPC로부터 얻은 ECM 파라미터 추출
SOC_param = [optimized_params_struct.SOC];
R0_param = [optimized_params_struct.R0];
R1_param = [optimized_params_struct.R1];
C1_param = [optimized_params_struct.C];
Crate_param = [optimized_params_struct.Crate];

% R0, R1, C1에 대한 보간 함수 생성
F_R0 = scatteredInterpolant(SOC_param', Crate_param', R0_param', 'linear', 'nearest');
F_R1 = scatteredInterpolant(SOC_param', Crate_param', R1_param', 'linear', 'nearest');
F_C1 = scatteredInterpolant(SOC_param', Crate_param', C1_param', 'linear', 'nearest');

%% 4. 칼만 필터 설정 - HPPC

% 초기 공분산 행렬 (HPPC 기반)
P_init_HPPC = diag([1e-4, 1e-4]);

% 프로세스 잡음 공분산 (HPPC 기반)
Q_HPPC = diag([1e-7, 1e-7]);

% 측정 잡음 공분산 (HPPC 기반)
R_HPPC = 1e-3; % 측정 잡음 특성에 따라 조정


%% 4. 칼만 필터 설정 - DRT

num_RC = length(theta_discrete); % RC 소자의 개수 (DRT 기반)
state_dimension = 1 + num_RC; % 상태 벡터 차원: [SOC; V_RC_1; V_RC_2; ... ; V_RC_n]

% 초기 공분산 행렬 (DRT 기반)
P_init_DRT = zeros(state_dimension);
P_init_DRT(1,1) = (0.0004)^2; % SOC에 대한 초기 분산
% V_RC_i에 대한 초기 분산은 모두 0으로 설정

% 프로세스 잡음 공분산 행렬 (DRT 기반)
Q_DRT = zeros(state_dimension);
Q_DRT(1,1) = 1e-14; % SOC에 대한 프로세스 잡음 분산
for i = 2:state_dimension
    Q_DRT(i,i) = (0.0016)^2; % 각 V_RC_i에 대한 프로세스 잡음 분산
end

% 측정 잡음 공분산 (DRT 기반)
R_DRT = 5.25e-16; % 측정 잡음 분산

%% 5. 트립 반복

num_trips = length(udds_data);

% 모든 트립의 결과를 저장할 배열 초기화
all_SOC_true = cell(num_trips-1, 1);
all_SOC_HPPC = cell(num_trips-1, 1);
all_SOC_DRT = cell(num_trips-1, 1);
all_Vt_meas = cell(num_trips-1, 1);
all_Vt_HPPC = cell(num_trips-1, 1);
all_Vt_DRT = cell(num_trips-1, 1);
all_time = cell(num_trips-1, 1);
all_current = cell(num_trips-1, 1);

for trip_num = 1:num_trips-1
    %% 5.1. 트립 데이터 추출
    trip_current = udds_data(trip_num).I;      % 전류 [A]
    trip_voltage = udds_data(trip_num).V;      % 전압 [V]
    trip_time = udds_data(trip_num).Time_duration; % 누적 시간 [s]
    trip_SOC_true = udds_data(trip_num).SOC;   % 실제 SOC (있는 경우)
    
    %% 5.2. 초기화
    
    % 시간 간격 계산 (DRT 기반에서 필요)
    dt = [0; diff(trip_time)];
    if dt(1) == 0
        dt(1) = dt(2);
    end
    
    % 초기 SOC 추정 (전압 기반)
    initial_voltage = trip_voltage(1);
    initial_soc = interp1(unique_ocv_values, unique_soc_values, initial_voltage, 'linear', 'extrap');
    
    % HPPC 기반 칼만 필터 초기화
    SOC_est_HPPC = initial_soc;
    V1_est_HPPC = 0;
    X_est_HPPC = [SOC_est_HPPC; V1_est_HPPC];
    P_HPPC = P_init_HPPC;
    
    % DRT 기반 칼만 필터 초기화
    SOC_est_DRT = initial_soc;
    X_est_DRT = zeros(state_dimension, 1);
    X_est_DRT(1) = SOC_est_DRT;
    P_DRT = P_init_DRT;
    
    % 초기 V_RC_est 설정 (DRT 기반)
    gamma_current_init = interp1(soc_sorted, gamma_sorted, SOC_est_DRT, 'linear', 'extrap');
    delta_theta = theta_discrete(2) - theta_discrete(1);
    R_i_init = gamma_current_init * delta_theta;
    C_i_init = tau_discrete ./ R_i_init;
    V_RC_init = trip_current(1) * R_i_init .* (1 - exp(-dt(1) ./ (R_i_init .* C_i_init)));
    X_est_DRT(2:end) = V_RC_init(:);
    
    % 결과 저장을 위한 변수 초기화
    num_samples = length(trip_time);
    SOC_save_HPPC = zeros(num_samples, 1);
    Vt_est_save_HPPC = zeros(num_samples, 1);
    SOC_save_DRT = zeros(num_samples, 1);
    Vt_est_save_DRT = zeros(num_samples, 1);
    Vt_meas_save = trip_voltage;
    Time_save = trip_time;
    
    % 초기값 저장
    SOC_save_HPPC(1) = SOC_est_HPPC;
    OCV_initial_HPPC = interp1(unique_soc_values, unique_ocv_values, SOC_est_HPPC, 'linear', 'extrap');
    Vt_est_save_HPPC(1) = OCV_initial_HPPC + V1_est_HPPC + F_R0(SOC_est_HPPC, 0) * trip_current(1);
    
    SOC_save_DRT(1) = SOC_est_DRT;
    OCV_initial_DRT = interp1(unique_soc_values, unique_ocv_values, SOC_est_DRT, 'linear', 'extrap');
    Vt_est_save_DRT(1) = OCV_initial_DRT + sum(X_est_DRT(2:end)) + R0_est_all(trip_num) * trip_current(1);
    
    %% 5.3. 메인 칼만 필터 루프
    
    for k = 2:num_samples
        % 공통 계산
        dt_k = trip_time(k) - trip_time(k-1); % 시간 간격 [s]
        ik = trip_current(k); % 현재 전류 [A]
        
        %% 5.3.1. HPPC 기반 칼만 필터 업데이트
        
        % SOC 예측 (쿨롱 카운팅)
        SOC_pred_HPPC = SOC_est_HPPC + (dt_k / (Config.cap * 3600)) * Config.coulomb_efficiency * ik;
        SOC_pred_HPPC = max(0, min(1, SOC_pred_HPPC));
        
        % 현재 C-rate 계산
        Crate_current = abs(ik) / Config.cap;
        
        % ECM 파라미터 보간
        R0_interp = F_R0(SOC_pred_HPPC, Crate_current);
        R1_interp = F_R1(SOC_pred_HPPC, Crate_current);
        C1_interp = F_C1(SOC_pred_HPPC, Crate_current);
        
        % 음수나 0 방지
        R0_interp = max(R0_interp, 1e-5);
        R1_interp = max(R1_interp, 1e-5);
        C1_interp = max(C1_interp, 1e-5);
        
        % 상태 천이 행렬 및 입력 행렬
        A_k_HPPC = [1, 0;
                    0, exp(-dt_k / (R1_interp * C1_interp))];
        B_k_HPPC = [-(dt_k / (Config.cap * 3600)) * Config.coulomb_efficiency;
                     R1_interp * (1 - exp(-dt_k / (R1_interp * C1_interp)))];
        
        % 상태 예측
        X_pred_HPPC = A_k_HPPC * X_est_HPPC + B_k_HPPC * ik;
        SOC_pred_HPPC = X_pred_HPPC(1);
        V1_pred_HPPC = X_pred_HPPC(2);
        
        % 공분산 예측
        P_predict_HPPC = A_k_HPPC * P_HPPC * A_k_HPPC' + Q_HPPC;
        
        % 전압 예측
        OCV_pred_HPPC = interp1(unique_soc_values, unique_ocv_values, SOC_pred_HPPC, 'linear', 'extrap');
        Vt_pred_HPPC = OCV_pred_HPPC + V1_pred_HPPC + R0_interp * ik;
        
        % 관측 행렬 H 계산
        delta_SOC = 1e-5;
        OCV_plus = interp1(unique_soc_values, unique_ocv_values, SOC_pred_HPPC + delta_SOC, 'linear', 'extrap');
        OCV_minus = interp1(unique_soc_values, unique_ocv_values, SOC_pred_HPPC - delta_SOC, 'linear', 'extrap');
        dOCV_dSOC = (OCV_plus - OCV_minus) / (2 * delta_SOC);
        H_k_HPPC = [dOCV_dSOC, 1];
        
        % 잔차 계산
        Vt_meas = trip_voltage(k);
        y_tilde_HPPC = Vt_meas - Vt_pred_HPPC;
        
        % 칼만 이득 계산
        S_k_HPPC = H_k_HPPC * P_predict_HPPC * H_k_HPPC' + R_HPPC;
        K_HPPC = (P_predict_HPPC * H_k_HPPC') / S_k_HPPC;
        
        % 상태 업데이트
        X_est_HPPC = X_pred_HPPC + K_HPPC * y_tilde_HPPC;
        SOC_est_HPPC = X_est_HPPC(1);
        V1_est_HPPC = X_est_HPPC(2);
        SOC_est_HPPC = max(0, min(1, SOC_est_HPPC));
        
        % 공분산 업데이트
        P_HPPC = (eye(2) - K_HPPC * H_k_HPPC) * P_predict_HPPC;
        
        % 전압 업데이트
        OCV_updated_HPPC = interp1(unique_soc_values, unique_ocv_values, SOC_est_HPPC, 'linear', 'extrap');
        Vt_est_HPPC = OCV_updated_HPPC + V1_est_HPPC + R0_interp * ik;
        
        % 결과 저장
        SOC_save_HPPC(k) = SOC_est_HPPC;
        Vt_est_save_HPPC(k) = Vt_est_HPPC;
        
        %% 5.3.2. DRT 기반 칼만 필터 업데이트
        
        % SOC 예측 (쿨롱 카운팅)
        SOC_pred_DRT = X_est_DRT(1) + (dt_k / (Config.cap * 3600)) * ik;
        SOC_pred_DRT = max(0, min(1, SOC_pred_DRT));
        
        % 현재 SOC에 대한 gamma 값 보간
        gamma_current = interp1(soc_sorted, gamma_sorted, SOC_pred_DRT, 'linear', 'extrap');
        
        % 각 RC 소자에 대한 R_i 및 C_i 계산
        R_i = gamma_current * delta_theta;
        C_i = tau_discrete ./ R_i;
        
        % 상태 천이 행렬 A_DRT 계산
        A_DRT = diag([1; exp(-dt_k ./ tau_discrete(:))]);
        
        % RC 전압 예측
        V_RC_pred = (exp(-dt_k ./ (R_i .* C_i))') .* X_est_DRT(2:end) + (R_i' .* (1 - exp(-dt_k ./ (R_i .* C_i))')) .* ik;
        
        % 상태 예측
        X_pred_DRT = [SOC_pred_DRT; V_RC_pred];
        
        % 공분산 예측
        P_pred_DRT = A_DRT * P_DRT * A_DRT' + Q_DRT;
        
        % 전압 예측
        OCV_pred_DRT = interp1(unique_soc_values, unique_ocv_values, SOC_pred_DRT, 'linear', 'extrap');
        Vt_pred_DRT = OCV_pred_DRT + R0_est_all(trip_num) * ik + sum(V_RC_pred);
        
        % 관측 행렬 H_DRT 계산
        delta_SOC = 1e-5;
        OCV_plus = interp1(unique_soc_values, unique_ocv_values, SOC_pred_DRT + delta_SOC, 'linear', 'extrap');
        OCV_minus = interp1(unique_soc_values, unique_ocv_values, SOC_pred_DRT - delta_SOC, 'linear', 'extrap');
        dOCV_dSOC = (OCV_plus - OCV_minus) / (2 * delta_SOC);
        
        H_DRT = zeros(1, state_dimension);
        H_DRT(1) = dOCV_dSOC;
        H_DRT(2:end) = 1;
        
        % 잔차 계산
        y_tilde_DRT = Vt_meas - Vt_pred_DRT;
        
        % 칼만 이득 계산
        S_DRT = H_DRT * P_pred_DRT * H_DRT' + R_DRT;
        K_DRT = (P_pred_DRT * H_DRT') / S_DRT;
        
        % 상태 업데이트
        X_est_DRT = X_pred_DRT + K_DRT * y_tilde_DRT;
        SOC_est_DRT = X_est_DRT(1);
        SOC_est_DRT = max(0, min(1, SOC_est_DRT));
        
        % 공분산 업데이트
        P_DRT = P_pred_DRT - K_DRT * H_DRT * P_pred_DRT;
        
        % 전압 업데이트
        OCV_updated_DRT = interp1(unique_soc_values, unique_ocv_values, SOC_est_DRT, 'linear', 'extrap');
        Vt_est_DRT = OCV_updated_DRT + R0_est_all(trip_num) * ik + sum(X_est_DRT(2:end));
        
        % 결과 저장
        SOC_save_DRT(k) = SOC_est_DRT;
        Vt_est_save_DRT(k) = Vt_est_DRT;
    end
    
    %% 5.4. 결과 저장 및 시각화
    
    % HPPC 기반 결과 저장
    HPPC_SOC_est = SOC_save_HPPC;
    HPPC_Time = Time_save;
    save(sprintf('HPPC_SOC_trip%d.mat', trip_num), 'HPPC_SOC_est', 'HPPC_Time');
    
    % DRT 기반 결과 저장
    DRT_SOC_est = SOC_save_DRT;
    DRT_Time = Time_save;
    save(sprintf('DRT_SOC_trip%d.mat', trip_num), 'DRT_SOC_est', 'DRT_Time');
    
    % 트립 결과를 셀 배열에 저장
    all_SOC_true{trip_num} = trip_SOC_true;
    all_SOC_HPPC{trip_num} = SOC_save_HPPC;
    all_SOC_DRT{trip_num} = SOC_save_DRT;
    all_Vt_meas{trip_num} = Vt_meas_save;
    all_Vt_HPPC{trip_num} = Vt_est_save_HPPC;
    all_Vt_DRT{trip_num} = Vt_est_save_DRT;
    all_time{trip_num} = Time_save;
    all_current{trip_num} = trip_current;
    
    % 결과 시각화 (개별 트립)
    figure('Name', sprintf('Trip %d Results', trip_num), 'NumberTitle', 'off');
    
    % Subplot 1: SOC 비교
    subplot(3,1,1);
    plot(Time_save , trip_SOC_true * 100, 'k', 'LineWidth', 1.5, 'DisplayName', 'True SOC');
    hold on;
    plot(Time_save , SOC_save_HPPC * 100, 'b--', 'LineWidth', 1.5, 'DisplayName', 'Estimated SOC (HPPC)');
    plot(Time_save , SOC_save_DRT * 100, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Estimated SOC (DRT)');
    xlabel('Time [s]');
    ylabel('SOC [%]');
    title(sprintf('Trip %d: SOC Estimation using Kalman Filter', trip_num));
    legend('Location', 'best');
    grid on;
    
    % Subplot 2: 터미널 전압 비교
    subplot(3,1,2);
    plot(Time_save , Vt_meas_save, 'k', 'LineWidth', 1.0, 'DisplayName', 'Measured Voltage');
    hold on;
    plot(Time_save , Vt_est_save_HPPC, 'b--', 'LineWidth', 1.0, 'DisplayName', 'Estimated Voltage (HPPC)');
    plot(Time_save , Vt_est_save_DRT, 'r--', 'LineWidth', 1.0, 'DisplayName', 'Estimated Voltage (DRT)');
    xlabel('Time [s]');
    ylabel('Terminal Voltage [V]');
    legend('Location', 'best');
    grid on;
    
    % Subplot 3: 전류 프로파일
    subplot(3,1,3);
    plot(Time_save , trip_current, 'g', 'LineWidth', 1.0);
    xlabel('Time [s]');
    ylabel('Current [A]');
    title('Current Profile');
    grid on;
end

%% 6. 모든 트립의 결과를 하나의 Figure에 통합하여 시각화

figure('Name', 'All Trips Comparison', 'NumberTitle', 'off');

% Subplot 1: SOC 비교
subplot(3,1,1);
hold on;
for trip_num = 1:num_trips-1
    if trip_num == 1
        % True SOC: 검은색 실선
        plot(all_time{trip_num}, all_SOC_true{trip_num} * 100, 'k-', 'LineWidth', 1, 'DisplayName', 'True SOC');

        % HPPC 기반 추정 SOC: 파란색 점선
        plot(all_time{trip_num}, all_SOC_HPPC{trip_num} * 100, 'b--', 'LineWidth', 1, 'DisplayName', 'Estimated SOC (HPPC)');
        
        % DRT 기반 추정 SOC: 빨간색 점선
        plot(all_time{trip_num}, all_SOC_DRT{trip_num} * 100, 'r--', 'LineWidth', 1, 'DisplayName', 'Estimated SOC (DRT)');
        
    else
        % True SOC: 검은색 실선 (Legend에 표시되지 않도록)
        plot(all_time{trip_num}, all_SOC_true{trip_num} * 100, 'k-', 'LineWidth', 1, 'HandleVisibility', 'off');

        % HPPC 기반 추정 SOC: 파란색 점선 (Legend에 표시되지 않도록)
        plot(all_time{trip_num}, all_SOC_HPPC{trip_num} * 100, 'b--', 'LineWidth', 1, 'HandleVisibility', 'off');
        
        % DRT 기반 추정 SOC: 빨간색 점선 (Legend에 표시되지 않도록)
        plot(all_time{trip_num}, all_SOC_DRT{trip_num} * 100, 'r--', 'LineWidth', 1, 'HandleVisibility', 'off');
        
    end
end
xlabel('Time [s]');
ylabel('SOC [%]');
title('All Trips: SOC Estimation Comparison');
legend('Location', 'best');
grid on;
hold off;

% Subplot 2: 터미널 전압 비교
subplot(3,1,2);
hold on;
for trip_num = 1:num_trips-1
    if trip_num == 1
        % Measured Voltage: 검은색 실선
        plot(all_time{trip_num}, all_Vt_meas{trip_num}, 'k-', 'LineWidth', 1, 'DisplayName', 'Measured Voltage');

        % HPPC 기반 추정 전압: 파란색 점선
        plot(all_time{trip_num}, all_Vt_HPPC{trip_num}, 'b--', 'LineWidth', 1, 'DisplayName', 'Estimated Voltage (HPPC)');
        
        % DRT 기반 추정 전압: 빨간색 점선
        plot(all_time{trip_num}, all_Vt_DRT{trip_num}, 'r--', 'LineWidth', 1, 'DisplayName', 'Estimated Voltage (DRT)');
        
      
    else
        % Measured Voltage: 검은색 실선 (Legend에 표시되지 않도록)
        plot(all_time{trip_num}, all_Vt_meas{trip_num}, 'k-', 'LineWidth', 1, 'HandleVisibility', 'off');
        
        % HPPC 기반 추정 전압: 파란색 점선 (Legend에 표시되지 않도록)
        plot(all_time{trip_num}, all_Vt_HPPC{trip_num}, 'b--', 'LineWidth', 1, 'HandleVisibility', 'off');
        
        % DRT 기반 추정 전압: 빨간색 점선 (Legend에 표시되지 않도록)
        plot(all_time{trip_num}, all_Vt_DRT{trip_num}, 'r--', 'LineWidth', 1, 'HandleVisibility', 'off');
        
        
    end
end
xlabel('Time [s]');
ylabel('Terminal Voltage [V]');
title('All Trips: Terminal Voltage Comparison');
legend('Location', 'best');
grid on;
hold off;

% Subplot 3: 전류 프로파일
subplot(3,1,3);
hold on;
for trip_num = 1:num_trips-1
    plot(all_time{trip_num}, all_current{trip_num}, 'g-', 'LineWidth', 1);
end
xlabel('Time [s]');
ylabel('Current [A]');
title('All Trips: Current Profile');
legend('Current Profile', 'Location', 'best'); % 전류 프로파일에도 Legend 추가
grid on;
hold off;


%% 7. 추가 저장 (필요 시)
% save('combined_kalman_filter_results.mat', 'all_SOC_true', 'all_SOC_HPPC', 'all_SOC_DRT', ...
%     'all_Vt_meas', 'all_Vt_HPPC', 'all_Vt_DRT', 'all_time', 'all_current');
