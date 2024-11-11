clc; clear; close all;

%% 시드 설정
rng(1);

%% Font size settings
axisFontSize = 14;
titleFontSize = 16;
legendFontSize = 12;
labelFontSize = 14;

%% 1. 데이터 로드

% ECM 파라미터 (HPPC 테스트로부터)
load('optimized_params_struct_final_ver2_2RC.mat'); % 필드: R0, R1, C1, R2, C2, SOC, avgI, m, Crate

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
n = 21; % noise vector 갯수
noise_percent = 0.05;

% OCV 중복 값 제거 (보간을 위해)
[unique_ocv_values, unique_idx] = unique(ocv_values);
unique_soc_values = soc_values(unique_idx);

%% 3. ECM 파라미터 준비 (HPPC 기반)

% 목표 C-rate 설정 (0.5C)
Crate_target = 0.5;

% HPPC로부터 얻은 ECM 파라미터 추출
SOC_param_all = [optimized_params_struct_final_ver2_2RC.SOC];
R0_param_all = [optimized_params_struct_final_ver2_2RC.R0];
R1_param_all = [optimized_params_struct_final_ver2_2RC.R1];
C1_param_all = [optimized_params_struct_final_ver2_2RC.C1];
R2_param_all = [optimized_params_struct_final_ver2_2RC.R2];
C2_param_all = [optimized_params_struct_final_ver2_2RC.C2];
Crate_param_all = [optimized_params_struct_final_ver2_2RC.Crate];

% C-rate가 0.5C인 데이터 인덱스 찾기 (부동 소수점 오차 고려)
tolerance = 1e-3;
indices_Crate_05 = abs(Crate_param_all - Crate_target) < tolerance;

% C-rate가 0.5C인 경우의 SOC와 ECM 파라미터 추출
SOC_param = SOC_param_all(indices_Crate_05);
R0_param = R0_param_all(indices_Crate_05);
R1_param = R1_param_all(indices_Crate_05);
C1_param = C1_param_all(indices_Crate_05);
R2_param = R2_param_all(indices_Crate_05);
C2_param = C2_param_all(indices_Crate_05);

% SOC_param에서 중복된 SOC 값 제거 및 해당 파라미터 값 평균화
[SOC_param_unique, ~, idx_unique] = unique(SOC_param);

% 중복된 SOC 값들에 대한 파라미터 값들을 평균화
R0_param_unique = accumarray(idx_unique, R0_param, [], @mean);
R1_param_unique = accumarray(idx_unique, R1_param, [], @mean);
C1_param_unique = accumarray(idx_unique, C1_param, [], @mean);
R2_param_unique = accumarray(idx_unique, R2_param, [], @mean);
C2_param_unique = accumarray(idx_unique, C2_param, [], @mean);

% SOC에 대한 보간 함수 생성 (C-rate는 0.5C로 고정)
F_R0 = @(SOC) interp1(SOC_param_unique, R0_param_unique, SOC, 'linear', 'extrap');
F_R1 = @(SOC) interp1(SOC_param_unique, R1_param_unique, SOC, 'linear', 'extrap');
F_C1 = @(SOC) interp1(SOC_param_unique, C1_param_unique, SOC, 'linear', 'extrap');
F_R2 = @(SOC) interp1(SOC_param_unique, R2_param_unique, SOC, 'linear', 'extrap');
F_C2 = @(SOC) interp1(SOC_param_unique, C2_param_unique, SOC, 'linear', 'extrap');

%% 4. 칼만 필터 설정 - HPPC (1-RC 및 2-RC 모델)

% 초기 공분산 행렬 (1-RC 및 2-RC 모델)
P_init_HPPC_1RC = [1e-3 0;
                   0 1e-3]; % [SOC; V1]

P_init_HPPC_2RC = [1e-3 0    0;
                   0    1e-3 0;
                   0    0    1e-3]; % [SOC; V1; V2]

% 프로세스 잡음 공분산 (1-RC 및 2-RC 모델)
Q_HPPC_1RC = [1e-6 0;
             0 0.1e-7];

Q_HPPC_2RC = [1e-6 0      0;
             0 0.1e-7 0;
             0 0     0.1e-7];

% 측정 잡음 공분산 (1-RC 및 2-RC 모델)
R_HPPC = 5.25e-2; % 측정 잡음 특성에 따라 조정

%% 5. 칼만 필터 설정 - DRT

num_RC = length(tau_discrete); % RC 소자의 개수 (DRT 기반)
state_dimension = 1 + num_RC; % 상태 벡터 차원: [SOC; V_RC_1; V_RC_2; ... ; V_RC_n]

% 초기 공분산 행렬 (DRT 기반)
P_init_DRT = zeros(state_dimension);
P_init_DRT(1,1) = (0.02)^2; % SOC에 대한 초기 분산
% V_RC_i에 대한 초기 분산은 모두 0으로 설정

% 프로세스 잡음 공분산 행렬 (DRT 기반)
Q_DRT = zeros(state_dimension);
Q_DRT(1,1) = 1e-14; % SOC에 대한 프로세스 잡음 분산
for i = 2:state_dimension
    Q_DRT(i,i) = (0.04)^2; % 각 V_RC_i에 대한 프로세스 잡음 분산
end

% 측정 잡음 공분산 (DRT 기반)
R_DRT = 5.25e-6; % 측정 잡음 분산

%% 7. 모든 트립에 대해 칼만 필터 적용

num_trips = length(udds_data);

% 모든 트립의 결과를 저장할 배열 초기화
all_SOC_true = cell(num_trips-1, 1);
all_SOC_CC = cell(num_trips-1, 1);
all_SOC_HPPC_1RC = cell(num_trips-1, 1);
all_SOC_HPPC_2RC = cell(num_trips-1, 1);
all_SOC_DRT = cell(num_trips-1, 1);
all_Vt_meas = cell(num_trips-1, 1);
all_Vt_HPPC_1RC = cell(num_trips-1, 1);
all_Vt_HPPC_2RC = cell(num_trips-1, 1);
all_Vt_DRT = cell(num_trips-1, 1);
all_time = cell(num_trips-1, 1);
all_current = cell(num_trips-1, 1);

% 이전 트립의 최종 상태를 저장할 변수 초기화
X_est_HPPC_1RC_prev = [];
P_HPPC_1RC_prev = [];
X_est_HPPC_2RC_prev = [];
P_HPPC_2RC_prev = [];
X_est_DRT_prev = [];
P_DRT_prev = [];

% 이전 트립의 최종 SOC 값을 저장할 변수 초기화
SOC_true_prev = [];
SOC_CC_prev = [];

% 시간 오프셋 초기화
total_time_offset = 0;

% --- Waitbar 초기화 ---
hWait = waitbar(0, 'Processing trips...'); % Waitbar 생성

% Define Color Matrix Using lines(9)
c_mat = lines(9);  % Define a color matrix with 9 distinct colors

try
    for trip_num = 1:num_trips-1
        %% Waitbar 업데이트 (백분율 포함)
        percent = (trip_num / (num_trips-1)) * 100;
        waitbar(trip_num / (num_trips-1), hWait, sprintf('Processing trip %d of %d... (%.2f%%)', trip_num, num_trips-1, percent));

        %% 7.1. 트립 데이터 추출
        trip_current = udds_data(trip_num).I;          % 전류 [A]
        trip_voltage = udds_data(trip_num).V;          % 전압 [V]
        trip_time = udds_data(trip_num).Time_duration; % 누적 시간 [s]
        trip_SOC_true = udds_data(trip_num).SOC;       % 실제 SOC (있는 경우)

        % 시작 시간을 0으로 조정하고, 시간 오프셋 적용
        trip_time = trip_time - trip_time(1); % 시작 시간을 0으로 조정
        trip_time = trip_time + total_time_offset; % 이전 트립의 종료 시간부터 시작하도록 이동

        %% 7.2. 전류에 Markov Noise 추가
        % Markov Noise 파라미터 설정
        %n = 21;
        %noise_percent = 0.05;
        initial_state = randsample(1:n, 1);

        % 전류에 노이즈 추가
        [noisy_trip_current, states_trip_current] = add_markov_noise(trip_current, n, noise_percent, initial_state);

       %% 7.3. 전압에 3% Gaussian Random Noise 추가
        voltage_noise_std = 0.03 * trip_voltage; % 각 전압 측정값의 3%를 표준 편차로 설정
        noisy_trip_voltage = trip_voltage + voltage_noise_std .* randn(size(trip_voltage));

        %% 7.4. 초기화

        % 시간 간격 계산 (DRT 기반에서 필요)
        dt = [0; diff(trip_time)];
        if dt(1) == 0
            dt(1) = dt(2);
        end

        if trip_num == 1
            % 첫 번째 트립의 경우 초기 SOC를 전압 기반으로 추정
            initial_voltage = trip_voltage(1);
            initial_soc = interp1(unique_ocv_values, unique_soc_values, initial_voltage, 'linear', 'extrap');

            % 이전 트립의 최종 SOC 값을 초기 SOC로 설정
            SOC_true_prev = initial_soc;
            SOC_CC_prev = initial_soc;

            % ECM 파라미터 보간 (초기 SOC에 대해 R1, C1, R2, C2를 추정)
            R1_initial = F_R1(initial_soc);
            C1_initial = F_C1(initial_soc);
            R2_initial = F_R2(initial_soc);
            C2_initial = F_C2(initial_soc);

            % 초기 V1_est_HPPC 및 V2_est_HPPC 계산
            dt_initial = dt(1); % 첫 시간 간격
            V1_init_HPPC = noisy_trip_current(1) * R1_initial * (1 - exp(-dt_initial / (R1_initial * C1_initial)));
            V2_init_HPPC = noisy_trip_current(1) * R2_initial * (1 - exp(-dt_initial / (R2_initial * C2_initial)));

            % 1-RC HPPC 기반 칼만 필터 초기화
            SOC_est_HPPC_1RC = initial_soc;
            V1_est_HPPC_1RC = V1_init_HPPC;
            X_est_HPPC_1RC = [SOC_est_HPPC_1RC; V1_est_HPPC_1RC];
            P_HPPC_1RC = P_init_HPPC_1RC;

            % 2-RC HPPC 기반 칼만 필터 초기화
            SOC_est_HPPC_2RC = initial_soc;
            V1_est_HPPC_2RC = V1_init_HPPC;
            V2_est_HPPC_2RC = V2_init_HPPC;
            X_est_HPPC_2RC = [SOC_est_HPPC_2RC; V1_est_HPPC_2RC; V2_est_HPPC_2RC];
            P_HPPC_2RC = P_init_HPPC_2RC;

            % DRT 기반 칼만 필터 초기화
            SOC_est_DRT = initial_soc;
            X_est_DRT = zeros(state_dimension, 1);
            X_est_DRT(1) = SOC_est_DRT;
            P_DRT = P_init_DRT;

            % 초기 V_RC_est 설정 (DRT 기반)
            gamma_current_init = interp1(soc_sorted, gamma_sorted, SOC_est_DRT, 'linear', 'extrap');
            gamma_current_init = gamma_current_init(:)'; % 1 x num_RC 벡터로 변환
            delta_theta = theta_discrete(2) - theta_discrete(1);
            R_i_init = gamma_current_init * delta_theta; % 1 x num_RC 벡터
            C_i_init = tau_discrete ./ R_i_init;         % 1 x num_RC 벡터
            V_RC_init = noisy_trip_current(1) * R_i_init .* (1 - exp(-dt_initial ./ (R_i_init .* C_i_init))); % 1 x num_RC 벡터
            X_est_DRT(2:end) = V_RC_init(:); % num_RC x 1 벡터로 변환하여 저장
        else
            % 이후 트립의 경우 이전 트립의 최종 상태를 사용
            X_est_HPPC_1RC = X_est_HPPC_1RC_prev;
            P_HPPC_1RC = P_HPPC_1RC_prev;
            X_est_HPPC_2RC = X_est_HPPC_2RC_prev;
            P_HPPC_2RC = P_HPPC_2RC_prev;
            X_est_DRT = X_est_DRT_prev;
            P_DRT = P_DRT_prev;
        end

        % 결과 저장을 위한 변수 초기화
        num_samples = length(trip_time);
        SOC_save_true = zeros(num_samples, 1);
        SOC_save_CC = zeros(num_samples, 1);
        SOC_save_HPPC_1RC = zeros(num_samples, 1);
        SOC_save_HPPC_2RC = zeros(num_samples, 1);
        SOC_save_DRT = zeros(num_samples, 1);
        Vt_est_save_HPPC_1RC = zeros(num_samples, 1);
        Vt_est_save_HPPC_2RC = zeros(num_samples, 1);
        Vt_est_save_DRT = zeros(num_samples, 1);
        Vt_meas_save = trip_voltage;
        Time_save = trip_time;
        trip_current = trip_current(:); % 열 벡터로 변환

        % 초기값 저장
        if trip_num == 1
            SOC_save_true(1) = SOC_true_prev; % 초기 SOC (첫 번째 트립)
            SOC_save_CC(1) = SOC_CC_prev;     % 초기 SOC (첫 번째 트립)
        else
            SOC_save_true(1) = SOC_true_prev; % 이전 트립의 최종 SOC를 사용
            SOC_save_CC(1) = SOC_CC_prev;     % 이전 트립의 최종 SOC를 사용
        end

        % 1-RC HPPC 초기 상태 저장
        SOC_save_HPPC_1RC(1) = X_est_HPPC_1RC(1);
        OCV_initial_HPPC_1RC = interp1(unique_soc_values, unique_ocv_values, X_est_HPPC_1RC(1), 'linear', 'extrap');
        Vt_est_save_HPPC_1RC(1) = OCV_initial_HPPC_1RC + X_est_HPPC_1RC(2) + F_R0(X_est_HPPC_1RC(1)) * noisy_trip_current(1);

        % 2-RC HPPC 초기 상태 저장
        SOC_save_HPPC_2RC(1) = X_est_HPPC_2RC(1);
        OCV_initial_HPPC_2RC = interp1(unique_soc_values, unique_ocv_values, X_est_HPPC_2RC(1), 'linear', 'extrap');
        Vt_est_save_HPPC_2RC(1) = OCV_initial_HPPC_2RC + X_est_HPPC_2RC(2) + X_est_HPPC_2RC(3) + F_R0(X_est_HPPC_2RC(1)) * noisy_trip_current(1);

        % DRT 초기 상태 저장
        SOC_save_DRT(1) = X_est_DRT(1);
        OCV_initial_DRT = interp1(unique_soc_values, unique_ocv_values, X_est_DRT(1), 'linear', 'extrap');
        Vt_est_save_DRT(1) = OCV_initial_DRT + sum(X_est_DRT(2:end)) + R0_est_all(trip_num) * noisy_trip_current(1);

        %% 7.5. 메인 루프

        for k = 2:num_samples
            % 공통 계산
            dt_k = trip_time(k) - trip_time(k-1); % 시간 간격 [s]
            ik = trip_current(k); % 실제 전류 [A]
            noisy_ik = noisy_trip_current(k); % 노이즈가 추가된 전류 [A]
            vk = trip_voltage(k); % 실제 전압 [V]
            noisy_vk = noisy_trip_voltage(k); % 노이즈가 추가된 전압 [V]

            %% True SOC 계산 (쿨롱 카운팅)
            SOC_true = SOC_save_true(k-1) + (dt_k / (Config.cap * 3600)) * ik * Config.coulomb_efficiency;
            SOC_save_true(k) = SOC_true;

            %% CC SOC 계산 (노이즈가 추가된 전류로 쿨롱 카운팅)
            SOC_CC = SOC_save_CC(k-1) + (dt_k / (Config.cap * 3600)) * noisy_ik * Config.coulomb_efficiency;
            SOC_save_CC(k) = SOC_CC;

            %% 1-RC HPPC 기반 칼만 필터 업데이트

            % SOC 예측 (쿨롱 카운팅)
            SOC_pred_HPPC_1RC = X_est_HPPC_1RC(1) + (dt_k / (Config.cap * 3600)) * Config.coulomb_efficiency * noisy_ik;

            % ECM 파라미터 보간 (SOC에 대해서만)
            R0_interp = F_R0(SOC_pred_HPPC_1RC);
            R1_interp = F_R1(SOC_pred_HPPC_1RC);
            C1_interp = F_C1(SOC_pred_HPPC_1RC);

            % 상태 천이 행렬 및 입력 행렬
            A_k_HPPC_1RC = [1, 0;
                            0, exp(-dt_k / (R1_interp * C1_interp))];
            B_k_HPPC_1RC = [-(dt_k / (Config.cap * 3600)) * Config.coulomb_efficiency;
                             R1_interp * (1 - exp(-dt_k / (R1_interp * C1_interp)))];

            % 상태 예측
            X_pred_HPPC_1RC = A_k_HPPC_1RC * X_est_HPPC_1RC + B_k_HPPC_1RC * noisy_ik;
            SOC_pred_HPPC_1RC = X_pred_HPPC_1RC(1);
            V1_pred_HPPC_1RC = X_pred_HPPC_1RC(2);

            % 공분산 예측
            P_predict_HPPC_1RC = A_k_HPPC_1RC * P_HPPC_1RC * A_k_HPPC_1RC' + Q_HPPC_1RC;

            % 전압 예측
            OCV_pred_HPPC_1RC = interp1(unique_soc_values, unique_ocv_values, SOC_pred_HPPC_1RC, 'linear', 'extrap');
            Vt_pred_HPPC_1RC = OCV_pred_HPPC_1RC + V1_pred_HPPC_1RC + R0_interp * noisy_ik;

            % 관측 행렬 H 계산
            delta_SOC = 1e-5;
            OCV_plus = interp1(unique_soc_values, unique_ocv_values, SOC_pred_HPPC_1RC + delta_SOC, 'linear', 'extrap');
            OCV_minus = interp1(unique_soc_values, unique_ocv_values, SOC_pred_HPPC_1RC - delta_SOC, 'linear', 'extrap');
            dOCV_dSOC = (OCV_plus - OCV_minus) / (2 * delta_SOC);
            H_k_HPPC_1RC = [dOCV_dSOC, 1];

            % 잔차 계산 (노이즈가 추가된 전압 사용)
            y_tilde_HPPC_1RC = noisy_vk - Vt_pred_HPPC_1RC;

            % 칼만 이득 계산
            S_k_HPPC_1RC = H_k_HPPC_1RC * P_predict_HPPC_1RC * H_k_HPPC_1RC' + R_HPPC;
            K_HPPC_1RC = (P_predict_HPPC_1RC * H_k_HPPC_1RC') / S_k_HPPC_1RC;

            % 상태 업데이트
            X_est_HPPC_1RC = X_pred_HPPC_1RC + K_HPPC_1RC * y_tilde_HPPC_1RC;

            % 공분산 업데이트
            P_HPPC_1RC = (eye(2) - K_HPPC_1RC * H_k_HPPC_1RC) * P_predict_HPPC_1RC;

            % 전압 업데이트
            OCV_updated_HPPC_1RC = interp1(unique_soc_values, unique_ocv_values, X_est_HPPC_1RC(1), 'linear', 'extrap');
            Vt_est_HPPC_1RC = OCV_updated_HPPC_1RC + X_est_HPPC_1RC(2) + F_R0(X_est_HPPC_1RC(1)) * noisy_ik;

            % 결과 저장
            SOC_save_HPPC_1RC(k) = X_est_HPPC_1RC(1);
            Vt_est_save_HPPC_1RC(k) = Vt_est_HPPC_1RC;

            %% 2-RC HPPC 기반 칼만 필터 업데이트

            % SOC 예측 (쿨롱 카운팅)
            SOC_pred_HPPC_2RC = X_est_HPPC_2RC(1) + (dt_k / (Config.cap * 3600)) * Config.coulomb_efficiency * noisy_ik;

            % ECM 파라미터 보간 (SOC에 대해서만)
            R0_interp_2RC = F_R0(SOC_pred_HPPC_2RC);
            R1_interp_2RC = F_R1(SOC_pred_HPPC_2RC);
            C1_interp_2RC = F_C1(SOC_pred_HPPC_2RC);
            R2_interp_2RC = F_R2(SOC_pred_HPPC_2RC);
            C2_interp_2RC = F_C2(SOC_pred_HPPC_2RC);

            % 상태 천이 행렬 및 입력 행렬
            A_k_HPPC_2RC = eye(3);
            A_k_HPPC_2RC(2,2) = exp(-dt_k / (R1_interp_2RC * C1_interp_2RC));
            A_k_HPPC_2RC(3,3) = exp(-dt_k / (R2_interp_2RC * C2_interp_2RC));

            B_k_HPPC_2RC = [-(dt_k / (Config.cap * 3600)) * Config.coulomb_efficiency;
                             R1_interp_2RC * (1 - exp(-dt_k / (R1_interp_2RC * C1_interp_2RC)));
                             R2_interp_2RC * (1 - exp(-dt_k / (R2_interp_2RC * C2_interp_2RC)))];

            % 상태 예측
            X_pred_HPPC_2RC = A_k_HPPC_2RC * X_est_HPPC_2RC + B_k_HPPC_2RC * noisy_ik;
            SOC_pred_HPPC_2RC = X_pred_HPPC_2RC(1);
            V1_pred_HPPC_2RC = X_pred_HPPC_2RC(2);
            V2_pred_HPPC_2RC = X_pred_HPPC_2RC(3);

            % 공분산 예측
            P_predict_HPPC_2RC = A_k_HPPC_2RC * P_HPPC_2RC * A_k_HPPC_2RC' + Q_HPPC_2RC;

            % 전압 예측
            OCV_pred_HPPC_2RC = interp1(unique_soc_values, unique_ocv_values, SOC_pred_HPPC_2RC, 'linear', 'extrap');
            Vt_pred_HPPC_2RC = OCV_pred_HPPC_2RC + V1_pred_HPPC_2RC + V2_pred_HPPC_2RC + R0_interp_2RC * noisy_ik;

            % 관측 행렬 H 계산
            delta_SOC_2RC = 1e-5;
            OCV_plus_2RC = interp1(unique_soc_values, unique_ocv_values, SOC_pred_HPPC_2RC + delta_SOC_2RC, 'linear', 'extrap');
            OCV_minus_2RC = interp1(unique_soc_values, unique_ocv_values, SOC_pred_HPPC_2RC - delta_SOC_2RC, 'linear', 'extrap');
            dOCV_dSOC_2RC = (OCV_plus_2RC - OCV_minus_2RC) / (2 * delta_SOC_2RC);
            H_k_HPPC_2RC = [dOCV_dSOC_2RC, 1, 1];

            % 잔차 계산 (노이즈가 추가된 전압 사용)
            y_tilde_HPPC_2RC = noisy_vk - Vt_pred_HPPC_2RC;

            % 칼만 이득 계산
            S_k_HPPC_2RC = H_k_HPPC_2RC * P_predict_HPPC_2RC * H_k_HPPC_2RC' + R_HPPC;
            K_HPPC_2RC = (P_predict_HPPC_2RC * H_k_HPPC_2RC') / S_k_HPPC_2RC;

            % 상태 업데이트
            X_est_HPPC_2RC = X_pred_HPPC_2RC + K_HPPC_2RC * y_tilde_HPPC_2RC;

            % 공분산 업데이트
            P_HPPC_2RC = (eye(3) - K_HPPC_2RC * H_k_HPPC_2RC) * P_predict_HPPC_2RC;

            % 전압 업데이트
            OCV_updated_HPPC_2RC = interp1(unique_soc_values, unique_ocv_values, X_est_HPPC_2RC(1), 'linear', 'extrap');
            Vt_est_HPPC_2RC = OCV_updated_HPPC_2RC + X_est_HPPC_2RC(2) + X_est_HPPC_2RC(3) + F_R0(X_est_HPPC_2RC(1)) * noisy_ik;

            % 결과 저장
            SOC_save_HPPC_2RC(k) = X_est_HPPC_2RC(1);
            Vt_est_save_HPPC_2RC(k) = Vt_est_HPPC_2RC;

            %% DRT 기반 칼만 필터 업데이트

            % SOC 예측 (쿨롱 카운팅)
            SOC_pred_DRT = X_est_DRT(1) + (dt_k / (Config.cap * 3600)) * noisy_ik;
            SOC_pred_DRT = max(0, min(1, SOC_pred_DRT));

            % 현재 SOC에 대한 gamma 값 보간
            gamma_current = interp1(soc_sorted, gamma_sorted, SOC_pred_DRT, 'linear', 'extrap');
            gamma_current = gamma_current(:)'; % 1 x num_RC 벡터로 변환

            % 각 RC 소자에 대한 R_i 및 C_i 계산
            delta_theta = theta_discrete(2) - theta_discrete(1);
            R_i = gamma_current * delta_theta; % 1 x num_RC 벡터
            C_i = tau_discrete ./ R_i;         % 1 x num_RC 벡터

            % 상태 천이 행렬 A_DRT 계산
            A_DRT = diag([1; exp(-dt_k ./ tau_discrete(:))]);

            % RC 전압 예측
            exp_term = exp(-dt_k ./ (R_i .* C_i)); % 1 x num_RC 벡터
            V_RC_pred = exp_term .* X_est_DRT(2:end)' + (R_i .* (1 - exp_term)) * noisy_ik; % 1 x num_RC 벡터

            % 상태 예측
            X_pred_DRT = [SOC_pred_DRT; V_RC_pred'];

            % 공분산 예측
            P_pred_DRT = A_DRT * P_DRT * A_DRT' + Q_DRT;

            % 전압 예측
            OCV_pred_DRT = interp1(unique_soc_values, unique_ocv_values, SOC_pred_DRT, 'linear', 'extrap');
            Vt_pred_DRT = OCV_pred_DRT + R0_est_all(trip_num) * noisy_ik + sum(V_RC_pred);

            % 관측 행렬 H_DRT 계산
            delta_SOC_DRT = 1e-10;
            OCV_plus_DRT = interp1(unique_soc_values, unique_ocv_values, SOC_pred_DRT + delta_SOC_DRT, 'linear', 'extrap');
            OCV_minus_DRT = interp1(unique_soc_values, unique_ocv_values, SOC_pred_DRT - delta_SOC_DRT, 'linear', 'extrap');
            dOCV_dSOC_DRT = (OCV_plus_DRT - OCV_minus_DRT) / (2 * delta_SOC_DRT);

            H_DRT = zeros(1, state_dimension);
            H_DRT(1) = dOCV_dSOC_DRT;
            H_DRT(2:end) = 1;

            % 잔차 계산 (노이즈가 추가된 전압 사용)
            y_tilde_DRT = noisy_vk - Vt_pred_DRT;

            % 칼만 이득 계산
            S_DRT = H_DRT * P_pred_DRT * H_DRT' + R_DRT;
            K_DRT = (P_pred_DRT * H_DRT') / S_DRT;

            % 상태 업데이트
            X_est_DRT = X_pred_DRT + K_DRT * y_tilde_DRT;

            % 공분산 업데이트
            P_DRT = (eye(state_dimension) - K_DRT * H_DRT) * P_pred_DRT;

            % 전압 업데이트
            OCV_updated_DRT = interp1(unique_soc_values, unique_ocv_values, X_est_DRT(1), 'linear', 'extrap');
            Vt_est_DRT = OCV_updated_DRT + R0_est_all(trip_num) * noisy_ik + sum(X_est_DRT(2:end));

            % 결과 저장
            SOC_save_DRT(k) = X_est_DRT(1);
            Vt_est_save_DRT(k) = Vt_est_DRT;
        end

        %% 7.6. 결과 저장 및 시각화

        % 트립 결과를 셀 배열에 저장
        all_SOC_true{trip_num} = SOC_save_true;
        all_SOC_CC{trip_num} = SOC_save_CC;
        all_SOC_HPPC_1RC{trip_num} = SOC_save_HPPC_1RC;
        all_SOC_HPPC_2RC{trip_num} = SOC_save_HPPC_2RC;
        all_SOC_DRT{trip_num} = SOC_save_DRT;
        all_Vt_meas{trip_num} = Vt_meas_save;
        all_Vt_HPPC_1RC{trip_num} = Vt_est_save_HPPC_1RC;
        all_Vt_HPPC_2RC{trip_num} = Vt_est_save_HPPC_2RC;
        all_Vt_DRT{trip_num} = Vt_est_save_DRT;
        all_time{trip_num} = Time_save;
        all_current{trip_num} = trip_current;

        % 결과 시각화 (개별 트립)
        figure('Name', sprintf('Trip %d Results', trip_num), 'NumberTitle', 'off');

        % Subplot 1: SOC 비교
        subplot(4,1,1);
        hold on;
        plot(Time_save, SOC_save_true * 100, 'Color', c_mat(1, :), 'LineWidth', 1.5, 'DisplayName', 'True SOC');
        plot(Time_save, SOC_save_CC * 100, 'Color', c_mat(2, :), 'LineWidth', 1.5, 'DisplayName', 'CC SOC');
        plot(Time_save, SOC_save_HPPC_1RC * 100, '--', 'Color', c_mat(3, :), 'LineWidth', 1.5, 'DisplayName', 'N-RC SOC (HPPC 1RC)');
        plot(Time_save, SOC_save_HPPC_2RC * 100, '--', 'Color', c_mat(4, :), 'LineWidth', 1.5, 'DisplayName', 'N-RC SOC (HPPC 2RC)');
        plot(Time_save, SOC_save_DRT * 100, '--', 'Color', c_mat(5, :), 'LineWidth', 1.5, 'DisplayName', 'DRT SOC');
        xlabel('Time [s]', 'FontSize', labelFontSize);
        ylabel('SOC [%]', 'FontSize', labelFontSize);
        title(sprintf('Trip %d: SOC Estimation using Kalman Filter', trip_num), 'FontSize', titleFontSize);
        legend('Location', 'best', 'FontSize', legendFontSize);
        grid on;
        hold off;

        % Subplot 2: 터미널 전압 비교
        subplot(4,1,2);
        hold on;
        plot(Time_save, Vt_meas_save, 'Color', c_mat(1, :), 'LineWidth', 1.0, 'DisplayName', 'Measured Voltage');
        plot(Time_save, Vt_est_save_HPPC_1RC, '--', 'Color', c_mat(3, :), 'LineWidth', 1.0, 'DisplayName', 'Estimated Voltage (N-RC 1RC)');
        plot(Time_save, Vt_est_save_HPPC_2RC, '--', 'Color', c_mat(4, :), 'LineWidth', 1.0, 'DisplayName', 'Estimated Voltage (N-RC 2RC)');
        plot(Time_save, Vt_est_save_DRT, '--', 'Color', c_mat(5, :), 'LineWidth', 1.0, 'DisplayName', 'Estimated Voltage (DRT)');
        xlabel('Time [s]', 'FontSize', labelFontSize);
        ylabel('Terminal Voltage [V]', 'FontSize', labelFontSize);
        title(sprintf('Trip %d: Voltage Estimation', trip_num), 'FontSize', titleFontSize);
        legend('Location', 'best', 'FontSize', legendFontSize);
        grid on;
        hold off;

        % Subplot 3: 전류 프로파일
        subplot(4,1,3);
        plot(Time_save, trip_current, 'Color', c_mat(6, :), 'LineWidth', 1.5, 'DisplayName', 'Current');
        xlabel('Time [s]', 'FontSize', labelFontSize);
        ylabel('Current [A]', 'FontSize', labelFontSize);
        title('Current Profile', 'FontSize', titleFontSize);
        grid on;

        % Subplot 4: 2-RC HPPC 상태
        subplot(4,1,4);
        hold on;
        plot(Time_save, X_est_HPPC_2RC(2) * ones(size(Time_save)), 'Color', c_mat(7, :), 'LineWidth', 1.0, 'DisplayName', 'V1 (2RC HPPC)');
        plot(Time_save, X_est_HPPC_2RC(3) * ones(size(Time_save)), 'Color', c_mat(8, :), 'LineWidth', 1.0, 'DisplayName', 'V2 (2RC HPPC)');
        xlabel('Time [s]', 'FontSize', labelFontSize);
        ylabel('RC Voltages [V]', 'FontSize', labelFontSize);
        title('2-RC HPPC RC Voltages', 'FontSize', titleFontSize);
        legend('Location', 'best', 'FontSize', legendFontSize);
        grid on;
        hold off;

        % 이전 트립의 최종 상태를 저장하여 다음 트립의 초기 상태로 사용
        X_est_HPPC_1RC_prev = X_est_HPPC_1RC;
        P_HPPC_1RC_prev = P_HPPC_1RC;
        X_est_HPPC_2RC_prev = X_est_HPPC_2RC;
        P_HPPC_2RC_prev = P_HPPC_2RC;
        X_est_DRT_prev = X_est_DRT;
        P_DRT_prev = P_DRT;

        % 이전 트립의 최종 SOC 값을 저장
        SOC_true_prev = SOC_save_true(end);
        SOC_CC_prev = SOC_save_CC(end);

        % 시간 오프셋 업데이트
        total_time_offset = trip_time(end); % 이전 트립의 종료 시간을 저장
    end
catch ME
    % 에러 발생 시 Waitbar 닫기
    close(hWait);
    rethrow(ME);  % 에러 다시 발생
end

% --- Waitbar 닫기 ---
close(hWait);

%% 8. 모든 트립의 결과를 하나의 Figure에 통합하여 시각화

% 8.1. 모든 트립의 데이터를 하나의 연속된 배열로 결합
all_time_concat = cell2mat(all_time);
all_SOC_true_concat = cell2mat(all_SOC_true);
all_SOC_CC_concat = cell2mat(all_SOC_CC);
all_SOC_HPPC_1RC_concat = cell2mat(all_SOC_HPPC_1RC);
all_SOC_HPPC_2RC_concat = cell2mat(all_SOC_HPPC_2RC);
all_SOC_DRT_concat = cell2mat(all_SOC_DRT);
all_Vt_meas_concat = cell2mat(all_Vt_meas);
all_Vt_HPPC_1RC_concat = cell2mat(all_Vt_HPPC_1RC);
all_Vt_HPPC_2RC_concat = cell2mat(all_Vt_HPPC_2RC);
all_Vt_DRT_concat = cell2mat(all_Vt_DRT);
all_current_concat = cell2mat(all_current);

% 8.2. Figure 생성
figure('Name', 'All Trips Comparison', 'NumberTitle', 'off');

% Subplot 1: SOC 비교
subplot(4,1,1);
hold on;
plot(all_time_concat, all_SOC_true_concat * 100, 'Color', c_mat(1, :), 'LineWidth', 1.5, 'DisplayName', 'True SOC');
plot(all_time_concat, all_SOC_CC_concat * 100, 'Color', c_mat(2, :), 'LineWidth', 1.5, 'DisplayName', 'CC SOC');
plot(all_time_concat, all_SOC_HPPC_1RC_concat * 100, '--', 'Color', c_mat(3, :), 'LineWidth', 1.5, 'DisplayName', 'N-RC SOC (HPPC 1RC)');
plot(all_time_concat, all_SOC_HPPC_2RC_concat * 100, '--', 'Color', c_mat(4, :), 'LineWidth', 1.5, 'DisplayName', 'N-RC SOC (HPPC 2RC)');
plot(all_time_concat, all_SOC_DRT_concat * 100, '--', 'Color', c_mat(5, :), 'LineWidth', 1.5, 'DisplayName', 'DRT SOC');
xlabel('Time [s]', 'FontSize', labelFontSize);
ylabel('SOC [%]', 'FontSize', labelFontSize);
title('All Trips: SOC Estimation using Kalman Filter', 'FontSize', titleFontSize);
legend('Location', 'best', 'FontSize', legendFontSize);
grid on;
hold off;

% Subplot 2: 터미널 전압 비교
subplot(4,1,2);
hold on;
plot(all_time_concat, all_Vt_meas_concat, 'Color', c_mat(1, :), 'LineWidth', 1.0, 'DisplayName', 'Measured Voltage');
plot(all_time_concat, all_Vt_HPPC_1RC_concat, '--', 'Color', c_mat(3, :), 'LineWidth', 1.0, 'DisplayName', 'Estimated Voltage (N-RC 1RC)');
plot(all_time_concat, all_Vt_HPPC_2RC_concat, '--', 'Color', c_mat(4, :), 'LineWidth', 1.0, 'DisplayName', 'Estimated Voltage (N-RC 2RC)');
plot(all_time_concat, all_Vt_DRT_concat, '--', 'Color', c_mat(5, :), 'LineWidth', 1.0, 'DisplayName', 'Estimated Voltage (DRT)');
xlabel('Time [s]', 'FontSize', labelFontSize);
ylabel('Terminal Voltage [V]', 'FontSize', labelFontSize);
title('All Trips: Voltage Estimation', 'FontSize', titleFontSize);
legend('Location', 'best', 'FontSize', legendFontSize);
grid on;
hold off;

% Subplot 3: 전류 프로파일
subplot(4,1,3);
plot(all_time_concat, all_current_concat, 'Color', c_mat(6, :), 'LineWidth', 1.5, 'DisplayName', 'Current');
xlabel('Time [s]', 'FontSize', labelFontSize);
ylabel('Current [A]', 'FontSize', labelFontSize);
title('All Trips: Current Profile', 'FontSize', titleFontSize);
grid on;

% Subplot 4: 2-RC HPPC 상태
subplot(4,1,4);
hold on;
plot(all_time_concat, all_SOC_HPPC_2RC_concat * 100, 'Color', c_mat(3, :), 'LineWidth', 1.5, 'DisplayName', 'V1 (HPPC 1RC)');
plot(all_time_concat, all_SOC_HPPC_2RC_concat * 100, 'Color', c_mat(4, :), 'LineWidth', 1.5, 'DisplayName', 'V2 (HPPC 2RC)');
xlabel('Time [s]', 'FontSize', labelFontSize);
ylabel('RC Voltages [V]', 'FontSize', labelFontSize);
title('All Trips: 2-RC HPPC RC Voltages', 'FontSize', titleFontSize);
legend('Location', 'best', 'FontSize', legendFontSize);
grid on;
hold off;

%% 9. 추가 시각화

% 9.1. 전체 SOC 그래프
figure('Name', 'Entire SOC Comparison', 'NumberTitle', 'off');
hold on;
plot(all_time_concat, all_SOC_true_concat * 100, 'Color', c_mat(1, :), 'LineWidth', 3, 'DisplayName', 'True SOC'); 
plot(all_time_concat, all_SOC_CC_concat * 100, 'Color', c_mat(2, :), 'LineWidth', 3, 'DisplayName', 'CC SOC'); 
plot(all_time_concat, all_SOC_HPPC_1RC_concat * 100, '--', 'Color', c_mat(3, :), 'LineWidth', 3, 'DisplayName', 'N-RC SOC (HPPC 1RC)'); 
plot(all_time_concat, all_SOC_HPPC_2RC_concat * 100, '--', 'Color', c_mat(4, :), 'LineWidth', 3, 'DisplayName', 'N-RC SOC (HPPC 2RC)'); 
plot(all_time_concat, all_SOC_DRT_concat * 100, '--', 'Color', c_mat(5, :), 'LineWidth', 3, 'DisplayName', 'DRT SOC'); 
xlabel('Time [s]', 'FontSize', labelFontSize);
ylabel('SOC [%]', 'FontSize', labelFontSize);
title('All Trips: SOC Estimation', 'FontSize', titleFontSize);
legend('Location', 'best', 'FontSize', legendFontSize);
grid on;
hold off;

% 9.2. 첫 번째 트립의 SOC 그래프
figure('Name', 'Trip 1 SOC Estimation', 'NumberTitle', 'off'); % 원하는 크기로 조정
hold on;
plot(all_time{1}, all_SOC_true{1} * 100, 'Color', c_mat(1, :), 'LineWidth', 3, 'DisplayName', 'True SOC'); 
plot(all_time{1}, all_SOC_CC{1} * 100, 'Color', c_mat(2, :), 'LineWidth', 3, 'DisplayName', 'CC SOC'); 
plot(all_time{1}, all_SOC_HPPC_1RC{1} * 100, '--', 'Color', c_mat(3, :), 'LineWidth', 3, 'DisplayName', 'N-RC SOC (HPPC 1RC)');
plot(all_time{1}, all_SOC_HPPC_2RC{1} * 100, '--', 'Color', c_mat(4, :), 'LineWidth', 3, 'DisplayName', 'N-RC SOC (HPPC 2RC)');
plot(all_time{1}, all_SOC_DRT{1} * 100, '--', 'Color', c_mat(5, :), 'LineWidth', 3, 'DisplayName', 'DRT SOC'); 
xlabel('Time [s]', 'FontSize', labelFontSize);
ylabel('SOC [%]', 'FontSize', labelFontSize);
title('Trip 1: SOC Estimation', 'FontSize', titleFontSize);
legend('Location', 'best', 'FontSize', legendFontSize);
grid on;
hold off;

%% 6. Markov Chain Noise를 추가하기 위한 함수 정의

function [noisy_I, states] = add_markov_noise(I_original, n, noise_percent, initial_state)
    % add_markov_noise - 전류 데이터에 마르코프 체인 기반 노이즈를 추가하는 함수
    %
    % 입력:
    %   I_original    - 원본 전류 데이터 (벡터)
    %   n             - 마르코프 체인의 상태 수 (정수)
    %   noise_percent - 노이즈의 스케일링 계수 (실수)
    %   initial_state - 초기 상태 (1부터 n 사이의 정수)
    %
    % 출력:
    %   noisy_I       - 노이즈가 추가된 전류 데이터 (벡터)
    %   states        - 각 시간에서의 마르코프 상태 (벡터)

    % 평균 전류 계산
    mean_I = mean(I_original);
    mean_noise = mean_I * noise_percent;

    % 최소 및 최대 노이즈 계산
    min_noise = min(I_original) * noise_percent;
    max_noise = max(I_original) * noise_percent;
    span = max_noise - min_noise;

    noise_vector = linspace(mean_noise - span/2, mean_noise + span/2, n); % 노이즈 벡터 중간값 = mean_noise로 설정

    % sigma 계산 (노이즈 분포의 표준 편차)
    sigma = span / 50;

    % 전이 확률 행렬 생성
    P = zeros(n);

    % 각 상태에 대한 전이 확률 계산
    for i = 1:n
        probabilities = normpdf(noise_vector, noise_vector(i), sigma);
        P(i, :) = probabilities / sum(probabilities);
    end

    % 노이즈 추가 및 상태 기록을 위한 초기화
    noisy_I = zeros(size(I_original));
    states = zeros(size(I_original));
    current_state = initial_state;

    for idx = 1:length(I_original)
        % 현재 상태의 노이즈를 전류에 추가
        noisy_I(idx) = I_original(idx) + noise_vector(current_state);

        % 상태 기록
        states(idx) = current_state;

        % 다음 상태로 전이
        current_state = randsample(1:n, 1, true, P(current_state, :));
    end
end
