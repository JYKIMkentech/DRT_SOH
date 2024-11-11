%% 초기 설정
clear; clc; close all;

% 샘플링 시간 및 총 시간 설정
dt = 1;            % 샘플링 시간 (초)
total_time = 36000; % 총 시뮬레이션 시간 (10시간)
time_vector = 0:dt:total_time;  % 시간 벡터

% 배터리 용량 설정
Q_nom = 2.3 * 3600;  % 배터리 용량 (As)

% 초기 SOC 설정
initial_soc = 1;  % SOC를 1로 설정

%% [tau, R] 데이터 생성
% tau와 R 값 생성 (20개 요소)
num_RC = 20;
tau_min = 1;       % 최소 시간 상수 (초)
tau_max = 10000;    % 최대 시간 상수 (초)
tau_array = logspace(log10(tau_min), log10(tau_max), num_RC);

% 저항 R 값 생성 (임의의 값)
R_array = 0.01 + 0.02 * rand(1, num_RC);  % 0.01 ~ 0.03 옴 사이의 값

% 커패시턴스 C 값 계산
C_array = tau_array ./ R_array;

%% SOC-OCV 곡선 생성
% SOC 값 생성 (0 ~ 1 사이)
soc_values = linspace(0, 1, 100);
% OCV 값 생성 (예시로 단순한 함수 사용)
ocv_values = 3.0 + 1.2 * soc_values - 0.5 * soc_values.^2;

%% 입력 전류 생성
% 0.1C로 일정하게 방전하는 전류 프로파일 생성
C_rate = 0.1;  % C-rate 설정
I_discharge = -C_rate * (Q_nom / 3600);  % 방전 전류 (A)
current_profile = I_discharge * ones(size(time_vector));  % 전체 시간 동안 일정한 방전 전류

%% 측정 전압 생성 (시뮬레이션)
% 실제 배터리에서 측정된 것처럼 전압을 생성
% 이 때 노이즈를 추가하여 현실감 있게 만듭니다.

% 상태 변수 초기화
num_samples = length(time_vector);
Vt_true = zeros(num_samples, 1);
SOC_true = zeros(num_samples, 1);
V_RC_true = zeros(num_RC, num_samples);

SOC_true(1) = initial_soc;

% 노이즈 없는 전압 계산
for k = 1:num_samples
    if k == 1
        % 초기 V_RC 값은 0으로 설정
        V_RC_true(:, k) = 0;
    else
        % RC 회로 전압 업데이트
        for i = 1:num_RC
            V_RC_true(i, k) = exp(-dt / tau_array(i)) * V_RC_true(i, k-1) + R_array(i) * (1 - exp(-dt / tau_array(i))) * current_profile(k-1);
        end
        % SOC 업데이트
        SOC_true(k) = SOC_true(k-1) + (dt / Q_nom) * current_profile(k-1);
    end
    % OCV 계산
    OCV_k = interp1(soc_values, ocv_values, SOC_true(k), 'linear', 'extrap');
    % 단자 전압 계산 (R0은 0.01 옴으로 설정)
    Vt_true(k) = OCV_k + sum(V_RC_true(:, k)) + 0.01 * current_profile(k);
end

% 측정 전압에 노이즈 추가
voltage_noise_level = 0.005;  % 전압 노이즈 레벨 (V)
Vt_measured = Vt_true + voltage_noise_level * randn(num_samples, 1);

%% DRT 기반 모델로 SOC 추정 (칼만 필터 적용)

% 선택할 RC 요소의 개수 설정 (모델 복잡도 조절)
num_RC_model = 5;  % 예시로 5개 요소 선택

% DRT 결과에서 상위 num_RC_model 개의 RC 요소 선택
[~, sorted_indices] = sort(R_array, 'descend');  % R 값이 큰 순서로 정렬
selected_indices = sorted_indices(1:num_RC_model);
tau_model = tau_array(selected_indices);
R_model = R_array(selected_indices);
C_model = C_array(selected_indices);

% 상태 공간 모델 구성
% 상태 벡터: [V_C1; V_C2; ...; V_CN; SOC]
N = num_RC_model;  % 선택한 RC 요소의 개수

% 상태 공간 행렬 초기화
A_c = zeros(N+1, N+1);
B_c = zeros(N+1, 1);
C_c = zeros(1, N+1);
D_c = 0.01;  % R0 값을 0.01 옴으로 설정

% A_c와 B_c 구성
for i = 1:N
    A_c(i, i) = -1 / (R_model(i) * C_model(i));
    B_c(i) = 1 / C_model(i);
    C_c(i) = 1;  % 출력 방정식에서 V_Ci의 계수
end
% SOC 방정식
A_c(N+1, N+1) = 0;
B_c(N+1) = -1 / Q_nom;

% 이산화
[A_d, B_d] = c2d(A_c, B_c, dt);
C_d = C_c;
D_d = D_c;

% 칼만 필터 구현
% 초기 상태 및 공분산
x_hat = zeros(N+1, 1);  % 초기 상태 추정값
x_hat(N+1) = initial_soc;  % SOC의 초기값 설정
P = eye(N+1) * 1e-3;  % 초기 오차 공분산

% 노이즈 공분산 행렬
Q_kf = eye(N+1) * 1e-5;  % 프로세스 노이즈 공분산
R_kf = 1e-4;  % 측정 노이즈 공분산

% 저장 변수
x_hat_history = zeros(N+1, num_samples);
Vt_estimated = zeros(num_samples, 1);

for k = 1:num_samples
    % 입력 및 출력 데이터
    I_k = current_profile(k);  % 입력 전류
    V_k = Vt_measured(k);      % 측정 전압

    % 예측 단계
    x_hat = A_d * x_hat + B_d * I_k;
    P = A_d * P * A_d' + Q_kf;

    % OCV 계산
    SOC_k = x_hat(N+1);
    % SOC의 범위를 0~1로 제한
    SOC_k = max(0, min(1, SOC_k));
    x_hat(N+1) = SOC_k;  % 제한된 SOC를 상태에 업데이트
    OCV_k = interp1(soc_values, ocv_values, SOC_k, 'linear', 'extrap');

    % 출력 예측
    y_hat = C_d * x_hat + D_d * I_k + OCV_k;

    % 칼만 이득 계산
    K = P * C_d' / (C_d * P * C_d' + R_kf);

    % 상태 업데이트
    x_hat = x_hat + K * (V_k - y_hat);
    P = (eye(N+1) - K * C_d) * P;

    % 상태 저장
    x_hat_history(:, k) = x_hat;
    Vt_estimated(k) = y_hat;
end

% SOC 추정 결과
SOC_estimated = x_hat_history(N+1, :);

%% 결과 시각화

% SOC 추정 결과 플롯
figure;
plot(time_vector / 3600, SOC_true, 'k-', 'LineWidth', 1.5); hold on;
plot(time_vector / 3600, SOC_estimated, 'b--', 'LineWidth', 1.5);
xlabel('Time (hours)');
ylabel('SOC');
title('True SOC vs Estimated SOC');
legend('True SOC', 'Estimated SOC');
grid on;

% 전압 비교 플롯
figure;
plot(time_vector / 3600, Vt_measured, 'k-', 'LineWidth', 1.5); hold on;
plot(time_vector / 3600, Vt_estimated, 'r--', 'LineWidth', 1.5);
xlabel('Time (hours)');
ylabel('Voltage (V)');
title('Measured Voltage vs Estimated Voltage');
legend('Measured Voltage', 'Estimated Voltage');
grid on;

% SOC 추정 오차 플롯
figure;
plot(time_vector / 3600, SOC_true - SOC_estimated, 'm-', 'LineWidth', 1.5);
xlabel('Time (hours)');
ylabel('SOC Estimation Error');
title('SOC Estimation Error over Time');
grid on;

