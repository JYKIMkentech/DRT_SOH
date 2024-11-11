clc; clear; close all;

%% 1. 저장된 데이터 로드

% HPPC 기반 칼만 필터 결과 로드
load('HPPC_SOC_trip1.mat', 'HPPC_SOC_est', 'HPPC_Time');

% DRT 기반 칼만 필터 결과 로드
load('DRT_SOC_trip1.mat', 'DRT_SOC_est', 'DRT_Time');

% 실제 SOC 로드 (udds_data에서 첫 번째 트립의 SOC 가져오기)
load('udds_data.mat', 'udds_data');
SOC_true = udds_data(1).SOC;      % 실제 SOC
Time_true = udds_data(1).t;       % 시간 벡터

%% 2. 시간 벡터 통일 및 보간

% 시간 벡터가 다를 수 있으므로, 공통의 시간 벡터를 생성하여 SOC를 보간합니다.
% 여기서는 가장 작은 시간 간격을 가진 벡터를 기준으로 합니다.

% 세 시간 벡터의 최소 및 최대 시간 확인
t_min = max([HPPC_Time(1), DRT_Time(1), Time_true(1)]);
t_max = min([HPPC_Time(end), DRT_Time(end), Time_true(end)]);

% 공통 시간 벡터 생성 (최소 시간 간격으로)
dt = min([mean(diff(HPPC_Time)), mean(diff(DRT_Time)), mean(diff(Time_true))]);
Common_Time = (t_min:dt:t_max)';

% SOC 보간
HPPC_SOC_interp = interp1(HPPC_Time, HPPC_SOC_est, Common_Time, 'linear', 'extrap');
DRT_SOC_interp = interp1(DRT_Time, DRT_SOC_est, Common_Time, 'linear', 'extrap');
SOC_true_interp = interp1(Time_true, SOC_true, Common_Time, 'linear', 'extrap');

%% 3. SOC 비교 그래프 생성

figure;
plot(Common_Time / 3600, SOC_true_interp * 100, 'k', 'LineWidth', 1.5, 'DisplayName', 'True SOC');
hold on;
plot(Common_Time / 3600, HPPC_SOC_interp * 100, 'b--', 'LineWidth', 1.5, 'DisplayName', 'HPPC-based SOC Estimation');
plot(Common_Time / 3600, DRT_SOC_interp * 100, 'r--', 'LineWidth', 1.5, 'DisplayName', 'DRT-based SOC Estimation');
xlabel('Time [hours]');
ylabel('SOC [%]');
title('Comparison of SOC Estimation Methods for Trip 1');
legend('Location', 'best');
grid on;
hold off;

%% 4. SOC 추정 오류 계산 및 출력

% HPPC 기반 SOC 추정 오류
HPPC_SOC_error = SOC_true_interp - HPPC_SOC_interp;
HPPC_SOC_RMSE = sqrt(mean(HPPC_SOC_error.^2)) * 100;  % [%]

% DRT 기반 SOC 추정 오류
DRT_SOC_error = SOC_true_interp - DRT_SOC_interp;
DRT_SOC_RMSE = sqrt(mean(DRT_SOC_error.^2)) * 100;  % [%]

% RMSE 출력
fprintf('HPPC-based SOC Estimation RMSE: %.4f%%\n', HPPC_SOC_RMSE);
fprintf('DRT-based SOC Estimation RMSE: %.4f%%\n', DRT_SOC_RMSE);

%% 5. SOC 추정 오류 그래프 (선택 사항)

figure;
plot(Common_Time / 3600, HPPC_SOC_error * 100, 'b', 'LineWidth', 1.5, 'DisplayName', sprintf('HPPC SOC Error (RMSE: %.4f%%)', HPPC_SOC_RMSE));
hold on;
plot(Common_Time / 3600, DRT_SOC_error * 100, 'r', 'LineWidth', 1.5, 'DisplayName', sprintf('DRT SOC Error (RMSE: %.4f%%)', DRT_SOC_RMSE));
xlabel('Time [hours]');
ylabel('SOC Error [%]');
title('SOC Estimation Error for Trip 1');
legend('Location', 'best');
grid on;
hold off;
