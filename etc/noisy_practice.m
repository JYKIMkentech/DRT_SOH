clc;
clear;
close all;

% Load UDDS data
load('G:\공유 드라이브\BSL_Data3\Driving cycles\03-21-17_00.29 25degC_UDDS_Pan18650PF.mat');

% Load UDDS data
udds_current = meas.Current; % UDDS 전류 데이터
udds_voltage = meas.Voltage; % UDDS 전압 데이터
udds_time = meas.Time; % UDDS 시간 데이터

% 파라미터 설정
n = 21;                    % 상태의 개수
sigma = 1e-1;              % 노이즈 분포의 표준편차 (1 mA)
noise_levels = linspace(-2e-1, 2e-1, n); % -2mA에서 +2mA로 21개 구간

% 전이 상태 행렬 P 초기화
P = zeros(n, n);

% 전이 확률 계산 (비대칭 설정)
for i = 1:n
    for j = 1:n
        % 비대칭 노이즈를 위한 전이 확률 계산
        if noise_levels(j) > noise_levels(i)
            P(i, j) = normpdf(noise_levels(j), noise_levels(i), sigma) * 1.5; % 양수 방향으로 편향
        else
            P(i, j) = normpdf(noise_levels(j), noise_levels(i), sigma);
        end
    end
    P(i, :) = P(i, :) / sum(P(i, :)); % 각 행의 합이 1이 되도록 정규화
end

% 노이즈 벡터 Q 초기화
Q = noise_levels';

% UDDS 전류 데이터를 true_current로 설정
true_current = udds_current; % UDDS 전류 데이터를 사용하여 true_current 설정

% 배터리 용량 및 초기 SOC 설정
battery_capacity = 2.9; % 배터리 용량 2.9 Ah
initial_soc = 0.9901;   % 초기 SOC
num_steps = length(true_current);
dt = mean(diff(udds_time)); % UDDS 데이터의 시간 간격

% SOC 계산을 위한 변수 초기화
true_soc = zeros(1, num_steps);
noisy_soc_markov = zeros(1, num_steps);
noisy_soc_rand = zeros(1, num_steps);

noisy_current_markov = zeros(1, num_steps);
noisy_current_rand = zeros(1, num_steps);

% 초기 SOC 설정
true_soc(1) = initial_soc;
noisy_soc_markov(1) = initial_soc;
noisy_soc_rand(1) = initial_soc;

% 초기 상태 설정 (중앙 상태에서 시작)
current_state = ceil(n/2);

% Markov chain 및 랜덤 노이즈 추가
for t = 2:num_steps
    % 마르코프 체인 기반 노이즈
    current_state = randsample(1:n, 1, true, P(current_state, :));
    noisy_current_markov(t) = true_current(t) + Q(current_state);
    
    % 랜덤 노이즈 추가 (-2mA ~ 2mA 범위)
    noisy_current_rand(t) = true_current(t) + (rand * 4e-3) - 2e-3;
    
    % True SOC 계산 
    true_soc(t) = true_soc(t-1) + (true_current(t) * dt) / (battery_capacity * 3600);
    
    % Noisy SOC 계산 (Markov noise)
    noisy_soc_markov(t) = noisy_soc_markov(t-1) + (noisy_current_markov(t) * dt) / (battery_capacity * 3600);
    
    % Noisy SOC 계산 (Random noise)
    noisy_soc_rand(t) = noisy_soc_rand(t-1) + (noisy_current_rand(t) * dt) / (battery_capacity * 3600);
end

% 전류 데이터 비교 plot
figure;
plot(1:num_steps, true_current * 1e3, 'b-', 'LineWidth', 2); % True current (mA)
hold on;
plot(1:num_steps, noisy_current_markov * 1e3, 'r--', 'LineWidth', 2); % Noisy current (Markov)
plot(1:num_steps, noisy_current_rand * 1e3, 'g-.', 'LineWidth', 2); % Noisy current (Random)
xlabel('Time Step');
ylabel('Current (mA)');
title('True Current vs Noisy Current (Markov vs Random Noise)');
legend('True Current', 'Noisy Current (Markov)', 'Noisy Current (Random)');
%xlim([0 1000])
grid on;
hold off;

% SOC 비교 plot
figure;
plot(1:num_steps, true_soc * 100, 'b-', 'LineWidth', 2); % True SOC (%)
hold on;
plot(1:num_steps, noisy_soc_markov * 100, 'r--', 'LineWidth', 2); % Noisy SOC (Markov)
plot(1:num_steps, noisy_soc_rand * 100, 'g-.', 'LineWidth', 2); % Noisy SOC (Random)
xlabel('Time Step');
ylabel('SOC (%)');
title('True SOC vs Noisy SOC (Markov vs Random Noise)');
legend('True SOC', 'Noisy SOC (Markov)', 'Noisy SOC (Random)');
%xlim([0 1000])
grid on;
hold off;


