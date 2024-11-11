clc; clear; close all;

%% 1. 데이터 로드

% 데이터 파일 경로를 사용자 환경에 맞게 수정하세요.
load('G:\공유 드라이브\BSL_Data3\Driving cycles\03-21-17_00.29 25degC_UDDS_Pan18650PF.mat');
load('udds_data.mat');

% 개별 트립 데이터
trip1_I = udds_data(1).I;
trip1_V = udds_data(1).V;
trip1_t = udds_data(1).t;

% 전체 트립 데이터
All_trips_I = meas.Current;
All_trips_V = meas.Voltage; 
All_trips_t = meas.Time; 

SOC_initial = 0.9901;

%% 2. Markov chain noise를 전류에 추가 및 SOC 계산

n = 21;
noise_percent = 0.05;

mean_I = mean(trip1_I); % 평균 전류
mean_noise = mean_I * noise_percent; % 노이즈 평균 전류 = 평균 전류 * 노이즈 퍼센트

min_noise = min(trip1_I) * noise_percent;  % 하한 노이즈 
max_noise = max(trip1_I) * noise_percent;  % 상한 노이즈 
span = max_noise - min_noise; % trip1에 표시되는 전류 범위 설정
noise_vector = linspace(mean_noise - span/2, mean_noise + span/2, n); % 노이즈 벡터 중간값 = mean_noise로 설정

P = zeros(n);

% 표준 편차 설정 
sigma = span / 50;

% 각 행에 대해 전이 확률 계산
for i = 1:n
    % 현재 상태에서 가능한 모든 다음 상태에 대한 확률 계산
    probabilities = normpdf(noise_vector, noise_vector(i), sigma);
    
    % 정규화하여 확률의 합이 1이 되도록 함
    P(i, :) = probabilities / sum(probabilities);
end

%% 3. Markov 체인을 사용하여 노이즈 추가 및 상태 기록

% 3.1. 개별 트립 데이터에 노이즈 추가
% 초기 상태를 무작위로 설정 (1부터 n까지의 정수 중 하나)
initial_state = randsample(1:n, 1);

noisy_I_trip1 = zeros(size(trip1_I));
states_trip1 = zeros(size(trip1_I));
current_state = initial_state;

for t = 1:length(trip1_I)
    % 현재 상태에 해당하는 노이즈 값을 추가
    noisy_I_trip1(t) = trip1_I(t) + noise_vector(current_state);
    
    % 상태 기록
    states_trip1(t) = current_state;
    
    % 다음 상태로 전이 using randsample
    % randsample(population, k, replace, weights)
    current_state = randsample(1:n, 1, true, P(current_state, :));
end

% SOC 계산 (개별 트립)
SOC_noisy_trip1 = SOC_initial + cumtrapz(trip1_t / 3600, noisy_I_trip1) / 2.9; % 배터리 용량 2.9 Ah 가정
SOC_true_trip1  = SOC_initial + cumtrapz(trip1_t / 3600, trip1_I) / 2.9 ;

% 3.2. 전체 트립 데이터에 노이즈 추가
% 초기 상태를 무작위로 설정 (1부터 n까지의 정수 중 하나)
initial_state = 9 ; %randsample(1:n, 1);


noisy_I_All_trips = zeros(size(All_trips_I));
states_All_trips = zeros(size(All_trips_I));
current_state = initial_state;

for t = 1:length(All_trips_I)
    % 현재 상태에 해당하는 노이즈 값을 추가
    noisy_I_All_trips(t) = All_trips_I(t) + noise_vector(current_state);
    
    % 상태 기록
    states_All_trips(t) = current_state;
    
    % 다음 상태로 전이
    current_state = randsample(1:n, 1, true, P(current_state, :));
end

% SOC 계산 (전체 트립)
SOC_noisy_All_trips = SOC_initial + cumtrapz(All_trips_t / 3600, noisy_I_All_trips) / 2.9; % 배터리 용량 2.9 Ah 가정
SOC_true_All_trips  = SOC_initial + cumtrapz(All_trips_t / 3600, All_trips_I) / 2.9 ;

%% 4. 결과 시각화 (각 플롯을 별도의 figure로 분리)

% 4.1. 개별 트립: 원본 전류 vs 노이즈 전류
figure('Name', '개별 트립: 원본 전류 vs 노이즈 전류', 'NumberTitle', 'off');
plot(trip1_t, trip1_I, 'b', 'LineWidth', 1.5);
hold on;
plot(trip1_t, noisy_I_trip1, 'r', 'LineWidth', 1.5);
title('개별 트립: 원본 전류 vs 노이즈 전류');
xlabel('Time (s)');
ylabel('Current (A)');
legend('Original Current', 'Noisy Current');
grid on;

% 4.2. 개별 트립: SOC 비교
figure('Name', '개별 트립: SOC 비교', 'NumberTitle', 'off');
plot(trip1_t, SOC_noisy_trip1 * 100, 'r', 'LineWidth', 1.5);
hold on;
plot(trip1_t, SOC_true_trip1 * 100, 'b', 'LineWidth', 1.5);
title('개별 트립: SOC 비교');
xlabel('Time (s)');
ylabel('SOC (%)');
legend('Noisy SOC', 'True SOC');
grid on;

% 4.3. 개별 트립: 시간에 따른 노이즈 상태 변화 및 전류 함께 시각화
figure('Name', '개별 트립: 노이즈 상태 변화 및 전류', 'NumberTitle', 'off');
yyaxis left
plot(trip1_t, trip1_I, 'b', 'LineWidth', 1.5);
hold on;
plot(trip1_t, noisy_I_trip1, 'r', 'LineWidth', 1.5);
ylabel('Current (A)');
yyaxis right
plot(trip1_t, noise_vector(states_trip1), 'g', 'LineWidth', 1.5);
ylabel('Noise Value (A)');
title('개별 트립: 노이즈 상태 변화 및 전류');
xlabel('Time (s)');
legend('Original Current', 'Noisy Current', 'Noise Value');
grid on;
xlim([22 32]); % 필요에 따라 조정

% 4.4. 전체 트립: 원본 전류 vs 노이즈 전류
figure('Name', '전체 트립: 원본 전류 vs 노이즈 전류', 'NumberTitle', 'off');
plot(All_trips_t, All_trips_I, 'b', 'LineWidth', 1.5);
hold on;
plot(All_trips_t, noisy_I_All_trips, 'r', 'LineWidth', 1.5);
title('전체 트립: 원본 전류 vs 노이즈 전류');
xlabel('Time (s)');
ylabel('Current (A)');
legend('Original Current', 'Noisy Current');
grid on;

% 4.5. 전체 트립: SOC 비교
figure('Name', '전체 트립: SOC 비교', 'NumberTitle', 'off');
plot(All_trips_t, SOC_noisy_All_trips * 100, 'r', 'LineWidth', 1.5);
hold on;
plot(All_trips_t, SOC_true_All_trips * 100, 'b', 'LineWidth', 1.5);
title('전체 트립: SOC 비교');
xlabel('Time (s)');
ylabel('SOC (%)');
legend('Noisy SOC', 'True SOC');
grid on;

% 4.6. 전체 트립: 시간에 따른 노이즈 상태 변화
figure('Name', '전체 트립: 시간에 따른 노이즈 상태 변화', 'NumberTitle', 'off');
plot(All_trips_t, noise_vector(states_All_trips), 'g', 'LineWidth', 1.5);
title('전체 트립: 시간에 따른 노이즈 상태 변화');
xlabel('Time (s)');
ylabel('Noise Value (A)');
grid on;

%% 5. SOC 차이 변화도 시각화

% 5.1. 개별 트립: SOC 차이
figure('Name', '개별 트립: SOC 차이', 'NumberTitle', 'off');
SOC_diff_trip1 = abs(SOC_noisy_trip1 - SOC_true_trip1) * 100; % SOC 차이를 퍼센트로 계산
plot(trip1_t, SOC_diff_trip1, 'm', 'LineWidth', 1.5);
title('개별 트립: SOC 차이 (Noisy - True)');
xlabel('Time (s)');
ylabel('SOC Difference (%)');
legend('SOC Difference');
grid on;

% 5.2. 전체 트립: SOC 차이
figure('Name', '전체 트립: SOC 차이', 'NumberTitle', 'off');
SOC_diff_All_trips = abs(SOC_noisy_All_trips - SOC_true_All_trips) * 100; % SOC 차이를 퍼센트로 계산
plot(All_trips_t, SOC_diff_All_trips, 'm', 'LineWidth', 1.5);
title('전체 트립: SOC 차이 (Noisy - True)');
xlabel('Time (s)');
ylabel('SOC Difference (%)');
legend('SOC Difference');
grid on;

%% 6. 전이 확률 분포 시각화

figure('Name', 'Transition Probability Distributions', 'NumberTitle', 'off');
num_rows = 7;
num_cols = 3;

for i = 1:n
    subplot(num_rows, num_cols, i);
    plot(noise_vector, P(i,:), 'b-', 'LineWidth', 1.5);
    title(['State ' num2str(i)]);
    xlabel('Noise Value (A)');
    ylabel('Probability');
    xlim([min(noise_vector), max(noise_vector)]);
    ylim([0, 1]);
    grid on;
end

% 빈 subplot 채우기 (n=21, 7x3=21)
% 만약 n가 21보다 작다면, 추가 subplot을 빈 상태로 유지
for i = n+1:num_rows*num_cols
    subplot(num_rows, num_cols, i);
    axis off;
end
