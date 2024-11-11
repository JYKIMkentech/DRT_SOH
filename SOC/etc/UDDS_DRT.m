clear; clc; close all;

%% 1. UDDS 주행 데이터 로드
% UDDS 주행 데이터를 로드합니다.
load('udds_data.mat');  % 'udds_data' 구조체를 로드합니다.

%% 2. SOC-OCV 데이터 로드
% SOC-OCV 데이터를 로드합니다.
load('soc_ocv.mat', 'soc_ocv');
soc_values = soc_ocv(:, 1);  % SOC 값
ocv_values = soc_ocv(:, 2);  % OCV 값

%% 3. DRT 추정에 필요한 파라미터 설정
n = 40;  % 이산 요소의 개수
tau_min = 0.1;     % 최소 시간 상수 (초)
tau_max = 1100;    % 최대 시간 상수 (초)

% Theta 및 tau 값 계산
theta_min = log(tau_min);
theta_max = log(tau_max);
theta_discrete = linspace(theta_min, theta_max, n);
tau_discrete = exp(theta_discrete);

% Delta theta 계산
delta_theta = theta_discrete(2) - theta_discrete(1);

% 정규화 파라미터
lambda = 0.001;  % 필요에 따라 조정 가능

% Gamma에 대한 1차 차분 행렬 L_gamma 생성
L_gamma = zeros(n-1, n);
for i = 1:n-1
    L_gamma(i, i) = -1;
    L_gamma(i, i+1) = 1;
end

% R0에 대한 정규화를 피하기 위해 L_aug 생성
L_aug = [L_gamma, zeros(n-1, 1)];

%% 4. 각 사이클에 대한 DRT 추정
num_cycles = length(udds_data);

% 결과 저장을 위한 배열 사전 할당
gamma_est_all = zeros(num_cycles-1, n);  % 마지막 사이클 제외
R0_est_all = zeros(num_cycles-1, 1);

for s = 1:num_cycles-1  % 마지막 사이클은 데이터가 짧으므로 제외
    fprintf('Processing Cycle %d/%d...\n', s, num_cycles-1);
    
    % 현재 사이클의 데이터 추출
    t = udds_data(s).t;
    ik = udds_data(s).I;
    V_sd = udds_data(s).V;
    SOC = udds_data(s).SOC;
    
    % 시간 간격 dt 계산
    delta_t = [0; diff(t)];
    dt = delta_t;
    dt(1) = dt(2);  % 첫 번째 dt 보정
    
    % OCV 계산 (SOC-OCV 테이블 사용)
    ocv_over_time = interp1(soc_values, ocv_values, SOC, 'linear', 'extrap');
    
    %% 4.1 W 행렬 생성
    % W 행렬 초기화
    W = zeros(length(t), n);
    for k_idx = 1:length(t)
        for i = 1:n
            if k_idx == 1
                W(k_idx, i) = ik(k_idx) * (1 - exp(-dt(k_idx) / tau_discrete(i))) * delta_theta;
            else
                W(k_idx, i) = W(k_idx-1, i) * exp(-dt(k_idx) / tau_discrete(i)) + ...
                              ik(k_idx) * (1 - exp(-dt(k_idx) / tau_discrete(i))) * delta_theta;
            end
        end
    end
    
    % R0 추정을 위한 W 행렬 확장
    W_aug = [W, ik(:)];  % ik(:)는 ik를 열 벡터로 변환
    
    %% 4.2 y 벡터 생성
    % y = V_sd - OCV
    y = V_sd - ocv_over_time;
    y = y(:);  % y를 열 벡터로 변환
    
    %% 4.3 정규화된 최소자승법을 통한 추정
    % Theta = [gamma; R0]
    % (W_aug' * W_aug + lambda * L_aug' * L_aug) * Theta = W_aug' * y
    Theta_est = (W_aug' * W_aug + lambda * (L_aug' * L_aug)) \ (W_aug' * y);
    
    % gamma와 R0 추정값 추출
    gamma_est = Theta_est(1:n);
    R0_est = Theta_est(n+1);
    
    % 추정값 저장
    gamma_est_all(s, :) = gamma_est';
    R0_est_all(s) = R0_est;
    
    %% 4.4 V_est 계산 (검증용)
    % V_RC 및 V_est 초기화
    V_RC = zeros(n, length(t));  % 각 요소의 전압
    V_est = zeros(length(t), 1);
    for k_idx = 1:length(t)
        if k_idx == 1
            for i = 1:n
                V_RC(i, k_idx) = gamma_est(i) * delta_theta * ik(k_idx) * (1 - exp(-dt(k_idx) / tau_discrete(i)));
            end
        else
            for i = 1:n
                V_RC(i, k_idx) = V_RC(i, k_idx-1) * exp(-dt(k_idx) / tau_discrete(i)) + ...
                                 gamma_est(i) * delta_theta * ik(k_idx) * (1 - exp(-dt(k_idx) / tau_discrete(i)));
            end
        end
        % 시간 k_idx에서의 V_est 계산
        V_est(k_idx) = ocv_over_time(k_idx) + R0_est * ik(k_idx) + sum(V_RC(:, k_idx));
    end
    
    %% 4.5 DRT Gamma 그래프 출력
    figure(1);
    subplot(4, 4, s);
    plot(theta_discrete, gamma_est, 'LineWidth', 1.5);
    xlabel('\theta = ln(\tau)');
    ylabel('\gamma');
    title(['DRT for Cycle ', num2str(s)]);
    grid on;
    
    %% 4.6 전압 비교 그래프 출력
    figure(2);
    subplot(4, 4, s);
    plot(t, V_sd, 'b', 'LineWidth', 1);
    hold on;
    plot(t, V_est, 'r--', 'LineWidth', 1);
    xlabel('Time (s)');
    ylabel('Voltage (V)');
    title(['Voltage Comparison for Cycle ', num2str(s)]);
    legend('Measured V_{sd}', 'Estimated V_{est}');
    grid on;
    hold off;
end

%% Plot

% plot(udds_time,udds_current);
% hold on
% plot(udds_time,udds_SOC);
% xlabel('time')
% ylabel('current')
% yyaxis right
% ylabel('soc')


