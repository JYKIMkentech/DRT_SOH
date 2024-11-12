clc; clear; close all;

% Set the random seed to ensure reproducibility
rng(0);  % 재현성을 위해 시드 설정

%% Parameters
num_scenarios = 10;  % 시나리오 수
num_waves = 3;       % 각 시나리오당 사인파 수
t = linspace(0, 1000, 10000);  % 시간 벡터 (0~1000초, 샘플링 포인트 10000개)
dt = t(2) - t(1);
n = 201; % true n

%% DRT parameters

mu_theta = log(10);       % 평균 값
sigma_theta = 1;          % 표준편차 값

% 이산화된 theta 값들 (-3sigma부터 +3sigma까지)
theta_min = mu_theta - 3*sigma_theta;
theta_max = mu_theta + 3*sigma_theta;
theta_discrete = linspace(theta_min, theta_max, n);

% 해당하는 tau 값들
tau_discrete = exp(theta_discrete);

% Delta theta
delta_theta = theta_discrete(2) - theta_discrete(1);

% 실제 gamma 분포
gamma_discrete_true = (1/(sigma_theta * sqrt(2*pi))) * exp(- (theta_discrete - mu_theta).^2 / (2 * sigma_theta^2));

% gamma를 최대값이 1이 되도록 정규화
gamma_discrete_true = gamma_discrete_true / max(gamma_discrete_true);

% Constraints
T_min = 15;           % 최소 주기 (초)
T_max = 250;          % 최대 주기 (초)

%% current calculation

A = zeros(num_scenarios, num_waves);   % 진폭 행렬
T = zeros(num_scenarios, num_waves);   % 주기 행렬
ik_scenarios = zeros(num_scenarios, length(t)); % 전류 시나리오 행렬

% Generate random amplitudes, periods, and current scenarios
for s = 1:num_scenarios
    % Random amplitudes that sum to 3
    temp_A = rand(1, num_waves);       % 3개의 랜덤 진폭 생성
    A(s, :) = 3 * temp_A / sum(temp_A);  % 진폭 합이 3이 되도록 정규화
    
    % Random periods between T_min and T_max on a linear scale
    T(s, :) = T_min + (T_max - T_min) * rand(1, num_waves);  % 선형 스케일에서 주기 생성
    
    % Generate the current scenario as the sum of three sine waves
    ik_scenarios(s, :) = A(s,1)*sin(2*pi*t / T(s,1)) + ...
                         A(s,2)*sin(2*pi*t / T(s,2)) + ...
                         A(s,3)*sin(2*pi*t / T(s,3));
end

%% Voltage calculation

V_est_all = zeros(num_scenarios, length(t));  % 각 시나리오의 V_est 저장

for s = 1:num_scenarios
    fprintf('Processing Scenario %d/%d...\n', s, num_scenarios);
    
    % 현재 시나리오의 전류
    ik = ik_scenarios(s, :);  % 로드된 전류 시나리오 사용
    
    %% 전압 초기화
    V_est = zeros(1, length(t));  % n-요소 모델을 통한 모델 전압 계산
    R0 = 0.1;  % 저항 (오움)
    OCV = 0;   % 개방 회로 전압
    V_RC = zeros(n, length(t));  % 각 요소의 전압
    
    %% 전압 계산
    for k_idx = 1:length(t)
        if k_idx == 1
            for i = 1:n
                V_RC(i, k_idx) = gamma_discrete_true(i) * delta_theta * ik(k_idx) * (1 - exp(-dt / tau_discrete(i)));
            end
        else
            for i = 1:n
                V_RC(i, k_idx) = V_RC(i, k_idx-1) * exp(-dt / tau_discrete(i)) + ...
                                 gamma_discrete_true(i) * delta_theta * ik(k_idx) * (1 - exp(-dt / tau_discrete(i)));
            end
        end
        V_est(k_idx) = OCV + R0 * ik(k_idx) + sum(V_RC(:, k_idx));
    end
    
    % 현재 시나리오의 V_est 저장
    V_est_all(s, :) = V_est;  % 이 시나리오의 계산된 V_est 저장



end




%% Plot 

disp('Amplitudes (A):');
disp(A);
disp('Periods (T):');
disp(T);

% Plot the 10 current and voltage scenarios in a 5x2 subplot grid
figure;
for s = 1:num_scenarios
    subplot(5, 2, s);  % 5x2 그리드의 서브플롯 생성
    yyaxis left;
    plot(t, ik_scenarios(s, :), 'b', 'LineWidth', 1.5);
    ylabel('Current (A)');
    yyaxis right;
    plot(t, V_est_all(s, :), 'r', 'LineWidth', 1.5);
    ylabel('Voltage (V)');
    title(['Scenario ', num2str(s)]);
    xlabel('Time (s)');
    grid on;
    legend('Current', 'Voltage');
end

% Adjust the layout for better spacing between subplots
sgtitle('Current and Voltage Scenarios for 10 Randomized Cases');  % 전체 그림 제목 추가

%% save

save('AS1.mat', 'A', 'T', 'ik_scenarios', 't');

