clc; clear; close all;

% Set the random seed to ensure reproducibility
rng(0);  % 재현성을 위해 시드 설정

% Parameters
num_scenarios = 10;  % 시나리오 수
num_waves = 3;       % 각 시나리오당 사인파 수
t = linspace(0, 1000, 10000);  % 시간 벡터 (0~1000초, 샘플링 포인트 10000개)

% Constraints
T_min = 15;           % 최소 주기 (초) - τ_min = 2.5초 대응
T_max = 250;          % 최대 주기 (초) - τ_max = 40초 대응

% Initialize matrices for amplitudes, periods, and current scenarios
A = zeros(num_scenarios, num_waves);   % 진폭 행렬
T = zeros(num_scenarios, num_waves);   % 주기 행렬
ik_scenarios = zeros(num_scenarios, length(t)); % 전류 시나리오 행렬

% Generate random amplitudes, periods, and current scenarios
for s = 1:num_scenarios
    % Random amplitudes that sum to 3
    temp_A = rand(1, num_waves);       % 3개의 랜덤 진폭 생성
    A(s, :) = 3 * temp_A / sum(temp_A);  % 진폭 합이 3이 되도록 정규화
    
    % Random periods between T_min and T_max on a logarithmic scale
    log_T_min = log10(T_min);
    log_T_max = log10(T_max);
    T_log = log_T_min + (log_T_max - log_T_min) * rand(1, num_waves);
    T(s, :) = 10.^T_log;  % 로그 스케일에서 선형 스케일로 변환
    
    % Generate the current scenario as the sum of three sine waves
    ik_scenarios(s, :) = A(s,1)*sin(2*pi*t / T(s,1)) + ...
                         A(s,2)*sin(2*pi*t / T(s,2)) + ...
                         A(s,3)*sin(2*pi*t / T(s,3));
end

% Save the generated amplitudes, periods, and current scenarios to AS1.mat
save('AS1.mat', 'A', 'T', 'ik_scenarios', 't');

% Display the generated values for verification
disp('Amplitudes (A):');
disp(A);
disp('Periods (T):');
disp(T);

% Plot the 10 current scenarios in a 5x2 subplot grid
figure;
for s = 1:num_scenarios
    subplot(5, 2, s);  % 5x2 그리드의 서브플롯 생성
    plot(t, ik_scenarios(s, :), 'LineWidth', 1.5);
    title(['Scenario ', num2str(s)]);
    xlabel('Time (s)');
    ylabel('Current (A)');
    grid on;
end

% Adjust the layout for better spacing between subplots
sgtitle('Current Scenarios for 10 Randomized Cases');  % 전체 그림 제목 추가


% clc; clear; close all;
% 
% % Set the random seed to ensure reproducibility
% rng(0);  % You can change this seed to any number to generate a different fixed random sequence
% 
% % Parameters
% num_scenarios = 10;  % Number of current scenarios
% num_waves = 3;       % Number of sine waves per scenario
% t = 0:0.01:100;      % Time vector
% 
% % Constraints
% T_min = 1;           % Minimum period value
% T_max = 20;          % Maximum period value
% 
% % Initialize matrices for amplitudes, periods, and current scenarios
% A = zeros(num_scenarios, num_waves);   % Amplitudes matrix
% T = zeros(num_scenarios, num_waves);   % Periods matrix
% ik_scenarios = zeros(num_scenarios, length(t)); % Current scenarios matrix
% 
% % Generate random amplitudes, periods, and current scenarios
% for s = 1:num_scenarios
%     % Random amplitudes that sum to 3
%     temp_A = rand(1, num_waves);       % Generate 3 random amplitudes
%     A(s, :) = 3 * temp_A / sum(temp_A);  % Normalize to make sum equal to 3
%     
%     % Random periods between T_min and T_max
%     T(s, :) = T_min + (T_max - T_min) * rand(1, num_waves);
%     
%     % Generate the current scenario as the sum of three sine waves
%     ik_scenarios(s, :) = A(s,1)*sin(2*pi*t / T(s,1)) + ...
%                          A(s,2)*sin(2*pi*t / T(s,2)) + ...
%                          A(s,3)*sin(2*pi*t / T(s,3));
% end
% 
% % Save the generated amplitudes, periods, and current scenarios to AS1.mat
% save('AS1.mat', 'A', 'T', 'ik_scenarios', 't');
% 
% % Display the generated values for verification
% disp('Amplitudes (A):');
% disp(A);
% disp('Periods (T):');
% disp(T);
% 
% % Plot the 10 current scenarios in a 5x2 subplot grid
% figure;
% for s = 1:num_scenarios
%     subplot(5, 2, s);  % Create a 5x2 grid of subplots
%     plot(t, ik_scenarios(s, :), 'LineWidth', 1.5);
%     title(['Scenario ', num2str(s)]);
%     xlabel('Time (s)');
%     ylabel('Current (A)');
%     grid on;
% end
% 
% % Adjust the layout for better spacing between subplots
% sgtitle('Current Scenarios for 10 Randomized Cases');  % Add a title for the entire figure
% 
% 
