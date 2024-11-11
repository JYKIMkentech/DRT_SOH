%% 2. AS1.mat 파일 로드 및 DRT 분석 수행

clc; clear; close all;

% AS1.mat 파일 로드
load('AS1.mat');  % 첫 번째 코드에서 저장한 A, T, ik_scenarios, t 변수를 불러옵니다.

%% Parameters 
n = 201;  % Number of discrete elements
dt = t(2) - t(1);  % Time step based on loaded time vector
num_scenarios = 10;  % Number of current scenarios
lambda = 0.153;  % Regularization parameter (주어진 람다 값)

%% DRT 

% 이봉형(bimodal) 감마 분포 생성
% 첫 번째 피크의 평균과 표준편차
mu_theta1 = log(10);     % 첫 번째 피크의 중심 위치
sigma_theta1 = 1;        % 첫 번째 피크의 폭

% 두 번째 피크의 평균과 표준편차
mu_theta2 = log(120);    % 두 번째 피크의 중심 위치
sigma_theta2 = 0.7;      % 두 번째 피크의 폭

% Theta 범위 설정 (두 피크를 모두 포함하도록)
theta_min = min([mu_theta1, mu_theta2]) - 3 * max([sigma_theta1, sigma_theta2]);
theta_max = max([mu_theta1, mu_theta2]) + 3 * max([sigma_theta1, sigma_theta2]);
theta_discrete = linspace(theta_min, theta_max, n);

% Corresponding tau values
tau_discrete = exp(theta_discrete);

% Delta theta
delta_theta = theta_discrete(2) - theta_discrete(1);

% 각 가우시안 분포 계산
gamma1 = (1 / (sigma_theta1 * sqrt(2 * pi))) * exp(- (theta_discrete - mu_theta1).^2 / (2 * sigma_theta1^2));
gamma2 = (1 / (sigma_theta2 * sqrt(2 * pi))) * exp(- (theta_discrete - mu_theta2).^2 / (2 * sigma_theta2^2));

% 두 가우시안 분포를 합산하여 이봉형 감마 분포 생성
gamma_discrete_true = gamma1 + gamma2;

% 감마 분포를 최대값이 1이 되도록 정규화
gamma_discrete_true = gamma_discrete_true / max(gamma_discrete_true);

% Analytical gamma estimates
gamma_analytical_all = zeros(num_scenarios, n);  % Analytical gamma estimates

% Voltage storage variables (V_est and V_sd for each scenario)
V_est_all = zeros(num_scenarios, length(t));  % For storing V_est for all scenarios
V_sd_all = zeros(num_scenarios, length(t));   % For storing V_sd for all scenarios

%% First-order difference matrix L
% 여기서는 1차 차분 행렬을 사용합니다.
L = zeros(n-1, n);
for i = 1:n-1
    L(i, i) = -1;
    L(i, i+1) = 1;
end

%% Voltage Synthesis and DRT Estimation
R0 = 0.1;  % Ohmic resistance
OCV = 0;   % Open Circuit Voltage

rng(0);  % Ensure reproducibility of noise
noise_level = 0.01;  % 노이즈 수준 설정

for s = 1:num_scenarios
    fprintf('Processing Scenario %d/%d...\n', s, num_scenarios);
    
    % Current for the scenario
    ik = ik_scenarios(s, :);  % 로드된 전류 시나리오 사용
    
    %% Initialize Voltage
    V_est = zeros(1, length(t));  % Model voltage calculated via n-element model
    V_RC = zeros(n, length(t));  % Voltages for each element
    
    %% Voltage Calculation
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
    
    % Store V_est for the current scenario
    V_est_all(s, :) = V_est;  % Save the calculated V_est for this scenario
    
    %% Add Noise to the Voltage
    V_sd = V_est + noise_level * randn(size(V_est));  % V_sd = synthetic measured voltage
    
    % Store V_sd for the current scenario
    V_sd_all(s, :) = V_sd;  % Save the noisy V_sd for this scenario
    
    %% Construct W Matrix
    W = zeros(length(t), n);  % Initialize W matrix
    for k_idx = 1:length(t)
        if k_idx == 1
            for i = 1:n
                W(k_idx, i) = ik(k_idx) * (1 - exp(-dt / tau_discrete(i))) * delta_theta;
            end
        else
            for i = 1:n
                W(k_idx, i) = W(k_idx-1, i) * exp(-dt / tau_discrete(i)) + ...
                              ik(k_idx) * (1 - exp(-dt / tau_discrete(i))) * delta_theta;
            end
        end
    end
    
    %% 감마 추정: quadprog를 이용하여 제약 조건 \(\gamma \geq 0\) 적용
    % Remove constants: Subtract OCV and R0*ik
    y_adjusted = V_sd' - OCV - R0 * ik';
    
    % Define H and f for quadprog
    H = W' * W + lambda * (L' * L);
    f = -W' * y_adjusted;
    
    % H를 대칭 행렬로 보정
    H = (H + H') / 2;
    
    % 제약 조건 설정: gamma >= 0
    lb = zeros(n, 1);  % 하한: 0
    ub = [];           % 상한: 없음
    
    % quadprog 옵션 설정
    options = optimoptions('quadprog', 'Display', 'off');
    
    % quadprog를 이용하여 최적화 문제 해결
    [gamma_quadprog, ~, exitflag] = quadprog(H, f, [], [], [], [], lb, ub, [], options);
    
    % 최적화 실패 시 경고 메시지 출력
    if exitflag ~= 1
        warning('quadprog did not converge for scenario %d', s);
    end
    
    %% 추정된 감마 저장
    gamma_analytical_all(s, :) = gamma_quadprog';
    
    %% Plot Voltage and DRT Comparison
    % Voltage Plot
    figure(1);  
    subplot(5, 2, s);
    yyaxis left
    plot(t, ik, 'b-', 'LineWidth', 1.5);
    ylabel('Current (A)');
    xlabel('Time (s)');
    grid on;
    
    yyaxis right
    plot(t, V_sd, 'r-', 'LineWidth', 1.5);
    ylabel('Voltage (V)');
    ylim([min(V_sd)-0.1, max(V_sd)+0.1]);
    
    % Update title with correct amplitudes and periods
    title(['Scenario ', num2str(s), ...
           ': A1=', num2str(A(s,1)), ', A2=', num2str(A(s,2)), ', A3=', num2str(A(s,3)), ...
           ', T1=', num2str(T(s,1)), ', T2=', num2str(T(s,2)), ', T3=', num2str(T(s,3))]);
    
    % Add legend
    legend({'Current (A)', 'Voltage (V)'}, 'Location', 'best');
    
    % DRT Comparison Plot
    figure(1 + s);  % DRT Comparison Figure for each scenario
    hold on;
    
    % Plot True gamma
    plot(theta_discrete, gamma_discrete_true, 'k-', 'LineWidth', 1.5, 'DisplayName', 'True \gamma');
    
    % Plot Estimated gamma from quadprog
    plot(theta_discrete, gamma_analytical_all(s, :), ':', 'Color', 'g', 'LineWidth', 1.5, 'DisplayName', 'Estimated \gamma (quadprog)');
    
    hold off;
    xlabel('\theta = ln(\tau)');
    ylabel('\gamma');
    title(['DRT Comparison for Scenario ', num2str(s), ' (\lambda = ', num2str(lambda), ')']);
    legend('Location', 'Best');
    grid on;
end


% 현재 시나리오의 A와 T 값 출력
fprintf('Scenario %d Parameters:\n', s);
fprintf('A1=%.3f, A2=%.3f, A3=%.3f\n', A(s,1), A(s,2), A(s,3));
fprintf('T1=%.3f, T2=%.3f, T3=%.3f\n', T(s,1), T(s,2), T(s,3));

