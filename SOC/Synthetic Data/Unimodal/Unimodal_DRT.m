clc; clear; close all;
%%

% 세타 (ln (tau)) 가 정규분포를 따른다 ---> tau는 로그정규분포를 따른다. 

% Theta = ln(tau) (x축)
% gamma(theta) = [ R(exp(theta)) * exp(theta) ] = [ R(tau) * tau ] (y축)
% R_i = gamma_i * delta theta % 면적은 저항 = gamma (세로) * delta (가로, 일정하게)

%% AS1.mat 파일 로드
load('AS1.mat');  % 첫 번째 코드에서 저장한 A, T, ik_scenarios, t 변수를 불러옵니다.

%% Parameters 
n = 201;  % 이산화 요소의 개수
dt = t(2) - t(1);  % 시간 간격
num_scenarios = 10;  % 전류 시나리오의 수
lambda = 2.56;  % 정규화 파라미터

%% DRT 

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

% Quadprog를 통한 gamma 추정값 저장 변수
gamma_quadprog_all = zeros(num_scenarios, n);  % Quadprog로 구한 gamma 추정값

% 전압 저장 변수 (각 시나리오의 V_est와 V_sd)
V_est_all = zeros(num_scenarios, length(t));  % 각 시나리오의 V_est 저장
V_sd_all = zeros(num_scenarios, length(t));   % 각 시나리오의 V_sd 저장

%% 일차 차분 행렬 L
L = zeros(n-1, n);
for i = 1:n-1
    L(i, i) = -1;
    L(i, i+1) = 1;
end

%% 그래프 스타일 설정
axisFontSize = 14;
titleFontSize = 12;
legendFontSize = 12;
labelFontSize = 12;

%% 색상 매트릭스 정의 (시나리오 수에 맞게 확장 가능)
% 여기서는 10개의 시나리오에 대해 서로 다른 색상을 정의합니다.
c_mat = lines(num_scenarios);  % MATLAB의 기본 색상 매트릭스 사용

%% 전압 합성 및 DRT 추정
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
    
    %% 전압에 노이즈 추가
    rng(0);  % 노이즈의 재현성을 보장
    noise_level = 0.01;
    V_sd = V_est + noise_level * randn(size(V_est));  % V_sd = 합성된 측정 전압
    
    % 현재 시나리오의 V_sd 저장
    V_sd_all(s, :) = V_sd;  % 이 시나리오의 노이즈가 추가된 V_sd 저장
    
    %% W 행렬 구성
    W = zeros(length(t), n);  % W 행렬 초기화
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
    
    %% Quadprog를 이용한 정규화된 최소자승법 솔루션
    % 상수 제거: OCV와 R0*ik를 빼줍니다.
    y_adjusted = V_sd' - OCV - R0 * ik';
    
    % Quadprog를 위한 행렬 및 벡터 구성
    H = 2 * (W' * W + lambda * (L' * L));
    f = -2 * W' * y_adjusted;
    
    % 부등식 제약조건: gamma ≥ 0
    A_ineq = -eye(n);
    b_ineq = zeros(n, 1);
    
    % Quadprog 옵션 설정
    options = optimoptions('quadprog', 'Display', 'off');
    
    % Quadprog를 사용하여 최적화 문제 해결
    gamma_quadprog = quadprog(H, f, A_ineq, b_ineq, [], [], [], [], [], options);
    
    %% Quadprog로 구한 gamma 저장
    gamma_quadprog_all(s, :) = gamma_quadprog';
    
    %% 전압 및 DRT 비교 플롯
    figure(1);  
    subplot(5, 2, s);
    
    yyaxis left
    plot(t, ik, 'b-', 'LineWidth', 1.5);
    ylabel('Current (A)', 'FontSize', labelFontSize);
    xlabel('Time (s)', 'FontSize', labelFontSize);
    grid on;
    set(gca, 'FontSize', axisFontSize);
    
    yyaxis right
    plot(t, V_sd, 'r-', 'LineWidth', 1.5);
    ylabel('Voltage (V)', 'FontSize', labelFontSize);
    ylim([min(V_sd)-0.1, max(V_sd)+0.1]);
    set(gca, 'FontSize', axisFontSize);
    
    % 제목 업데이트 (올바른 진폭과 주기 포함)
    title(['Scenario ', num2str(s), ...
           ': A1=', num2str(A(s,1)), ', A2=', num2str(A(s,2)), ', A3=', num2str(A(s,3)), ...
           ', T1=', num2str(T(s,1)), ', T2=', num2str(T(s,2)), ', T3=', num2str(T(s,3))], ...
           'FontSize', titleFontSize);
    
    % 범례 추가
    legend({'Current (A)', 'Voltage (V)'}, 'Location', 'best', 'FontSize', legendFontSize);
    
    %% DRT 비교 플롯
    figure(1 + s);  % 각 시나리오에 대한 DRT 비교 그림
    hold on;
    
    % 실제 gamma 플롯
    plot(theta_discrete, gamma_discrete_true, 'k-', 'LineWidth', 1.5, 'DisplayName', 'True gamma');
    
    % Quadprog로 구한 gamma 플롯
    plot(theta_discrete, gamma_quadprog_all(s, :), ':', 'Color', c_mat(s, :), 'LineWidth', 1.5, 'DisplayName', ['Scenario ', num2str(s)]);
    
    hold off;
    xlabel('\theta = ln(\tau)', 'FontSize', labelFontSize);
    ylabel('\gamma', 'FontSize', labelFontSize);
    title(['DRT Comparison for Scenario ', num2str(s), ' (\lambda = ', num2str(lambda), ')'], ...
          'FontSize', titleFontSize);
    set(gca, 'FontSize', axisFontSize);
    legend('Location', 'Best', 'FontSize', legendFontSize);
    
end

%% 추가 그래프: Scenarios 6, 7, 8, 9

% 선택된 시나리오
selected_scenarios = [6, 7, 8, 9];

% Figure 2: I(t)와 V(t) for Scenarios 6,7,8,9 as Subplots
figure(2);
for idx = 1:length(selected_scenarios)
    s = selected_scenarios(idx);
    subplot(2, 2, idx);
    
    yyaxis left
    plot(t, ik_scenarios(s, :), 'Color', c_mat(s, :), 'LineWidth', 1.5);
    ylabel('Current (A)', 'FontSize', labelFontSize);
    yyaxis right
    plot(t, V_sd_all(s, :), 'Color', c_mat(s, :), 'LineWidth', 1.5);
    ylabel('Voltage (V)', 'FontSize', labelFontSize);
    xlabel('Time (s)', 'FontSize', labelFontSize);
    
    title(['Scenario ', num2str(s)], 'FontSize', titleFontSize);
    
    legend({'Current (A)', 'Voltage (V)'}, 'Location', 'best', 'FontSize', legendFontSize);
    
    set(gca, 'FontSize', axisFontSize);
end
sgtitle('Unimodal: Current and Voltage vs Time', 'FontSize', titleFontSize);

% Figure 3: gamma vs theta for Scenarios 6,7,8,9
figure(3);
hold on;
for idx = 1:length(selected_scenarios)
    s = selected_scenarios(idx);
    plot(theta_discrete, gamma_quadprog_all(s, :), '--', 'LineWidth', 1.5, ...
        'Color', c_mat(s, :), 'DisplayName', ['Scenario ', num2str(s)]);
end
plot(theta_discrete, gamma_discrete_true, 'k-', 'LineWidth', 2, 'DisplayName', 'True gamma');
hold off;
xlabel('\theta = ln(\tau [s])', 'FontSize', labelFontSize);
ylabel('\gamma', 'FontSize', labelFontSize);
title('Unimodal: Estimated gamma', 'FontSize', titleFontSize);
set(gca, 'FontSize', axisFontSize);
legend('Location', 'Best', 'FontSize', legendFontSize);
