clc; clear; close all;

%% 1. Parameters
n = 21;                      % RC 요소의 개수 (DRT를 부드럽게 하기 위해 증가)
t = 0:0.01:100;              % 시간 벡터 (초)
dt = t(2) - t(1);            % 시간 간격
num_scenarios = 10;          % 전류 시나리오의 수
lambda = 0.0409;             % 정규화 파라미터

%% 2. Define Amplitudes and Periods for Current Synthesis
A = [1, 1, 1;          % 시나리오 1   
     1.7, 0.6, 0.7;    % 시나리오 2
     0.2, 0.5, 2.3;    % 시나리오 3
     1.3, 1.1, 0.6;    % 시나리오 4
     1.7, 1.8, 0.5;    % 시나리오 5
     1.27, 1.33, 0.4;  % 시나리오 6
     1.2, 1.6, 0.2;    % 시나리오 7
     0.9, 0.7, 2.4;    % 시나리오 8
     1.1, 1.1, 0.8;    % 시나리오 9
     0.1, 0.1, 2.8];   % 시나리오 10

T = [1, 5, 20;         % 시나리오 1
     2, 4, 20;         % 시나리오 2
     1, 20, 25;        % 시나리오 3
     1.5, 5.3, 19.8;   % 시나리오 4
     2.5, 4.2, 20.5;   % 시나리오 5
     1.5, 20.9, 24.2;  % 시나리오 6
     1.3, 6, 19.3;     % 시나리오 7
     2.2, 4.8, 20.2;   % 시나리오 8
     2, 20.8, 26.1;    % 시나리오 9
     1.1, 4.3, 20.1];  % 시나리오 10

%% 3. Generate Synthetic Current Data (Multi-Sine Approach)
ik_scenarios = zeros(num_scenarios, length(t)); % 전류 시나리오 초기화

for s = 1:num_scenarios
    % 각 시나리오에 대한 세 개의 사인파 합산
    ik_scenarios(s, :) = A(s,1)*sin(2*pi*t / T(s,1)) + ...
                         A(s,2)*sin(2*pi*t / T(s,2)) + ...
                         A(s,3)*sin(2*pi*t / T(s,3));
end

%% 4. Define Time Constants on Logarithmic Scale (Natural Log)
% 로그정규분포의 파라미터 설정 (tau의 평균과 표준편차)
mu_tau = 10;        % tau의 평균
sigma_tau = 5;      % tau의 표준편차
Var_tau = sigma_tau^2;  % tau의 분산

% 로그 정규 분포에서 정규 분포의 파라미터(mu_log, sigma_log) 유도
sigma_sq = log(1 + (Var_tau / mu_tau^2));  % sigma_log^2 = ln(1 + (Var(tau) / E(tau)^2))
sigma_log = sqrt(sigma_sq);                % sigma_log = sqrt(sigma_sq)
mu_log = log(mu_tau) - (sigma_sq / 2);     % mu_log = ln(E(tau)) - sigma_log^2 / 2

% 계산된 mu_log와 sigma_log 출력 (디버깅용)
fprintf('Calculated mu_log: %.4f\n', mu_log);
fprintf('Calculated sigma_log: %.4f\n', sigma_log);

% tau의 범위 설정 (양수 범위 내에서 적절하게 설정)
a = 0.01;           % tau의 최소값
b = 100;            % tau의 최대값

% theta_j와 tau_discrete_log의 정의
theta_j = linspace(log(a), log(b), n);    % 로그 공간에서 선형으로 분포된 theta
tau_discrete_log = exp(theta_j);          % tau_j = e^theta_j

%% 5. True DRT Parameters (Gamma Function over ln(tau))
% 원래의 로그 정규 분포 f_tau(tau_j) 계산
f_tau = lognpdf(tau_discrete_log, mu_log, sigma_log);

% 변환된 분포 gamma(theta_j) 계산
gamma_theta = f_tau .* tau_discrete_log;    % gamma(theta) = f_tau(tau_j) * tau_j

% gamma(theta_j)를 정규화하여 전체 면적이 1이 되도록 함
R_discrete_true_log = gamma_theta / trapz(theta_j, gamma_theta);

%% 6. Define Regularization Matrix L (1st Order Difference)
% 1차 차분 행렬 L 생성
L = zeros(n-1, n);
for i = 1:n-1
    L(i, i) = -1;
    L(i, i+1) = 1;
end

%% 7. Initialize Storage for Results
R_estimated_analytic = zeros(num_scenarios, n);   % 분석적 DRT 추정치 저장
V_est_all = zeros(num_scenarios, length(t));      % 모든 시나리오에 대한 추정 전압
V_sd_all = zeros(num_scenarios, length(t));       % 모든 시나리오에 대한 합성 측정 전압

%% 8. Voltage Synthesis and Plotting Setup
figure(1);  
sgtitle('각 시나리오의 전류 및 전압');

% 각 시나리오에 대한 전류 플롯
for s = 1:num_scenarios
    subplot(5, 2, s);
    yyaxis left
    plot(t, ik_scenarios(s, :), 'b-', 'LineWidth', 1.5);
    ylabel('전류 (A)');
    xlabel('시간 (s)');
    grid on;
end

%% 9. Processing Each Scenario (Logarithmic Time Scale & Analytical Solution)
for s = 1:num_scenarios
    fprintf('시나리오 처리 중 %d/%d...\n', s, num_scenarios);
    
    % 시나리오에 대한 전류 (m x 1)
    ik = ik_scenarios(s, :)';  % 열 벡터로 변환 (m x 1)
    
    %% 9.1. Initialize Voltage
    V_est = zeros(length(t),1);      % n-RC 모델을 통한 추정 전압 (m x 1)
    R0 = 0.1;                         % 오믹 저항 (Ohms)
    OCV = 3.7;                        % 개방 회로 전압 (V)
    V_RC = zeros(n, length(t));       % 각 요소에 대한 RC 전압 (n x m)
    
    %% 9.2. Initial Voltage Calculation (First Time Step)
    for i = 1:n
        V_RC(i, 1) = ik(1) * R_discrete_true_log(i) * (1 - exp(-dt / tau_discrete_log(i)));
    end
    V_est(1) = OCV + R0 * ik(1) + sum(V_RC(:, 1));
    
    %% 9.3. Voltage Calculation for t > 1
    for k_idx = 2:length(t)
        for i = 1:n
            % RC 전압을 이전 시간 단계의 데이터로 계산
            V_RC(i, k_idx) = exp(-dt / tau_discrete_log(i)) * V_RC(i, k_idx-1) + ...
                              R_discrete_true_log(i) * (1 - exp(-dt / tau_discrete_log(i))) * ik(k_idx);       
        end
        V_est(k_idx) = OCV + R0 * ik(k_idx) + sum(V_RC(:, k_idx));
    end
    
    % 현재 시나리오에 대한 V_est 저장
    V_est_all(s, :) = V_est';  % 행 벡터로 저장 (1 x m)
    
    %% 9.4. Add Noise to the Voltage
    rng(s);  % 각 시나리오에 대한 노이즈 재현성 보장
    noise_level = 0.01;
    V_sd = V_est + noise_level * randn(size(V_est));  % 합성 측정 전압 (m x 1)
    
    % 현재 시나리오에 대한 V_sd 저장
    V_sd_all(s, :) = V_sd';  % 행 벡터로 저장 (1 x m)
    
    %% 9.5. Construct System Matrix W for Logarithmic Time Scale
    W = zeros(length(t), n);  % W 행렬 초기화 (m x n)
    for j = 1:n
        for k_idx = 1:length(t)
            if k_idx == 1
                W(k_idx,j) = ik(k_idx) * (1 - exp(-dt / tau_discrete_log(j)));
            else
                W(k_idx,j) = exp(-dt / tau_discrete_log(j)) * W(k_idx-1,j) + ...
                            ik(k_idx) * (1 - exp(-dt / tau_discrete_log(j)));
            end
        end
    end
    
    %% 9.6. Analytical Solution with Regularization using L
    % (W^T W + lambda * L^T L) 계산
    A_matrix = W' * W + lambda * (L' * L);  % (n x n)
    
    % W^T (V_sd - OCV - I * R0) 계산
    b_vector = W' * (V_sd - OCV - ik * R0);  % (n x 1)
    
    % 분석적 해법을 통한 R 계산
    R_analytic = A_matrix \ b_vector;         % (n x 1)
    
    % 음수 값 제거 (비음수 강제)
    R_analytic(R_analytic < 0) = 0;
    
    % 분석적 DRT 저장
    R_estimated_analytic(s, :) = R_analytic';  % 행 벡터로 저장 (1 x n)
    
    %% 9.7. Plot Voltage on Existing Subplots
    subplot(5, 2, s);
    yyaxis right
    plot(t, V_sd, 'r-', 'LineWidth', 1.5);
    ylabel('전압 (V)');
    ylim([min(V_sd)-0.1, max(V_sd)+0.1]);
    
    % 올바른 진폭과 주기를 포함한 제목 업데이트
    title(['시나리오 ', num2str(s), ...
           ': A1=', num2str(A(s,1)), ', A2=', num2str(A(s,2)), ', A3=', num2str(A(s,3)), ...
           ', T1=', num2str(T(s,1)), ', T2=', num2str(T(s,2)), ', T3=', num2str(T(s,3))]);
    
    % 범례 추가
    legend({'전류 (A)', '전압 (V)'}, 'Location', 'best');
end

%% 10. Plot the DRT Comparison for Each Scenario
for s = 1:num_scenarios
    figure(1 + s);  % 각 시나리오에 대한 DRT 비교 그림
    hold on;
    
    % True DRT vs ln(tau) 플롯
    plot(theta_j, R_discrete_true_log, 'k-', 'LineWidth', 2, 'DisplayName', 'True DRT');
    
    % 분석적 DRT vs ln(tau) 플롯
    plot(theta_j, R_estimated_analytic(s, :), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Analytical DRT');
    
    hold off;
    xlabel('ln(\tau) (로그 시간 상수)');
    ylabel('\gamma(\theta)');
    title(['시나리오 ', num2str(s), '에 대한 DRT 비교 (\lambda = ', num2str(lambda), ')']);
    legend('Location', 'BestOutside');
    grid on;
end

%% 11. Summary Plot Comparing Analytical Solution Across All Scenarios
figure;
hold on;
colors = lines(num_scenarios);

for s = 1:num_scenarios
    plot(theta_j, R_estimated_analytic(s, :), '--', 'Color', colors(s,:), 'LineWidth', 1, 'DisplayName', ['Analytical S', num2str(s)]);
end

% True DRT vs ln(tau) 플롯
plot(theta_j, R_discrete_true_log, 'k-', 'LineWidth', 2, 'DisplayName', 'True DRT');

xlabel('ln(\tau) (로그 시간 상수)');
ylabel('\gamma(\theta)');
title('모든 시나리오에 대한 DRT 추정 비교 (분석적 해법)');
legend('Location', 'BestOutside');
grid on;
hold off;

%% 12. Verify Dimensions (Optional)
disp('행렬의 크기:');
disp(['W: ', mat2str(size(W))]);
disp(['R_estimated_analytic: ', mat2str(size(R_estimated_analytic))]);
disp(['L: ', mat2str(size(L))]);


