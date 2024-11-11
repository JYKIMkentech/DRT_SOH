clc; clear; close all;
%%
% 세타 (ln (tau)) 가 정규분포를 따른다 ---> tau는 로그정규분포를 따른다.

% Theta = ln(tau) (x축)
% gamma(theta) = [ R(exp(theta)) * exp(theta) ] = [ R(tau) * tau ] (y축)
% R_i = gamma_i * delta theta % 면적은 저항 = gamma (세로) * delta (가로, 일정하게)

%% AS1.mat 파일 로드
load('AS1.mat');  % A, T, ik_scenarios, t 변수를 불러옵니다.

%% Parameters 
n = 40;  % 이산화 요소의 개수
num_scenarios = 10;  % 전류 시나리오의 수
lambda = 0.58;  % 정규화 파라미터
N_resample = 200;  % 부트스트랩 샘플 수

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

% 실제 gamma 분포 (True DRT)
gamma_discrete_true = (1/(sigma_theta * sqrt(2*pi))) * exp(- (theta_discrete - mu_theta).^2 / (2 * sigma_theta^2));

% gamma를 최대값이 1이 되도록 정규화
gamma_discrete_true = gamma_discrete_true / max(gamma_discrete_true);

%% 일차 차분 행렬 L
L = zeros(n-1, n);
for i = 1:n-1
    L(i, i) = -1;
    L(i, i+1) = 1;
end

%% 전압 및 DRT 추정 저장 변수 초기화
gamma_original_all = zeros(num_scenarios, n);  % 원본 데이터로부터 구한 gamma 저장
V_est_all = zeros(num_scenarios, length(t));  % 각 시나리오의 V_est 저장
V_sd_all = zeros(num_scenarios, length(t));   % 각 시나리오의 V_sd 저장

%% 전압 합성 및 DRT 추정
for s = 1:num_scenarios
    fprintf('Processing Scenario %d/%d...\n', s, num_scenarios);
    
    % 현재 시나리오의 전류
    ik_original = ik_scenarios(s, :);  % 원본 전류 시나리오 사용
    t_original = t;  % 원본 시간 벡터
    
    % 시간 간격 계산 (dt_original)
    dt_original = t_original(2:end) - t_original(1:end-1);  % dt(k) = t(k+1) - t(k)
    
    %% 전압 초기화
    V_est = zeros(1, length(t_original));  % n-요소 모델을 통한 모델 전압 계산
    R0 = 0.1;  % 저항 (오움)
    OCV = 0;   % 개방 회로 전압
    V_RC = zeros(n, length(t_original));  % 각 요소의 전압
    
    %% 전압 계산 (원본 데이터로)
    for k_idx = 1:length(t_original)
        if k_idx == 1
            dt_k = dt_original(1);  % 첫 번째 dt
            for i = 1:n
                V_RC(i, k_idx) = gamma_discrete_true(i) * delta_theta * ik_original(k_idx) * (1 - exp(-dt_k / tau_discrete(i)));
            end
        elseif k_idx < length(t_original)
            dt_k = dt_original(k_idx);  % dt(k) = t(k+1) - t(k)
            for i = 1:n
                V_RC(i, k_idx) = V_RC(i, k_idx-1) * exp(-dt_k / tau_discrete(i)) + ...
                                 gamma_discrete_true(i) * delta_theta * ik_original(k_idx) * (1 - exp(-dt_k / tau_discrete(i)));
            end
        else
            dt_k = dt_original(end);  % 마지막 dt
            for i = 1:n
                V_RC(i, k_idx) = V_RC(i, k_idx-1) * exp(-dt_k / tau_discrete(i)) + ...
                                 gamma_discrete_true(i) * delta_theta * ik_original(k_idx) * (1 - exp(-dt_k / tau_discrete(i)));
            end
        end
        V_est(k_idx) = OCV + R0 * ik_original(k_idx) + sum(V_RC(:, k_idx));
    end
    
    % 현재 시나리오의 V_est 저장
    V_est_all(s, :) = V_est;  % 이 시나리오의 계산된 V_est 저장
    
    %% 전압에 노이즈 추가
    rng(0);  % 노이즈의 재현성을 보장
    noise_level = 0.01;
    V_sd = V_est + noise_level * randn(size(V_est));  % V_sd = 합성된 측정 전압
    
    % 현재 시나리오의 V_sd 저장
    V_sd_all(s, :) = V_sd;  % 이 시나리오의 노이즈가 추가된 V_sd 저장
    
    %% 원본 데이터로부터 gamma 추정 (DRT Original)
    % W 행렬 구성
    W_original = zeros(length(t_original), n);
    for k_idx = 1:length(t_original)
        if k_idx == 1
            dt_k = dt_original(1);
            for i = 1:n
                W_original(k_idx, i) = ik_original(k_idx) * (1 - exp(-dt_k / tau_discrete(i))) * delta_theta;
            end
        elseif k_idx < length(t_original)
            dt_k = dt_original(k_idx);
            for i = 1:n
                W_original(k_idx, i) = W_original(k_idx-1, i) * exp(-dt_k / tau_discrete(i)) + ...
                                       ik_original(k_idx) * (1 - exp(-dt_k / tau_discrete(i))) * delta_theta;
            end
        else
            dt_k = dt_original(end);
            for i = 1:n
                W_original(k_idx, i) = W_original(k_idx-1, i) * exp(-dt_k / tau_discrete(i)) + ...
                                       ik_original(k_idx) * (1 - exp(-dt_k / tau_discrete(i))) * delta_theta;
            end
        end
    end
    
    % 상수 제거: OCV와 R0*ik를 빼줍니다.
    y_adjusted_original = V_sd' - OCV - R0 * ik_original';
    
    % Quadprog를 위한 행렬 및 벡터 구성
    H_original = 2 * (W_original' * W_original + lambda * (L' * L));
    f_original = -2 * W_original' * y_adjusted_original;
    
    % 부등식 제약조건: gamma ≥ 0
    A_ineq = -eye(n);
    b_ineq = zeros(n, 1);
    
    % Quadprog 옵션 설정
    options = optimoptions('quadprog', 'Display', 'off');
    
    % Quadprog를 사용하여 최적화 문제 해결
    gamma_original = quadprog(H_original, f_original, A_ineq, b_ineq, [], [], [], [], [], options);
    
    % 원본 데이터로부터 구한 gamma 저장
    gamma_original_all(s, :) = gamma_original';
    
    %% 부트스트랩을 통한 gamma 추정 (DRT Resample)
    gamma_resample = zeros(N_resample, n);  % 부트스트랩으로 추정한 gamma 저장
    
    for b = 1:N_resample
        % 부트스트랩 샘플링 (복원 추출)
        idx_bootstrap = randsample(length(t_original), length(t_original), true);
        t_bootstrap = t_original(idx_bootstrap);
        ik_bootstrap = ik_original(idx_bootstrap);
        V_sd_bootstrap = V_sd(idx_bootstrap);
        
        % 중복된 시간 점을 제거하고 고유한 시간 점 찾기
        [t_bootstrap_unique, unique_idx] = unique(t_bootstrap);
        ik_bootstrap_unique = ik_bootstrap(unique_idx);
        V_sd_bootstrap_unique = V_sd_bootstrap(unique_idx);
        
        % 시간과 데이터를 시간 순서대로 정렬
        [t_bootstrap_sorted, sort_idx] = sort(t_bootstrap_unique);
        ik_bootstrap_sorted = ik_bootstrap_unique(sort_idx);
        V_sd_bootstrap_sorted = V_sd_bootstrap_unique(sort_idx);
        
        % 시간 간격 계산
        if length(t_bootstrap_sorted) > 1
            dt_bootstrap = t_bootstrap_sorted(2:end) - t_bootstrap_sorted(1:end-1);  % dt(k) = t(k+1) - t(k)
        else
            dt_bootstrap = dt_original(1);  % 기본값으로 설정
        end
        
        %% W 행렬 구성 (정렬된 부트스트랩 데이터로)
        W_bootstrap = zeros(length(t_bootstrap_sorted), n);  % W 행렬 초기화
        for k_idx = 1:length(t_bootstrap_sorted)
            if k_idx == 1
                dt_k = dt_bootstrap(1);
                for i = 1:n
                    W_bootstrap(k_idx, i) = ik_bootstrap_sorted(k_idx) * (1 - exp(-dt_k / tau_discrete(i))) * delta_theta;
                end
            elseif k_idx < length(t_bootstrap_sorted)
                dt_k = dt_bootstrap(k_idx);
                for i = 1:n
                    W_bootstrap(k_idx, i) = W_bootstrap(k_idx-1, i) * exp(-dt_k / tau_discrete(i)) + ...
                                      ik_bootstrap_sorted(k_idx) * (1 - exp(-dt_k / tau_discrete(i))) * delta_theta;
                end
            else
                dt_k = dt_bootstrap(end);
                for i = 1:n
                    W_bootstrap(k_idx, i) = W_bootstrap(k_idx-1, i) * exp(-dt_k / tau_discrete(i)) + ...
                                      ik_bootstrap_sorted(k_idx) * (1 - exp(-dt_k / tau_discrete(i))) * delta_theta;
                end
            end
        end
        
        %% Quadprog를 이용한 정규화된 최소자승법 솔루션 (부트스트랩 데이터로)
        % 상수 제거: OCV와 R0*ik를 빼줍니다.
        y_adjusted_bootstrap = V_sd_bootstrap_sorted' - OCV - R0 * ik_bootstrap_sorted';
        
        % Quadprog를 위한 행렬 및 벡터 구성
        H = 2 * (W_bootstrap' * W_bootstrap + lambda * (L' * L));
        f = -2 * W_bootstrap' * y_adjusted_bootstrap;
        
        % Quadprog를 사용하여 최적화 문제 해결
        gamma_quadprog = quadprog(H, f, A_ineq, b_ineq, [], [], [], [], [], options);
        
        %% 부트스트랩으로 구한 gamma 저장
        gamma_resample(b, :) = gamma_quadprog';
    end
    
    %% gamma 차이 계산 및 신뢰 구간 구하기
    gamma_diff = gamma_resample - gamma_original';  % 각 세타에 대해 차이 계산
    
    % 5% 및 95% 백분위수 계산
    gamma_diff_lower = prctile(gamma_diff, 5, 1);  % 5% 백분위수
    gamma_diff_upper = prctile(gamma_diff, 95, 1); % 95% 백분위수
    
   
    %% 전압 및 DRT 비교 플롯
    figure(1);  
    subplot(5, 2, s);
    yyaxis left
    plot(t_original, ik_original, 'b-', 'LineWidth', 1.5);
    ylabel('Current (A)');
    xlabel('Time (s)');
    grid on;
    
    yyaxis right
    plot(t_original, V_sd, 'r-', 'LineWidth', 1.5);
    ylabel('Voltage (V)');
    ylim([min(V_sd)-0.1, max(V_sd)+0.1]);
    
    % 제목 업데이트 (올바른 진폭과 주기 포함)
    title(['Scenario ', num2str(s), ...
           ': A1=', num2str(A(s,1)), ', A2=', num2str(A(s,2)), ', A3=', num2str(A(s,3)), ...
           ', T1=', num2str(T(s,1)), ', T2=', num2str(T(s,2)), ', T3=', num2str(T(s,3))]);
    
    % 범례 추가
    legend({'Current (A)', 'Voltage (V)'}, 'Location', 'best');
    
    % DRT comparison plot for each scenario
    figure(1 + s);  % 각 시나리오의 DRT 비교를 위한 새 figure 생성
    hold on;
    
    % True gamma plot (True DRT) as a black solid line
    plot(theta_discrete, gamma_discrete_true, 'k-', 'LineWidth', 3, 'DisplayName', 'True');
    
    % Original gamma values with square markers, outline only in red
    plot(theta_discrete, gamma_original_all(s, :), 's-', ...
         'MarkerFaceColor', 'none', ...      % 마커 내부를 투명하게 설정
         'MarkerEdgeColor', 'r', ...        % 마커 외곽선을 빨간색으로 설정
         'Color', 'r', ...                  % 선 색상을 빨간색으로 설정
         'LineWidth', 1, ...
         'DisplayName', 'Est');
    
    % Error bars for resampling results with red color
    errorbar(theta_discrete, gamma_original_all(s, :), -gamma_diff_lower, gamma_diff_upper, 's', ...
             'MarkerFaceColor', 'none', ...    % 마커 내부를 투명하게 설정
             'MarkerEdgeColor', 'r', ...      % 마커 외곽선을 빨간색으로 설정
             'Color', 'r', ...                % 에러바 색상을 빨간색으로 설정
             'LineWidth', 1.5, ...
             'HandleVisibility', 'off');        % 레전드에 포함되지 않도록 설정
    
    hold off;
    xlabel('\theta = ln(\tau)');
    ylabel('\gamma');
    title(['DRT Comparison for Scenario ', num2str(s), ' (\lambda = ', num2str(lambda), ')']);
    legend('True', 'Est', 'Location', 'Best');  % 레전드를 "True"와 "Est"로 설정


    
end
