clc; clear; close all;

%% Parameters 
n_values = [3, 5, 21];   % 다양한 n 값 (RC 요소의 수)
t = 0:0.01:100;           % 시간 벡터 (고정)
dt = t(2) - t(1);         % 시간 간격 (고정)
num_scenarios = 1;        % 전류 시나리오 수 (고정)
lambda = 0.1;             % 정규화 파라미터 (두 번째 스크립트와 일치)
noise_level = 0.01;       % 전압 측정 노이즈 수준 (고정)

%% 전류 합성용 진폭 및 주기 정의 (고정)
A = [1, 1, 1];            % 시나리오 1
% A = [1.7, 0.6, 0.7];    % 시나리오 2
% ...                   % 추가 시나리오 주석 처리

T = [1, 5, 20];           % 시나리오 1
% T = [2, 4, 20];        % 시나리오 2
% ...                   % 추가 시나리오 주석 처리

%% 다중 사인파 접근 방식을 이용한 합성 전류 데이터 생성 (고정)
ik_scenarios = zeros(num_scenarios, length(t)); % 전류 시나리오 초기화

for s = 1:num_scenarios
    % 각 시나리오에 대해 세 개의 사인파 합산
    ik_scenarios(s, :) = A(s,1)*sin(2*pi*t / T(s,1)) + ...
                         A(s,2)*sin(2*pi*t / T(s,2)) + ...
                         A(s,3)*sin(2*pi*t / T(s,3));
end

%% DRT 결과 저장 초기화
DRT_results = cell(length(n_values), num_scenarios); % 각 n과 시나리오에 대한 DRT 저장

%% 다양한 n 값에 대한 반복
for n_idx = 1:length(n_values)
    n = n_values(n_idx); % 현재 n 값
    fprintf('Processing for n = %d...\n', n);
    
    %% tau 이산 값 정의 (현재 n에 기반)
    tau_discrete = linspace(0.01, 20, n);  % 이산 tau 값
    
    %% 실제 DRT 파라미터 (R_discrete_true)
    mu = 10;
    sigma = 5;
    R_discrete_true = normpdf(tau_discrete, mu, sigma);
    R_discrete_true = R_discrete_true / max(R_discrete_true);  % 최대값 1로 정규화
    
    %% 1차 미분 행렬 L 정의 (현재 n에 기반)
    L = zeros(n-1, n);
    for i_L = 1:n-1
        L(i_L, i_L) = -1;
        L(i_L, i_L+1) = 1;
    end
    
    %% 각 시나리오에 대한 반복
    for s = 1:num_scenarios
        fprintf('  Scenario %d/%d...\n', s, num_scenarios);
        
        % 현재 시나리오의 전류
        ik = ik_scenarios(s, :);  % 현재 시나리오 입력 전류
        
        %% 전압 초기화
        V_est = zeros(1, length(t));      % 모델 전압 (n-RC 모델을 통해 계산)
        R0 = 0.1;                          % 내부 저항 (옴)
        OCV = 0;                           % 개방 회로 전압
        V_RC = zeros(n, length(t));        % 각 RC 요소의 전압
        
        %% 초기 전압 계산 (첫 번째 시간 스텝)
        for i = 1:n
            V_RC(i, 1) = ik(1) * R_discrete_true(i) * (1 - exp(-dt / tau_discrete(i)));
        end
        V_est(1) = OCV + R0 * ik(1) + sum(V_RC(:, 1));
        
        %% 이후 시간 스텝에 대한 전압 계산
        for k_idx = 2:length(t)
            for i = 1:n
                % 이전 시간 스텝을 기반으로 RC 전압 계산
                V_RC(i, k_idx) = exp(-dt / tau_discrete(i)) * V_RC(i, k_idx-1) + ...
                                  R_discrete_true(i) * (1 - exp(-dt / tau_discrete(i))) * ik(k_idx);       
            end
            V_est(k_idx) = OCV + R0 * ik(k_idx) + sum(V_RC(:, k_idx));
        end
        
        %% 전압에 노이즈 추가
        rng(0);  % 노이즈 재현성 보장
        V_sd = V_est + noise_level * randn(size(V_est));  % V_sd = 합성 측정 전압
        
        %% W 행렬 구성
        W = zeros(length(t), n);  % W 행렬 초기화
        for k_idx = 1:length(t)
            for i = 1:n
                if k_idx == 1
                    W(k_idx, i) = ik(k_idx) * (1 - exp(-dt / tau_discrete(i)));
                else
                    W(k_idx, i) = exp(-dt / tau_discrete(i)) * W(k_idx-1, i) + ...
                                  ik(k_idx) * (1 - exp(-dt / tau_discrete(i)));
                end
            end
        end
        
        %% 분석적 해 계산 (정규화 포함)
        % 상수 제거: OCV와 R0*ik를 빼줌
        y_adjusted = V_sd' - OCV - R0 * ik';
        
        % 정규화된 선형 방정식 풀이
        R_analytical = (W' * W + lambda * (L' * L)) \ (W' * y_adjusted);
        R_analytical(R_analytical < 0) = 0;  % 비음수 강제
        
        %% DRT 결과 저장
        DRT_results{n_idx, s} = R_analytical';
    end
end

%% 각 시나리오에 대한 DRT 비교 플롯 (다양한 n 값과 실제 DRT)
for s = 1:num_scenarios
    figure(s);
    hold on;
    
    % 실제 DRT 플롯
    n_default_idx = find(n_values == 21); % n=21을 기준/reference로 가정
    tau_discrete_default = linspace(0.01, 20, n_values(n_default_idx));
    R_discrete_true_default = normpdf(tau_discrete_default, 10, 5);
    R_discrete_true_default = R_discrete_true_default / max(R_discrete_true_default);
    plot(tau_discrete_default, R_discrete_true_default, 'k-', 'LineWidth', 2, 'DisplayName', 'True DRT');
    
    % 각 n에 대한 분석적 DRT 플롯
    colors = lines(length(n_values));
    for n_idx = 1:length(n_values)
        n = n_values(n_idx);
        tau_discrete = linspace(0.01, 20, n);
        R_analytical = DRT_results{n_idx, s};
        plot(tau_discrete, R_analytical, 'LineWidth', 1.5, 'Color', colors(n_idx,:), 'DisplayName', ['n = ', num2str(n)]);
    end
    
    hold off;
    xlabel('\tau (Time Constant)');
    ylabel('R (Resistance)');
    title(['DRT Comparison for Scenario ', num2str(s)]);
    legend('Location', 'BestOutside');
    grid on;
end

%% 함수들

% 주어진 R_discrete로 V_est를 계산하는 함수
function V_est = calculate_voltage(R_discrete, tau_discrete, ik, dt, n, R0, OCV, t)
    V_est = zeros(1, length(t));      % 추정 전압 초기화
    V_RC = zeros(n, length(t));       % 각 RC 요소의 전압 초기화

    % 초기 전압 계산 (첫 번째 시간 스텝)
    for i = 1:n
        V_RC(i, 1) = ik(1) * R_discrete(i) * (1 - exp(-dt / tau_discrete(i)));
    end
    V_est(1) = OCV + R0 * ik(1) + sum(V_RC(:, 1));

    % 이후 시간 스텝에 대한 전압 계산
    for k = 2:length(t)
        for i = 1:n
            V_RC(i, k) = exp(-dt / tau_discrete(i)) * V_RC(i, k-1) + ...
                         R_discrete(i) * (1 - exp(-dt / tau_discrete(i))) * ik(k);
        end
        V_est(k) = OCV + R0 * ik(k) + sum(V_RC(:, k));
    end
end


