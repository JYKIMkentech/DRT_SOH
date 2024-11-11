clc; clear; close all;

%% 파라미터 설정
n = 21;  % RC 요소의 수
t = 0:0.01:100;  % 시간 벡터
dt = t(2) - t(1);

% 합성 전류 데이터 생성 (사인파의 합)
A = 1; 
T1 = 1;
T2 = 5;
T3 = 20;
I1 = A * sin(2 * pi * t / T1);
I2 = A * sin(2 * pi * t / T2);
I3 = A * sin(2 * pi * t / T3);
ik = I1 + I2 + I3;  % 총 전류

% 실제 DRT를 위한 파라미터 (R_discrete)
mu = 10;
sigma = 5;
tau_discrete = linspace(0.01, 20, n);  % 이산 tau 값들

% 정규분포를 사용하여 실제 R_discrete 계산
R_discrete = normpdf(tau_discrete, mu, sigma);
R_discrete = R_discrete / max(R_discrete);  % 최대값이 1이 되도록 정규화

% 초기 전압 설정
V_est = zeros(1, length(t));  % 추정 전압
R0 = 0.1;  % 내부 저항
OCV = 0;   % 개방 회로 전압
V_RC = zeros(n, length(t));  % 각 요소의 RC 전압

%% 초기 전압 계산 (첫 번째 시간 스텝)
for i = 1:n
    V_RC(i, 1) = ik(1) * R_discrete(i) * (1 - exp(-dt / tau_discrete(i))); % ik(1) = 0 
end
V_est(1) = OCV + R0 * ik(1) + sum(V_RC(:, 1));

%% 이후 시간 스텝에 대한 이산 시간 전압 계산
for k = 2:length(t)
    for i = 1:n
        % 이전 시간 스텝을 기반으로 RC 전압 계산
        V_RC(i, k) = exp(-dt / tau_discrete(i)) * V_RC(i, k-1) + R_discrete(i) * (1 - exp(-dt / tau_discrete(i))) * ik(k);       
    end
    V_est(k) = OCV + R0 * ik(k) + sum(V_RC(:, k));
end

%% 전압에 노이즈 추가
noise_level = 0.01;
V_noisy = V_est + noise_level * randn(size(V_est));

%% 정규화 파라미터 및 미분 행렬 설정
lambda = 0.1;  % 필요에 따라 조정

% 1차 미분 행렬 L 생성
L = diff(eye(n));

%% fmincon을 사용한 최적화
% 초기값 설정
initial_R = ones(1, n);  % R_discrete에 대한 초기 추측
lb = zeros(1, n);  % 하한 (R은 음수가 될 수 없음)
ub = [];  % 상한 없음

% 최적화 옵션
options_fmincon = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'interior-point', 'MaxFunctionEvaluations', 1e5);

% 비용 함수 정의
cost_func = @(R) cost_function(R, tau_discrete, ik, V_noisy, dt, n, R0, OCV, t, lambda, L);

% 최적화 수행
R_optimized_fmincon = fmincon(cost_func, initial_R, [], [], [], [], lb, ub, [], options_fmincon);

%% quadprog를 사용한 최적화
% 사전 계산 상수
alpha_i = exp(-dt ./ tau_discrete);
beta_i = 1 - alpha_i;

% V 행렬 사전 계산
V_matrix = zeros(length(t), n);
for i = 1:n
    v_i = zeros(length(t), 1);
    v_i(1) = beta_i(i) * ik(1);
    for k = 2:length(t)
        v_i(k) = alpha_i(i) * v_i(k-1) + beta_i(i) * ik(k);
    end
    V_matrix(:, i) = v_i;
end

% y 계산
y = V_noisy' - OCV - R0 * ik';

% W^T W, W^T y, L^T L 계산
W = V_matrix;  % W는 V_matrix
WTW = W' * W;
WTy = W' * y;
LTL = L' * L;

% quadprog를 위한 이차 비용 함수 설정
H = 2 * (WTW + lambda * LTL);
f = -2 * WTy;

% quadprog가 최소화하는 비용 함수는 (1/2) x^T H x + f^T x 형태임

% 경계 설정 (R >= 0)
lb = zeros(n, 1);
ub = [];

% quadprog 옵션
options_qp = optimoptions('quadprog', 'Display', 'iter');

% quadprog를 사용하여 최적화 수행
[R_optimized_qp, fval_qp, exitflag_qp, output_qp] = quadprog(H, f, [], [], [], [], lb, ub, [], options_qp);

%% 해석적 해 계산 (제약 조건 없음)
R_analytical = (WTW + lambda * LTL) \ WTy;

% 음수 값을 0으로 설정 (비음수 제약 조건 적용)
R_analytical_nonneg = max(R_analytical, 0);

%% DRT 비교 플롯 (실제 DRT vs fmincon DRT vs quadprog DRT vs 해석적 해)
figure;
plot(tau_discrete, R_discrete, 'k-', 'LineWidth', 2);  % 실제 DRT (검은색 실선)
hold on;
stem(tau_discrete, R_discrete, 'ko', 'LineWidth', 1.5);  % 실제 DRT 점들 (검은색 원)

plot(tau_discrete, R_optimized_fmincon, 'b-', 'LineWidth', 2);  % fmincon으로 최적화된 DRT (파란색 실선)
stem(tau_discrete, R_optimized_fmincon, 'bo', 'LineWidth', 1.5);  % fmincon DRT 점들 (파란색 원)

plot(tau_discrete, R_optimized_qp, 'r-', 'LineWidth', 2);  % quadprog로 최적화된 DRT (빨간색 실선)
stem(tau_discrete, R_optimized_qp, 'ro', 'LineWidth', 1.5);  % quadprog DRT 점들 (빨간색 원)

plot(tau_discrete, R_analytical_nonneg, 'g-', 'LineWidth', 2);  % 해석적 해 (녹색 실선)
stem(tau_discrete, R_analytical_nonneg, 'go', 'LineWidth', 1.5);  % 해석적 해 점들 (녹색 원)

xlabel('\tau (Time Constant)');
ylabel('R (Resistance)');
legend('True DRT','', 'Optimized DRT (fmincon)', '', 'Optimized DRT (quadprog)', '', 'Analytical Solution', '');
title('True vs Optimized Distribution of Relaxation Times (DRT)');
grid on;

%% 함수들

% 주어진 R_discrete로 V_est를 계산하는 함수
function V_est = calculate_voltage(R_discrete, tau_discrete, ik, dt, n, R0, OCV, t)
    V_est = zeros(1, length(t));  % 추정 전압 초기화
    V_RC = zeros(n, length(t));  % 각 요소의 RC 전압

    % 초기 전압 계산 (첫 번째 시간 스텝)
    for i = 1:n
        V_RC(i, 1) = ik(1) * R_discrete(i) * (1 - exp(-dt / tau_discrete(i)));
    end
    V_est(1) = OCV + R0 * ik(1) + sum(V_RC(:, 1));

    % 이후 시간 스텝에 대한 이산 시간 전압 계산
    for k = 2:length(t)
        for i = 1:n
            V_RC(i, k) = exp(-dt / tau_discrete(i)) * V_RC(i, k-1) + R_discrete(i) * (1 - exp(-dt / tau_discrete(i))) * ik(k);
        end
        V_est(k) = OCV + R0 * ik(k) + sum(V_RC(:, k));
    end
end

% fmincon을 위한 비용 함수 (잔차 제곱합 및 정규화 포함)
function cost = cost_function(R_discrete, tau_discrete, ik, V_noisy, dt, n, R0, OCV, t, lambda, L)
    % 현재 R_discrete에 대한 추정 전압 계산
    V_est = calculate_voltage(R_discrete, tau_discrete, ik, dt, n, R0, OCV, t);
    
    % 잔차 계산 (노이즈가 추가된 전압과 추정 전압의 차이)
    residuals = V_noisy - V_est;
    data_fidelity = sum(residuals.^2);
    
    % 정규화 항목 (1차 미분 페널티)
    regularization = lambda * norm(L * R_discrete', 2)^2;
    
    % 총 비용
    cost = data_fidelity + regularization;
end
