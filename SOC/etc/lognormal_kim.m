clc; clear; close all;

% ln(tau) = theta 가 정규분포를 따르는 것
m = log(10);  % ln(tau)의 평균
v = 0.2;      % ln(tau)의 표준편차

% theta (ln(tau))의 범위 설정 (3 시그마)
theta_min = m - 3*v;  % 평균 - 3*표준편차
theta_max = m + 3*v;  % 평균 + 3*표준편차

theta_values = linspace(theta_min, theta_max, 1000); % ln(tau)을 일정 간격으로 나눔

% ln(tau)의 normal distribution 계산
normal_pdf = normpdf(theta_values, m, v);

% normal_pdf 최대값으로 정규화
normal_pdf = normal_pdf / max(normal_pdf);  % 최대값으로 정규화

% tau의 범위 설정 --> theta 기준으로 tau 범위 설정
tau_min = exp(theta_min);
tau_max = exp(theta_max);

tau_values = exp(theta_values); % tau = exp(theta)

% tau의 lognormal distribution 계산
lognormal_pdf = lognpdf(tau_values, m, v);

% 그래프 그리기
figure;

% 첫 번째 서브플롯: ln(tau)의 정규분포 (정규화 적용)
subplot(2,1,1);
plot(theta_values, normal_pdf,  'r-', 'LineWidth', 2);
xlabel('ln(\tau) = \theta');
ylabel('Normalized Probability Density');
title('Normalized Normal Distribution of ln(\tau) (Reference DRT)');
grid on;

% 두 번째 서브플롯: tau의 로그-정규분포 (변경 없음)
subplot(2,1,2);
plot(tau_values, lognormal_pdf, 'b-', 'LineWidth', 2);
xlabel('\tau');
ylabel('Probability Density');
title('Lognormal Distribution of \tau (DRT)');
grid on;

% 그래프 레이아웃 정규화 설정
set(gcf, 'Units', 'normalized', 'Position', [0.1, 0.1, 0.8, 0.8]);  % 화면 비율로 설정



