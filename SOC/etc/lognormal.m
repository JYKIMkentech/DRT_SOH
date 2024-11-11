% 주어진 평균과 표준 편차
mean_lognormal = 1.0;
std_lognormal = 0.5;

% 로그정규 분포의 매개변수 계산
sigma_squared = log(1 + (std_lognormal^2) / (mean_lognormal^2));
sigma = sqrt(sigma_squared);
mu = log(mean_lognormal) - (sigma_squared / 2);

fprintf('계산된 매개변수:\n');
fprintf('mu = %.5f\n', mu);
fprintf('sigma = %.5f\n', sigma);

% 로그정규 분포 객체 생성
pd = makedist('Lognormal', 'mu', mu, 'sigma', sigma);

% 샘플 데이터 생성
num_samples = 10000;
tau = random(pd, num_samples, 1);

% 로그정규 분포의 PDF 그리기
x = linspace(min(tau), max(tau), 1000);
y = pdf(pd, x);

figure;
subplot(2,1,1);
plot(x, y, 'r-', 'LineWidth', 2);
xlabel('\tau');
ylabel('Resistance');
title('lognormal distribution of \tau');
grid on;

% ln(tau) 계산
ln_tau = log(tau);

% ln(tau)에 대한 정규 분포 매개변수
mu_ln = mu;
sigma_ln = sigma;

% 정규 분포의 PDF 계산
x_ln = linspace(min(ln_tau), max(ln_tau), 1000);
y_ln = pdf('Normal', x_ln, mu_ln, sigma_ln);

% ln(tau)의 정규 분포 곡선 그리기
subplot(2,1,2);
plot(x_ln, y_ln, 'b-', 'LineWidth', 2);
xlabel('ln(\tau)');
ylabel('Gamma');
title('normal distribution of ln(\tau)');
grid on;

% 추가: 로그정규 분포의 특성 확인
computed_mean = mean(tau);
computed_std = std(tau);
fprintf('샘플로부터 계산된 평균: %.5f\n', computed_mean);
fprintf('샘플로부터 계산된 표준 편차: %.5f\n', computed_std);

% ln(tau)의 특성 확인
computed_mean_ln = mean(ln_tau);
computed_std_ln = std(ln_tau);
fprintf('ln(tau) 샘플의 평균: %.5f\n', computed_mean_ln);
fprintf('ln(tau) 샘플의 표준 편차: %.5f\n', computed_std_ln);

