clc; clear; close all;

%% Description

% Theta = ln(tau) (x축)
% gamma(theta) = [ R(exp(theta)) * exp(theta) ] = [ R(tau) * tau ] (y축)
% R_i = gamma_i * delta theta

% True n = 201;

%% Graphic

axisFontSize = 14;
titleFontSize = 12;
legendFontSize = 12;
labelFontSize = 12;

%% Load Data

file_path = 'G:\공유 드라이브\Battery Software Lab\Projects\DRT\SD\';
mat_files = dir(fullfile(file_path, '*.mat'));

for file = mat_files'
    load(fullfile(file_path, file.name));
end

%% Parameters

% 데이터 세트 목록 및 이름
AS_structs = {AS1_1per, AS1_2per, AS2_1per, AS2_2per};
AS_names = {'AS1_1per', 'AS1_2per', 'AS2_1per', 'AS2_2per'};
Gamma_structs = {Gamma_unimodal, Gamma_unimodal, Gamma_bimodal, Gamma_bimodal};

% 사용자가 처리할 데이터 세트 선택
fprintf('Available datasets:\n');
for idx = 1:length(AS_names)
    fprintf('%d: %s\n', idx, AS_names{idx});
end
dataset_idx = input('Select a dataset to process (enter the number): ');

% 선택된 데이터 세트로 설정
AS_data = AS_structs{dataset_idx};
AS_name = AS_names{dataset_idx};
Gamma_data = Gamma_structs{dataset_idx};
lambda = 0.518;  % 람다 값 설정

num_scenarios = length(AS_data);  % 시나리오 수 (예: 10개)
c_mat = lines(num_scenarios);  % 시나리오 수에 따라 색상 매트릭스 생성
n = length(Gamma_data.theta);  % RC 요소의 개수 (예: 201개)
OCV = 0;
R0 = 0.1;

% 일차 차분 행렬 L 생성 (정규화에 사용)
L = zeros(n-1, n);
for i = 1:n-1
    L(i, i) = -1;
    L(i, i+1) = 1;
end

%% DRT and Uncertainty Estimation

% theta 및 tau 설정
theta_discrete = Gamma_data.theta';
gamma_discrete_true = Gamma_data.gamma';
tau_discrete = exp(theta_discrete);
delta_theta = theta_discrete(2) - theta_discrete(1);

% DRT 추정 결과를 저장할 변수
gamma_quadprog_all = zeros(num_scenarios, n);
V_est_all = [];
V_sd_all = [];
ik_scenarios = [];

% 불확실성 저장을 위한 변수
num_resamples = 200;  % 부트스트랩 반복 횟수
gamma_resample_all_scenarios = zeros(num_scenarios, num_resamples, n);
gamma_lower_all = zeros(num_scenarios, n);
gamma_upper_all = zeros(num_scenarios, n);

for s = 1:num_scenarios
    fprintf('Processing %s Scenario %d/%d...\n', AS_name, s, num_scenarios);

    % 현재 시나리오의 데이터 가져오기
    V_sd = AS_data(s).V;    % 측정 전압 (열 벡터)
    ik = AS_data(s).I;      % 전류 (열 벡터)
    t = AS_data(s).t;       % 시간 벡터
    dt = mean(diff(t));     % 평균 시간 간격
    ik_scenarios(s, :) = ik';  % 전류 저장 (행 벡터로 변환)

    % W 행렬 초기화
    W = zeros(length(t), n);
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

    % y_adjusted 계산
    y_adjusted = V_sd - OCV - R0 * ik;

    % Quadprog를 위한 행렬 및 벡터 구성
    H = 2 * (W' * W + lambda * (L' * L));
    f = -2 * W' * y_adjusted;

    % 부등식 제약조건: gamma ≥ 0
    A_ineq = -eye(n);
    b_ineq = zeros(n, 1);

    % Quadprog 옵션 설정
    options = optimoptions('quadprog', 'Display', 'off');

    % Quadprog를 사용하여 최적화 문제 해결 (gamma_original)
    gamma_quadprog = quadprog(H, f, A_ineq, b_ineq, [], [], [], [], [], options);

    % 결과 저장
    gamma_quadprog_all(s, :) = gamma_quadprog';
    V_est = OCV + R0 * ik + W * gamma_quadprog;
    V_est_all(s, :) = V_est';
    V_sd_all(s, :) = V_sd';

    %% Bootstrap for Uncertainty Estimation

    % 부트스트랩을 위한 변수 초기화
    gamma_resample_all = zeros(num_resamples, n);  % 각 반복의 gamma를 저장할 배열

    for b = 1:num_resamples
        % 복원 추출로 데이터 인덱스 샘플링
        resample_idx = randsample(length(t), length(t), true);

        % 샘플링된 인덱스로부터 데이터 추출
        t_resampled = t(resample_idx);
        ik_resampled = ik(resample_idx);
        V_sd_resampled = V_sd(resample_idx);

        % 중복된 시간 제거 및 해당 데이터 정리
        [t_resampled_unique, unique_idx] = unique(t_resampled);
        ik_resampled_unique = ik_resampled(unique_idx);
        V_sd_resampled_unique = V_sd_resampled(unique_idx);

        % 시간 순서대로 정렬
        [t_resampled_sorted, sort_idx] = sort(t_resampled_unique);
        ik_resampled_sorted = ik_resampled_unique(sort_idx);
        V_sd_resampled_sorted = V_sd_resampled_unique(sort_idx);

        % 새로운 시간 간격 dt_resampled 계산
        dt_resampled = [t_resampled_sorted(1); diff(t_resampled_sorted)];

        % W_resampled 행렬 초기화
        W_resampled = zeros(length(t_resampled_sorted), n);
        for i = 1:n
            for k_idx = 1:length(t_resampled_sorted)
                if k_idx == 1
                    W_resampled(k_idx, i) = ik_resampled_sorted(k_idx) * ...
                        (1 - exp(-dt_resampled(k_idx) / tau_discrete(i))) * delta_theta;
                else
                    W_resampled(k_idx, i) = W_resampled(k_idx-1, i) * exp(-dt_resampled(k_idx) / tau_discrete(i)) + ...
                        ik_resampled_sorted(k_idx) * (1 - exp(-dt_resampled(k_idx) / tau_discrete(i))) * delta_theta;
                end
            end
        end

        % y_adjusted_resampled 계산
        y_adjusted_resampled = V_sd_resampled_sorted - OCV - R0 * ik_resampled_sorted;

        % Quadprog를 위한 행렬 및 벡터 구성
        H_resampled = 2 * (W_resampled' * W_resampled + lambda * (L' * L));
        f_resampled = -2 * W_resampled' * y_adjusted_resampled;

        % Quadprog를 사용하여 최적화 문제 해결
        gamma_resample = quadprog(H_resampled, f_resampled, A_ineq, b_ineq, [], [], [], [], [], options);

        % 결과 저장
        gamma_resample_all(b, :) = gamma_resample';
    end

    % gamma_resample의 백분위수 계산 (5%와 95%)
    gamma_resample_percentiles = prctile(gamma_resample_all - gamma_quadprog', [5 95]);

    % 불확실성 범위 계산
    gamma_lower = gamma_quadprog' + gamma_resample_percentiles(1, :);
    gamma_upper = gamma_quadprog' + gamma_resample_percentiles(2, :);

    % 각 시나리오에 대한 결과 저장
    gamma_lower_all(s, :) = gamma_lower;
    gamma_upper_all(s, :) = gamma_upper;
    gamma_resample_all_scenarios(s, :, :) = gamma_resample_all;
end

%% 결과 플롯

% DRT 비교 플롯 (모든 시나리오)
figure('Name', [AS_name, ': DRT Comparison with Uncertainty'], 'NumberTitle', 'off');
hold on;
for s = 1:num_scenarios
    % 불확실성을 errorbar로 표시
    errorbar(theta_discrete, gamma_quadprog_all(s, :), ...
        gamma_quadprog_all(s, :) - gamma_lower_all(s, :), ...
        gamma_upper_all(s, :) - gamma_quadprog_all(s, :), ...
        '--', 'LineWidth', 1.5, 'Color', c_mat(s, :), 'DisplayName', ['Scenario ', num2str(s)]);
end
plot(theta_discrete, gamma_discrete_true, 'k-', 'LineWidth', 2, 'DisplayName', 'True \gamma');
hold off;
xlabel('\theta = ln(\tau [s])', 'FontSize', labelFontSize);
ylabel('\gamma', 'FontSize', labelFontSize);
title([AS_name, ': Estimated \gamma with Uncertainty (\lambda = ', num2str(lambda), ')'], 'FontSize', titleFontSize);
set(gca, 'FontSize', axisFontSize);
legend('Location', 'Best', 'FontSize', legendFontSize);

% 선택된 시나리오에 대한 추가 그래프
selected_scenarios = [6, 7, 8, 9];

% DRT 비교 플롯 (선택된 시나리오)
figure('Name', [AS_name, ': Selected Scenarios DRT Comparison with Uncertainty'], 'NumberTitle', 'off');
hold on;
for idx_s = 1:length(selected_scenarios)
    s = selected_scenarios(idx_s);
    % 불확실성을 errorbar로 표시
    errorbar(theta_discrete, gamma_quadprog_all(s, :), ...
        gamma_quadprog_all(s, :) - gamma_lower_all(s, :), ...
        gamma_upper_all(s, :) - gamma_quadprog_all(s, :), ...
        '--', 'LineWidth', 1.5, 'Color', c_mat(s, :), 'DisplayName', ['Scenario ', num2str(s)]);
end
plot(theta_discrete, gamma_discrete_true, 'k-', 'LineWidth', 2, 'DisplayName', 'True \gamma');
hold off;
xlabel('\theta = ln(\tau [s])', 'FontSize', labelFontSize);
ylabel('\gamma', 'FontSize', labelFontSize);
title([AS_name, ': Estimated \gamma with Uncertainty for Selected Scenarios'], 'FontSize', titleFontSize);
set(gca, 'FontSize', axisFontSize);
legend('Location', 'Best', 'FontSize', legendFontSize);

% 각 시나리오별 subplot 생성
figure('Name', [AS_name, ': Individual Scenario DRTs'], 'NumberTitle', 'off');
num_cols = 5;  % subplot 열 개수
num_rows = ceil(num_scenarios / num_cols);  % subplot 행 개수

for s = 1:num_scenarios
    subplot(num_rows, num_cols, s);
    % 불확실성을 errorbar로 표시
    errorbar(theta_discrete, gamma_quadprog_all(s, :), ...
        gamma_quadprog_all(s, :) - gamma_lower_all(s, :), ...
        gamma_upper_all(s, :) - gamma_quadprog_all(s, :), ...
        'LineWidth', 1.0, 'Color', c_mat(s, :));
    hold on;
    plot(theta_discrete, gamma_discrete_true, 'k-', 'LineWidth', 1.5);
    hold off;
    xlabel('\theta', 'FontSize', labelFontSize);
    ylabel('\gamma', 'FontSize', labelFontSize);
    title(['Scenario ', num2str(s)], 'FontSize', titleFontSize);
    set(gca, 'FontSize', axisFontSize);
    grid on;
end