clc; clear; close all;

%% 설정
lambda_values = logspace(-4, 9, 5);
tau_min = 0.1;

%% 데이터 로드
file_path = 'G:\공유 드라이브\Battery Software Lab\Projects\DRT\SD_new\';
mat_files = dir(fullfile(file_path, '*.mat'));
if isempty(mat_files)
    error('데이터 파일이 존재하지 않습니다. 경로를 확인해주세요.');
end
for file = mat_files'
    load(fullfile(file_path, file.name));
end

%% 데이터셋 선택
datasets = {'AS1_1per_new', 'AS1_2per_new', 'AS2_1per_new', 'AS2_2per_new'};
disp('데이터셋을 선택하세요:');
for i = 1:length(datasets)
    fprintf('%d. %s\n', i, datasets{i});
end
dataset_idx = input('데이터셋 번호를 입력하세요: ');
if isempty(dataset_idx) || dataset_idx < 1 || dataset_idx > length(datasets)
    error('유효한 데이터셋 번호를 입력해주세요.');
end
selected_dataset_name = datasets{dataset_idx};
if ~exist(selected_dataset_name, 'var')
    error('선택한 데이터셋이 로드되지 않았습니다.');
end
selected_dataset = eval(selected_dataset_name);

%% 타입 선택 및 데이터 준비
types = unique({selected_dataset.type});
disp('타입을 선택하세요:');
for i = 1:length(types)
    fprintf('%d. %s\n', i, types{i});
end
type_idx = input('타입 번호를 입력하세요: ');
if isempty(type_idx) || type_idx < 1 || type_idx > length(types)
    error('유효한 타입 번호를 입력해주세요.');
end
selected_type = types{type_idx};
type_indices = strcmp({selected_dataset.type}, selected_type);
type_data = selected_dataset(type_indices);
if isempty(type_data)
    error('선택한 타입에 해당하는 데이터가 없습니다.');
end
SN_list = [type_data.SN];

%% 새로운 필드 추가
new_fields = {'Lambda_vec', 'CVE', 'Lambda_hat'};
for nf = 1:length(new_fields)
    field_name = new_fields{nf};
    if ~isfield(selected_dataset, field_name)
        [selected_dataset.(field_name)] = deal([]);
    end
end

%% 람다 최적화 및 교차 검증
scenario_numbers = SN_list;
validation_indices = nchoosek(scenario_numbers, 2);
num_folds = size(validation_indices, 1);
cve_lambda = zeros(length(lambda_values), 1);

for l_idx = 1:length(lambda_values)
    lambda = lambda_values(l_idx);
    cve_total = 0;
    for fold = 1:num_folds
        val_scenarios = validation_indices(fold, :);
        train_scenarios = setdiff(scenario_numbers, val_scenarios);
        gamma_estimated = estimate_gamma(lambda, train_scenarios, type_data, tau_min);
        cve_total = cve_total + calculate_error(gamma_estimated, val_scenarios, type_data, tau_min);
    end
    cve_lambda(l_idx) = cve_total / num_folds;
    fprintf('Lambda %e, CVE: %f\n', lambda, cve_lambda(l_idx));
end
[~, min_idx] = min(cve_lambda);
optimal_lambda = lambda_values(min_idx);

%% 결과 저장
for i = 1:length(type_data)
    type_data(i).Lambda_vec = lambda_values;
    type_data(i).CVE = cve_lambda;
    type_data(i).Lambda_hat = optimal_lambda;
end
selected_dataset(type_indices) = type_data;
assignin('base', selected_dataset_name, selected_dataset);

%% 그래프 그리기
figure;
semilogx(lambda_values, cve_lambda, 'b-', 'LineWidth', 1.5); hold on;
semilogx(optimal_lambda, cve_lambda(min_idx), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('\lambda (정규화 파라미터)');
ylabel('교차 검증 오류 (CVE)');
title('로그 스케일 \lambda 에 따른 CVE 그래프');
grid on;
legend({'CVE', ['최적 \lambda = ', num2str(optimal_lambda, '%.2e')]}, 'Location', 'best');
hold off;

%% 함수 정의
function gamma_estimated = estimate_gamma(lambda, train_scenarios, type_data, tau_min)
    W_total = []; y_total = [];
    for s = train_scenarios
        idx = find([type_data.SN] == s, 1);
        scenario_data = type_data(idx);
        V_sd = scenario_data.V(:);
        ik = scenario_data.I(:);
        dt = scenario_data.dt;
        n = scenario_data.n;
        tau_max = scenario_data.dur;
        theta_discrete = linspace(log(tau_min), log(tau_max), n)';
        tau_discrete = exp(theta_discrete);
        delta_theta = theta_discrete(2) - theta_discrete(1);
        W_s = compute_W(ik, tau_discrete, delta_theta, dt);
        y_s = V_sd - 0 - 0.1 * ik;
        W_total = [W_total; W_s];
        y_total = [y_total; y_s];
    end
    L = diff(eye(size(W_total, 2)));
    H = 2 * (W_total' * W_total + lambda * (L' * L));
    f = -2 * W_total' * y_total;
    options = optimoptions('quadprog', 'Display', 'off');
    gamma_estimated = quadprog(H, f, -eye(size(W_total, 2)), zeros(size(W_total, 2), 1), [], [], [], [], [], options);
end

function error_total = calculate_error(gamma_estimated, val_scenarios, type_data, tau_min)
    error_total = 0;
    for s = val_scenarios
        idx = find([type_data.SN] == s, 1);
        scenario_data = type_data(idx);
        V_actual = scenario_data.V(:);
        ik = scenario_data.I(:);
        dt = scenario_data.dt;
        n = scenario_data.n;
        tau_max = scenario_data.dur;
        theta_discrete = linspace(log(tau_min), log(tau_max), n)';
        tau_discrete = exp(theta_discrete);
        delta_theta = theta_discrete(2) - theta_discrete(1);
        V_predicted = predict_voltage(gamma_estimated, ik, tau_discrete, delta_theta, dt);
        error_total = error_total + sum((V_predicted - V_actual).^2);
    end
end

function W = compute_W(ik, tau_discrete, delta_theta, dt)
    ik = ik(:);
    n = length(tau_discrete);
    len_t = length(ik);
    W = zeros(len_t, n);
    exp_dt_tau = exp(-dt ./ tau_discrete)';
    W(1, :) = ik(1) * (1 - exp_dt_tau) * delta_theta;
    for k = 2:len_t
        W(k, :) = W(k - 1, :) .* exp_dt_tau + ik(k) * (1 - exp_dt_tau) * delta_theta;
    end
end

function V_predicted = predict_voltage(gamma_estimated, ik, tau_discrete, delta_theta, dt)
    ik = ik(:);
    len_t = length(ik);
    n = length(gamma_estimated);
    V_predicted = zeros(len_t, 1);
    exp_dt_tau = exp(-dt ./ tau_discrete)';
    V_RC = zeros(len_t, n);
    for k = 1:len_t
        if k == 1
            V_RC(k, :) = gamma_estimated' .* ik(k) .* (1 - exp_dt_tau) * delta_theta;
        else
            V_RC(k, :) = V_RC(k - 1, :) .* exp_dt_tau + gamma_estimated' .* ik(k) .* (1 - exp_dt_tau) * delta_theta;
        end
        V_predicted(k) = 0 + 0.1 * ik(k) + sum(V_RC(k, :));
    end
end

