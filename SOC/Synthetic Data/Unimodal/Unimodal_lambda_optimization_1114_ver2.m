clc; clear; close all;

%% 설명

% 이 코드는 다양한 데이터셋과 타입에 대해 최적의 람다(lambda)를 찾고 결과를 저장하는 코드입니다.
% 사용자는 데이터셋과 타입을 선택할 수 있으며, 선택한 데이터에 대해 교차 검증을 통해
% 최적의 람다를 찾고 결과를 저장합니다.

%% 사용자 정의 변수 설정

% 람다 값 범위 설정 (Grid Search를 위한 람다 값)
lambda_values = logspace(-4, 9, 5);

% tau_min 설정
tau_min = 0.1;   % tau 최소값

%% 그래픽 설정

axisFontSize = 14;
titleFontSize = 12;
legendFontSize = 12;
labelFontSize = 12;

%% 데이터 로드

% 데이터 파일 경로 설정
file_path = 'G:\공유 드라이브\Battery Software Lab\Projects\DRT\SD_new\';

% 파일 목록 가져오기
mat_files = dir(fullfile(file_path, '*.mat')); % 데이터 파일들

% 데이터 로드
if isempty(mat_files)
    error('데이터 파일이 존재하지 않습니다. 경로를 확인해주세요.');
end

for file = mat_files'
    load(fullfile(file_path, file.name));
end

%% 데이터셋 선택

% 데이터셋 목록
datasets = {'AS1_1per_new', 'AS1_2per_new', 'AS2_1per_new', 'AS2_2per_new'};

% 데이터셋 선택
disp('데이터셋을 선택하세요:');
for i = 1:length(datasets)
    fprintf('%d. %s\n', i, datasets{i});
end
dataset_idx = input('데이터셋 번호를 입력하세요: ');

% 입력 유효성 검사
if isempty(dataset_idx) || ~isnumeric(dataset_idx) || dataset_idx < 1 || dataset_idx > length(datasets)
    error('유효한 데이터셋 번호를 입력해주세요.');
end

selected_dataset_name = datasets{dataset_idx};

% 데이터셋 존재 여부 확인
if ~exist(selected_dataset_name, 'var')
    error('선택한 데이터셋이 로드되지 않았습니다.');
end

selected_dataset = eval(selected_dataset_name);

%% 모든 요소에 새로운 필드 추가

% 추가할 새로운 필드 목록
new_fields = {'Lambda_vec', 'CVE', 'Lambda_hat'};

% 모든 요소에 새로운 필드 추가 (빈 값으로 초기화)
for nf = 1:length(new_fields)
    field_name = new_fields{nf};
    if ~isfield(selected_dataset, field_name)
        [selected_dataset.(field_name)] = deal([]);
    end
end

%% 타입 선택

% 선택한 데이터셋에서 사용 가능한 타입 목록 추출
if ~isfield(selected_dataset, 'type')
    error('선택한 데이터셋에 "type" 필드가 존재하지 않습니다.');
end

types = unique({selected_dataset.type});

% 타입 선택
disp('타입을 선택하세요:');
for i = 1:length(types)
    fprintf('%d. %s\n', i, types{i});
end
type_idx = input('타입 번호를 입력하세요: ');

% 입력 유효성 검사
if isempty(type_idx) || ~isnumeric(type_idx) || type_idx < 1 || type_idx > length(types)
    error('유효한 타입 번호를 입력해주세요.');
end

selected_type = types{type_idx};

% 선택한 타입의 데이터 추출
type_indices = find(strcmp({selected_dataset.type}, selected_type));

if isempty(type_indices)
    error('선택한 타입에 해당하는 데이터가 없습니다.');
end

type_data = selected_dataset(type_indices);

% 시나리오 번호(SN) 목록 추출
if ~isfield(type_data, 'SN')
    error('데이터에 "SN" 필드가 존재하지 않습니다.');
end

SN_list = [type_data.SN];

% 데이터 확인
fprintf('선택한 데이터셋: %s\n', selected_dataset_name);
fprintf('선택한 타입: %s\n', selected_type);
fprintf('시나리오 번호: ');
disp(SN_list);

%% 모든 시나리오에 대해 람다 최적화 수행

% 교차 검증을 통한 람다 최적화

% 시나리오 번호 설정
num_scenarios = length(SN_list);
scenario_numbers = SN_list;

% 학습 시나리오에서 2개를 검증 세트로 선택하는 조합 생성
validation_indices = nchoosek(scenario_numbers, 2);
num_folds = size(validation_indices, 1);

% CVE 저장 변수 초기화
cve_lambda = zeros(length(lambda_values), 1);

% 각 람다에 대해 교차 검증 수행
for l_idx = 1:length(lambda_values)
    lambda = lambda_values(l_idx);
    cve_total = 0;

    for fold = 1:num_folds
        % 검증 세트와 학습 세트 분리
        val_scenarios = validation_indices(fold, :);
        train_scenarios = setdiff(scenario_numbers, val_scenarios);

        % 학습 데이터로 gamma 추정
        gamma_estimated = estimate_gamma(lambda, train_scenarios, type_data, tau_min);

        % 검증 데이터로 에러 계산
        error_fold = calculate_error(gamma_estimated, val_scenarios, type_data, tau_min);

        % 폴드의 에러 합산
        cve_total = cve_total + error_fold;
    end

    % 평균 CVE 계산
    cve_lambda(l_idx) = cve_total / num_folds;
    fprintf('Lambda %e, CVE: %f\n', lambda, cve_lambda(l_idx));
end

% 최적 람다 찾기
[~, min_idx] = min(cve_lambda);
optimal_lambda = lambda_values(min_idx);

%% 결과 저장

% 선택한 타입의 모든 데이터에 결과 저장
for i = 1:length(type_data)
    type_data(i).Lambda_vec = lambda_values;
    type_data(i).CVE = cve_lambda;
    type_data(i).Lambda_hat = optimal_lambda;
end

% 업데이트된 타입 데이터를 선택된 데이터셋에 반영
selected_dataset(type_indices) = type_data;

% 변경된 데이터셋을 원래 변수에 저장
assignin('base', selected_dataset_name, selected_dataset);

% 결과를 .mat 파일로 저장 (필요한 경우)
% save(fullfile(file_path, [selected_dataset_name '.mat']), selected_dataset_name);

%% 그래프 그리기

% CVE vs 람다 그래프 그리기
figure('Name', 'CVE vs Lambda', 'NumberTitle', 'off');
semilogx(lambda_values, cve_lambda, 'b-', 'LineWidth', 1.5); % CVE vs Lambda plot
hold on;

% 최적 λ 포인트 표시
semilogx(optimal_lambda, cve_lambda(min_idx), 'ro', 'MarkerSize', 10, 'LineWidth', 2);

% 최적 λ 텍스트 추가
optimal_lambda_str = ['최적 \lambda = ', num2str(optimal_lambda, '%.2e')];

% 레이블 및 제목
xlabel('\lambda (정규화 파라미터)', 'FontSize', labelFontSize);
ylabel('교차 검증 오류 (CVE)', 'FontSize', labelFontSize);
title('로그 스케일 \lambda 에 따른 CVE 그래프', 'FontSize', titleFontSize);

% 그리드 및 범례
grid on;
set(gca, 'FontSize', axisFontSize);
legend({'CVE', optimal_lambda_str}, 'Location', 'best', 'FontSize', legendFontSize);

hold off;

%% 함수 정의

% Gamma 추정 함수
function [gamma_estimated] = estimate_gamma(lambda, train_scenarios, type_data, tau_min)
    % 학습 시나리오에서 W와 y_adjusted를 누적하여 구성
    W_total = [];
    y_total = [];

    for s = train_scenarios
        % 시나리오 데이터 가져오기
        idx = find([type_data.SN] == s, 1);
        if isempty(idx)
            error('시나리오 번호 %d에 해당하는 데이터가 없습니다.', s);
        end
        scenario_data = type_data(idx);

        % 필요한 데이터 추출
        if ~all(isfield(scenario_data, {'V', 'I', 'dt', 'n', 'dur'}))
            error('데이터에 필요한 필드가 없습니다.');
        end
        V_sd = scenario_data.V(:);  % 실제 전압
        ik = scenario_data.I(:);    % 전류
        dt = scenario_data.dt;      % 시간 간격
        n = scenario_data.n;        % 요소 개수
        tau_max = scenario_data.dur;  % tau 최대값

        % tau_discrete 설정
        theta_min = log(tau_min);
        theta_max = log(tau_max);
        theta_discrete = linspace(theta_min, theta_max, n)';
        tau_discrete = exp(theta_discrete);

        delta_theta = theta_discrete(2) - theta_discrete(1);

        % W 행렬 계산
        W_s = compute_W(ik, tau_discrete, delta_theta, dt);

        % y_s 계산
        OCV = 0;  % OCV는 0으로 가정
        R0 = 0.1;  % R0는 0.1로 가정
        y_s = V_sd - OCV - R0 * ik;

        % 누적
        W_total = [W_total; W_s];
        y_total = [y_total; y_s];
    end

    % Regularization Matrix L 생성
    L = diff(eye(size(W_total, 2)));

    % Quadratic programming을 위한 행렬 및 벡터 구성
    H = 2 * (W_total' * W_total + lambda * (L' * L));
    f = -2 * W_total' * y_total;

    % 부등식 제약조건 설정: gamma >= 0
    A_ineq = -eye(size(W_total, 2));
    b_ineq = zeros(size(W_total, 2), 1);

    % quadprog 옵션 설정
    options = optimoptions('quadprog', 'Display', 'off');

    % quadprog를 사용하여 gamma 추정
    gamma_estimated = quadprog(H, f, A_ineq, b_ineq, [], [], [], [], [], options);

    if isempty(gamma_estimated)
        error('Gamma 추정에 실패하였습니다.');
    end
end

% 에러 계산 함수
function error_total = calculate_error(gamma_estimated, val_scenarios, type_data, tau_min)
    error_total = 0;

    for s = val_scenarios
        % 시나리오 데이터 가져오기
        idx = find([type_data.SN] == s, 1);
        if isempty(idx)
            error('시나리오 번호 %d에 해당하는 데이터가 없습니다.', s);
        end
        scenario_data = type_data(idx);

        % 필요한 데이터 추출
        if ~all(isfield(scenario_data, {'V', 'I', 'dt', 'n', 'dur'}))
            error('데이터에 필요한 필드가 없습니다.');
        end
        V_actual = scenario_data.V(:);  % 실제 전압
        ik = scenario_data.I(:);        % 전류
        dt = scenario_data.dt;          % 시간 간격
        n = scenario_data.n;            % 요소 개수
        tau_max = scenario_data.dur;    % tau 최대값

        % tau_discrete 설정
        theta_min = log(tau_min);
        theta_max = log(tau_max);
        theta_discrete = linspace(theta_min, theta_max, n)';
        tau_discrete = exp(theta_discrete);

        delta_theta = theta_discrete(2) - theta_discrete(1);

        % 전압 예측
        V_predicted = predict_voltage(gamma_estimated, ik, tau_discrete, delta_theta, dt);

        % 에러 계산
        error_total = error_total + sum((V_predicted - V_actual).^2);
    end
end

% W 행렬 계산 함수
function W = compute_W(ik, tau_discrete, delta_theta, dt)
    ik = ik(:);  % ik를 열 벡터로 변환

    n = length(tau_discrete);
    len_t = length(ik);
    W = zeros(len_t, n);  % W 행렬 초기화

    exp_dt_tau = exp(-dt ./ tau_discrete);

    % 초기값 계산
    W(1, :) = ik(1) * (1 - exp_dt_tau)' * delta_theta;

    % W 행렬 계산
    for k_idx = 2:len_t
        W(k_idx, :) = W(k_idx - 1, :) .* exp_dt_tau' + ik(k_idx) * (1 - exp_dt_tau)' * delta_theta;
    end
end

% 전압 예측 함수
function V_predicted = predict_voltage(gamma_estimated, ik, tau_discrete, delta_theta, dt)
    ik = ik(:);  % ik를 열 벡터로 변환

    len_t = length(ik);
    n = length(gamma_estimated);
    V_predicted = zeros(len_t, 1);

    OCV = 0;  % OCV는 0으로 가정
    R0 = 0.1;  % R0는 0.1로 가정

    exp_dt_tau = exp(-dt ./ tau_discrete);

    % V_RC 초기화
    V_RC = zeros(len_t, n);

    % V_RC 및 V_predicted 계산
    for k_idx = 1:len_t
        if k_idx == 1
            V_RC(k_idx, :) = gamma_estimated' .* ik(k_idx) .* (1 - exp_dt_tau)' * delta_theta;
        else
            V_RC(k_idx, :) = V_RC(k_idx - 1, :) .* exp_dt_tau' + ...
                             gamma_estimated' .* ik(k_idx) .* (1 - exp_dt_tau)' * delta_theta;
        end
        V_predicted(k_idx) = OCV + R0 * ik(k_idx) + sum(V_RC(k_idx, :));
    end
end

