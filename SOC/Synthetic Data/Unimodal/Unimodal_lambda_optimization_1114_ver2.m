clc; clear; close all;

%% 설명

% 이 코드는 다양한 데이터셋과 타입에 대해 최적의 람다(lambda)를 찾고 결과를 저장하는 코드입니다.
% 사용자는 데이터셋과 타입을 선택할 수 있으며, 선택한 데이터에 대해 교차 검증을 통해
% 최적의 람다를 찾고 결과를 저장합니다.

%% 그래픽 설정

axisFontSize = 14;
titleFontSize = 12;
legendFontSize = 12;
labelFontSize = 12;

%% 데이터 로드

% 데이터 파일 경로 설정
file_path = 'G:\공유 드라이브\Battery Software Lab\Projects\DRT\SD_new\';

% 파일 목록 가져오기
mat_files = dir(fullfile(file_path, '*.mat')); % AS1_1per_new, AS1_2per_new, AS2_1per_new, AS2_2per_new, Unimodal_gamma, Bimodal_gamma

% 데이터 로드
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
selected_dataset_name = datasets{dataset_idx};
selected_dataset = eval(selected_dataset_name);

%% 타입 선택

% 선택한 데이터셋에서 사용 가능한 타입 목록 추출
types = unique({selected_dataset.type});

% 타입 선택
disp('타입을 선택하세요:');
for i = 1:length(types)
    fprintf('%d. %s\n', i, types{i});
end
type_idx = input('타입 번호를 입력하세요: ');
selected_type = types{type_idx};

% 선택한 타입의 데이터 추출
type_indices = find(strcmp({selected_dataset.type}, selected_type));
type_data = selected_dataset(type_indices);

% 시나리오 번호(SN) 목록 추출
SN_list = [type_data.SN];

% 데이터 확인
fprintf('선택한 데이터셋: %s\n', selected_dataset_name);
fprintf('선택한 타입: %s\n', selected_type);
fprintf('시나리오 번호: ');
disp(SN_list);

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

%% 람다 최적화 여부 확인

% 선택한 타입의 첫 번째 데이터에서 Lambda_vec 존재 여부 확인
if ~isempty(type_data(1).Lambda_vec)
    % 이미 람다 최적화가 수행된 경우
    disp('이미 람다 최적화가 수행되었습니다.');
    recompute = input('람다 최적화를 다시 수행하시겠습니까? (y/n): ', 's');
    if strcmpi(recompute, 'y')
        perform_lambda_optimization = true;
    else
        perform_lambda_optimization = false;
    end
else
    % 람다 최적화가 수행되지 않은 경우
    perform_lambda_optimization = true;
end

if perform_lambda_optimization
    %% 람다 값 설정

    lambda_values = logspace(-4, 9, 50);  % 람다 값 범위 설정

    %% 교차 검증을 통한 람다 최적화

    % 시나리오 번호 설정 (1부터 10까지)
    num_scenarios = 10;
    scenario_numbers = 1:num_scenarios;

    % 학습 시나리오 번호 리스트
    train_scenarios_full = scenario_numbers;

    % 학습 시나리오에서 2개를 검증 세트로 선택하는 조합 생성
    validation_indices = nchoosek(train_scenarios_full, 2);
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
            train_scenarios = setdiff(train_scenarios_full, val_scenarios);

            % 학습 데이터로 gamma 추정
            gamma_estimated = estimate_gamma(lambda, train_scenarios, type_data);

            % 검증 데이터로 에러 계산
            error_fold = calculate_error(gamma_estimated, val_scenarios, type_data);

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
    eval([selected_dataset_name ' = selected_dataset;']);

    % 결과를 .mat 파일로 저장 (필요한 경우)
    save(fullfile(file_path, [selected_dataset_name '.mat']), selected_dataset_name);
else
    % 이미 최적화된 람다 값 가져오기
    optimal_lambda = type_data(1).Lambda_hat;
    lambda_values = type_data(1).Lambda_vec;
    cve_lambda = type_data(1).CVE;
    [~, min_idx] = min(cve_lambda);
end

%% 그래프 그리기

% CVE vs 람다 그래프 그리기
figure;
semilogx(lambda_values, cve_lambda, 'b-', 'LineWidth', 1.5); % CVE vs Lambda plot
hold on;

% 최적 \(\lambda\) 포인트 표시
semilogx(optimal_lambda, cve_lambda(min_idx), 'ro', 'MarkerSize', 10, 'LineWidth', 2);

% 최적 \(\lambda\) 텍스트 추가
optimal_lambda_str = ['최적 \lambda = ', num2str(optimal_lambda, '%.2e')];

% 레이블 및 제목
xlabel('\lambda (정규화 파라미터)');
ylabel('교차 검증 오류 (CVE)');
title('로그 스케일 \lambda 에 따른 CVE 그래프');

% 그리드 및 범례
grid on;
set(gca, 'YScale', 'log');  % Y축 로그 스케일 설정
legend({'CVE', optimal_lambda_str}, 'Location', 'best');

hold off;

%% 함수 정의

% Gamma 추정 함수
function [gamma_estimated] = estimate_gamma(lambda, train_scenarios, type_data)
    % 학습 시나리오에서 W와 y_adjusted를 누적하여 구성
    W_total = [];
    y_total = [];

    for s = train_scenarios
        % 시나리오 데이터 가져오기
        scenario_data = type_data([type_data.SN] == s);

        % 필요한 데이터 추출
        V_sd = scenario_data.V(:);  % 실제 전압, 열 벡터로 변환
        ik = scenario_data.I(:);  % 전류, 열 벡터로 변환
        t = scenario_data.t;  % 시간
        dt = scenario_data.dt;  % 시간 간격
        n = scenario_data.n;  % 요소 개수

        % tau_discrete 설정 (로그 스케일로 균등 분할)
        theta_min = -10;
        theta_max = 10;
        theta_discrete = linspace(theta_min, theta_max, n);
        tau_discrete = exp(theta_discrete);

        delta_theta = theta_discrete(2) - theta_discrete(1);

        % First-order difference matrix L 생성
        L = zeros(n-1, n);
        for i = 1:n-1
            L(i, i) = -1;
            L(i, i+1) = 1;
        end

        % W 행렬 계산
        W_s = compute_W(ik, tau_discrete, delta_theta, dt);

        % y_s 계산 (V_sd에서 OCV와 R0*I를 뺀 값)
        OCV = 0;  % OCV는 0으로 가정
        R0 = 0.1;  % R0는 0.1로 가정
        y_s = V_sd - OCV - R0 * ik;

        % 누적
        W_total = [W_total; W_s];
        y_total = [y_total; y_s];  % y_s를 열 벡터로 그대로 추가

    end

    % Quadratic programming을 위한 행렬 및 벡터 구성
    H = W_total' * W_total + lambda * (L' * L);
    f = -W_total' * y_total;

    % 부등식 제약조건 설정: gamma >= 0  ==>  -I * gamma <= 0
    A_ineq = -eye(length(tau_discrete));  % A = -I
    b_ineq = zeros(length(tau_discrete), 1);  % b = 0

    % quadprog 옵션 설정
    options = optimoptions('quadprog', 'Display', 'off');

    % quadprog를 사용하여 gamma 추정
    gamma_estimated = quadprog(H, f, A_ineq, b_ineq, [], [], [], [], [], options);
end

% 에러 계산 함수
function error_total = calculate_error(gamma_estimated, val_scenarios, type_data)
    error_total = 0;

    for s = val_scenarios
        % 시나리오 데이터 가져오기
        scenario_data = type_data([type_data.SN] == s);

        % 필요한 데이터 추출
        V_actual = scenario_data.V(:);  % 실제 전압, 열 벡터로 변환
        ik = scenario_data.I(:);  % 전류, 열 벡터로 변환
        t = scenario_data.t;  % 시간
        dt = scenario_data.dt;  % 시간 간격
        n = scenario_data.n;  % 요소 개수

        % tau_discrete 설정 (로그 스케일로 균등 분할)
        theta_min = -10;
        theta_max = 10;
        theta_discrete = linspace(theta_min, theta_max, n);
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
    W = zeros(len_t, n);  % Initialize W matrix

    for k_idx = 1:len_t
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
end

% 전압 예측 함수
function V_predicted = predict_voltage(gamma_estimated, ik, tau_discrete, delta_theta, dt)
    ik = ik(:);  % ik를 열 벡터로 변환

    len_t = length(ik);
    n = length(gamma_estimated);
    V_predicted = zeros(len_t, 1);
    V_RC = zeros(n, len_t);

    OCV = 0;  % OCV는 0으로 가정
    R0 = 0.1;  % R0는 0.1로 가정

    for k_idx = 1:len_t
        if k_idx == 1
            for i = 1:n
                V_RC(i, k_idx) = gamma_estimated(i) * ik(k_idx) * (1 - exp(-dt / tau_discrete(i))) * delta_theta;
            end
        else
            for i = 1:n
                V_RC(i, k_idx) = V_RC(i, k_idx-1) * exp(-dt / tau_discrete(i)) + ...
                                 gamma_estimated(i) * ik(k_idx) * (1 - exp(-dt / tau_discrete(i))) * delta_theta;
            end
        end
        V_predicted(k_idx) = OCV + R0 * ik(k_idx) + sum(V_RC(:, k_idx));
    end
end

