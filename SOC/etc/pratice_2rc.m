% 스크립트 시작
clc;clear;close all;
% 데이터 로드
% 이전과 동일하게 데이터를 로드합니다.
data = load('G:\공유 드라이브\BSL_Data3\HPPC\03-11-17_08.47 25degC_5Pulse_HPPC_Pan18650PF.mat');

% 시간, 전압, 전류 데이터 추출
time = data.meas.Time;
voltage = data.meas.Voltage;
current = data.meas.Current;

data1.I = current;
data1.V = voltage;
data1.t = time;

% 전류 상태 구분
data1.type = char(zeros([length(data1.t), 1]));
data1.type(data1.I > 0) = 'C';
data1.type(data1.I == 0) = 'R';
data1.type(data1.I < 0) = 'D';

% 스텝 구분
data1_length = length(data1.t);
data1.step = zeros(data1_length, 1);
m = 1;
data1.step(1) = m;
for j = 2:data1_length
    if data1.type(j) ~= data1.type(j-1)
        m = m + 1;
    end
    data1.step(j) = m;
end

vec_step = unique(data1.step);
num_step = length(vec_step);

% 각 스텝별로 데이터를 구조체에 저장
data_line = struct('V', zeros(1, 1), 'I', zeros(1, 1), 't', zeros(1, 1), 'indx', zeros(1, 1), 'type', char('R'), ...
    'steptime', zeros(1, 1), 'T', zeros(1, 1), 'SOC', zeros(1, 1));
data = repmat(data_line, num_step, 1);

for i_step = 1:num_step
    range = find(data1.step == vec_step(i_step));
    data(i_step).V = data1.V(range);
    data(i_step).I = data1.I(range);
    data(i_step).t = data1.t(range);
    data(i_step).indx = range;
    data(i_step).type = data1.type(range(1));
    data(i_step).steptime = data1.t(range);
    data(i_step).T = zeros(size(range)); % 온도 데이터가 없으므로 0으로 설정
end

% 초기 SOC 설정 (1로 가정)
initial_SOC = 1;
capacity_Ah = 2.9; % 배터리 용량 (Ah)

% Discharge 스텝 구하기
step_chg = [];
step_dis = [];

for i = 1:length(data)
    if strcmp(data(i).type, 'C')
        step_chg(end+1) = i;
    elseif strcmp(data(i).type, 'D')
        step_dis(end+1) = i;
    end
end

%% R0 추출

% 평균 전류 계산
for i = 1:length(data)
    data(i).avgI = mean(data(i).I);
end

% 전압 변화량 계산
for i = 1:length(data)
    if i == 1
        data(i).deltaV = zeros(size(data(i).V));
    else
        data(i).deltaV = data(i).V - data(i-1).V(end);
    end
end

% 저항 값 계산
for i = 1:length(data)
    if data(i).avgI == 0
        data(i).R = zeros(size(data(i).V));
    else
        data(i).R = (data(i).deltaV / data(i).avgI) .* ones(size(data(i).V));
    end
end

% 시간 초기화
for i = 1:length(data)
    initialTime = data(i).t(1);
    data(i).t = data(i).t - initialTime;
end

% R0 추출 (첫 번째 저항 값 사용)
for i = 1:length(step_dis)
    if length(data(step_dis(i)).t) >= 5
        data(step_dis(i)).R001s = data(step_dis(i)).R(1);
        data(step_dis(i)).R0 = data(step_dis(i)).R001s;
    else
        data(step_dis(i)).R001s = NaN;
        data(step_dis(i)).R0 = NaN;
    end
end

%% SOC 값 설정

% SOC 값을 정의된 패턴에 따라 생성
soc_values = [1, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05];
steps_per_level = 5;

% SOC 배열 초기화
SOC = zeros(length(step_dis), 1);
current_index = 1;

for i = 1:length(soc_values)
    end_index = min(current_index + steps_per_level - 1, length(step_dis));
    SOC(current_index:end_index) = soc_values(i);
    current_index = end_index + 1;
end

% 데이터에 SOC 값 할당
for i = 1:length(step_dis)
    data(step_dis(i)).SOC = SOC(i);
end

% 마지막 데이터의 SOC 값 설정
data(130).SOC = 0.05;

%% 2-RC 모델을 이용한 파라미터 추출

% 구조체 생성
optimized_params_struct = struct('R0', [], 'R1', [], 'R2', [], 'C1', [], 'C2', [], 'SOC', [], 'avgI', [], 'm', [], 'cost', []);

% 초기 추정값 설정
initial_guess = [0.005, 0.005, 1500, 3000]; % [R1, R2, C1, C2]

% 하한값 설정
lb = [0, 0, 0, 0];

for i = 1:length(step_dis)
    deltaV_exp = data(step_dis(i)).deltaV;
    time_exp = data(step_dis(i)).t;
    avgI = data(step_dis(i)).avgI;
    R0 = data(step_dis(i)).R0; % R0 값 가져오기

    % 스텝의 시간 길이 확인
    step_duration = time_exp(end) - time_exp(1);

    if step_duration >= 0
        best_cost = Inf;
        best_params = [];
        best_m = [];
        tau_candidates = [];

        % tau 계산을 위해 deltaV_exp의 63.2% 지점 찾기
        deltaV_total = deltaV_exp(end) - deltaV_exp(1);
        deltaV_63 = deltaV_exp(1) + 0.632 * deltaV_total;

        % deltaV_exp에서 deltaV_63에 가장 가까운 지점 찾기
        [~, idx_tau] = min(abs(deltaV_exp - deltaV_63));

        % tau 계산
        tau = time_exp(idx_tau);

        if isempty(tau) || tau <= 0
            tau = 1; % tau가 계산되지 않을 경우 기본값 설정
        end

        % m 후보군 설정
        %m_candidates = [2 / tau, 1.05, 1.2, 1.4, 0.5/ tau, 1/tau , 4/tau , 10/tau , 0.1/tau, 0];
        m_candidates = [0.1];
        

        % 중복 제거 및 정렬
        m_candidates = unique(m_candidates);
        m_candidates = sort(m_candidates);

        % fmincon을 사용하여 최적화 수행
        options = optimoptions('fmincon', 'Display', 'off', 'MaxIterations', 100);

        % createOptimProblem에서 x0는 단일 벡터여야 합니다.
        problem = createOptimProblem('fmincon', 'objective', @(params) cost_function(params, time_exp, deltaV_exp, avgI, R0, m_candidates(1)), ...
            'x0', initial_guess, 'lb', lb, 'ub', [], 'options', options);

        % MultiStart 객체 생성
        ms = MultiStart('Display', 'off', 'UseParallel', false);

        % 시작점 생성 (RandomStartPointSet 사용)
        startpoints = RandomStartPointSet('NumStartPoints', 10);

        % m 후보군에 대해 반복
        for m_idx = 1:length(m_candidates)
            m = m_candidates(m_idx);

            % 최적화 문제에 m 값을 전달하기 위해 익명 함수 사용
            problem.objective = @(params) cost_function(params, time_exp, deltaV_exp, avgI, R0, m);

            % 최적화 실행
            [opt_params, fval, exitflag, output, solutions] = run(ms, problem, startpoints);

            % 최적화가 실패한 경우 패스
            if exitflag <= 0
                continue;
            end

            % 음수 파라미터가 나오면 패스
            if any(opt_params < 0)
                continue;
            end

            % 가장 작은 잔차를 가진 파라미터 선택
            if fval < best_cost
                best_cost = fval;
                best_params = opt_params;
                best_m = m;
            end

            % m = 2 / tau에서 잔차가 충분히 작으면 반복 중단
            if m == 2 / tau && fval < 1e-4 % 임계값은 상황에 맞게 조절
                break;
            end
        end

        % 유효한 파라미터를 찾은 경우에만 저장
        if ~isempty(best_params)
            % 최적의 파라미터 저장
            optimized_params_struct(i).R0 = R0; % 고정된 R0 값 사용
            optimized_params_struct(i).R1 = best_params(1);
            optimized_params_struct(i).R2 = best_params(2);
            optimized_params_struct(i).C1 = best_params(3);
            optimized_params_struct(i).C2 = best_params(4);
            optimized_params_struct(i).SOC = mean(data(step_dis(i)).SOC);
            optimized_params_struct(i).Crate = avgI / data(step_dis(2)).avgI;
            optimized_params_struct(i).m = best_m;
            optimized_params_struct(i).cost = best_cost;

            % 최적의 m과 파라미터로 모델 계산
            voltage_model = model_func(time_exp, R0, best_params(1), best_params(2), best_params(3), best_params(4), avgI);

            % 그래프 그리기
            figure('Position', [0 0 800 600]);

            lw = 2;
            msz = 10;

            color1 = [0, 0.4470, 0.7410];
            color2 = [0.8500, 0.3250, 0.0980];
            subplot(3, 1, [1 2]);

            plot(time_exp, deltaV_exp, 'b-', 'LineWidth', lw, 'Color', color1);
            hold on;

            plot(time_exp, voltage_model, 'r--', 'LineWidth', lw, 'Color', color2);

            soc_text = sprintf('SOC: %.2f%%', optimized_params_struct(i).SOC * 100);
            crate_text = sprintf('C-rate: %.2f', optimized_params_struct(i).Crate);
            m_text = sprintf('m: %.2f', best_m);
            text(time_exp(1) + (time_exp(end) - time_exp(1)) * 0.05, max(deltaV_exp) * 0.9, {soc_text, crate_text, m_text}, 'FontSize', 12, 'Color', 'black', 'FontWeight', 'bold');

            legend('실험 데이터', '모델 결과');
            xlabel('시간 (sec)');
            ylabel('전압 (V)');
            title('실험 데이터와 모델 결과');

            set(gca, 'FontSize', 16, 'LineWidth', 2);

            subplot(3, 1, 3);

            weight_function = exp(-best_m * time_exp);
            plot(time_exp, weight_function, 'g-', 'LineWidth', lw, 'Color', [0, 1, 0]);

            legend('Weight Function');
            xlabel('시간 (sec)');
            ylabel('가중치');
            title('가중치 함수');

            set(gca, 'FontSize', 16, 'LineWidth', 2);

            set(gca, 'Position', [0.13, 0.1, 0.775, 0.25]);
            set(gcf, 'Position', [0, 0, 800, 800]);
        else
            % 유효한 파라미터를 찾지 못한 경우 해당 스텝을 건너뜁니다.
            fprintf('스텝 %d에서 유효한 파라미터를 찾지 못했습니다.\n', i);
            continue;
        end
    end
end

%% R1, R2 값의 이상치 제거

% 유효한 파라미터만 가진 구조체 배열로 갱신
optimized_params_struct = optimized_params_struct(~cellfun(@isempty, {optimized_params_struct.R1}));

% R1 값의 평균과 표준 편차 계산
R1_values = [optimized_params_struct.R1];
R1_mean = mean(R1_values);
R1_std = std(R1_values);

% R2 값의 평균과 표준 편차 계산
R2_values = [optimized_params_struct.R2];
R2_mean = mean(R2_values);
R2_std = std(R2_values);

% 이상치를 감지하기 위한 임계값 설정 (예: 평균 ± 3 표준편차)
threshold = 3;
outliers_R1 = abs(R1_values - R1_mean) > threshold * R1_std;
outliers_R2 = abs(R2_values - R2_mean) > threshold * R2_std;
outliers = outliers_R1 | outliers_R2;

% 이상치가 아닌 데이터만 선택
optimized_params_filtered = optimized_params_struct(~outliers);

% 이상치가 제거된 데이터를 저장
save('optimized_params_filtered_2RC.mat', 'optimized_params_filtered');
save('optimized_params_struct_2RC.mat', 'optimized_params_struct');

% R0, R1, R2, C1, C2 평균값 계산 (이상치 제거 후)
R0_mean_filtered = mean([optimized_params_filtered.R0]);
R1_mean_filtered = mean([optimized_params_filtered.R1]);
R2_mean_filtered = mean([optimized_params_filtered.R2]);
C1_mean_filtered = mean([optimized_params_filtered.C1]);
C2_mean_filtered = mean([optimized_params_filtered.C2]);

% 결과 출력
fprintf('R0의 평균 값 : %.6f\n', R0_mean_filtered);
fprintf('R1의 평균 값 : %.6f\n', R1_mean_filtered);
fprintf('R2의 평균 값 : %.6f\n', R2_mean_filtered);
fprintf('C1의 평균 값 : %.6f\n', C1_mean_filtered);
fprintf('C2의 평균 값 : %.6f\n', C2_mean_filtered);

%% 함수

function cost = cost_function(params, time, deltaV, I, R0, m)
    R1 = params(1);
    R2 = params(2);
    C1 = params(3);
    C2 = params(4);

    % 모델 함수를 사용하여 예측 전압 계산
    voltage_model = model_func(time, R0, R1, R2, C1, C2, I);

    % 오차 계산
    error = deltaV - voltage_model;

    % 시간에 따라 가중치 함수 적용
    time_weights = exp(-m * time);

    % 가중 평균 제곱근 오차 계산
    weighted_error = error .* time_weights;
    cost = sqrt(mean(weighted_error.^2));
end

function voltage = model_func(time, R0, R1, R2, C1, C2, I)
    voltage = I * (R0 ...
        + R1 * (1 - exp(-time / (R1 * C1))) ...
        + R2 * (1 - exp(-time / (R2 * C2))));
end

% 스크립트 끝
