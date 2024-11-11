clc; clear; close all;

%% 1. 데이터 로드
% 실제 데이터 파일 위치로 경로를 교체하세요
load('G:\공유 드라이브\BSL_Data3\HPPC\03-11-17_08.47 25degC_5Pulse_HPPC_Pan18650PF.mat');

%% 2. 데이터 전처리

% 데이터 구조 초기화
data1.I = meas.Current;
data1.V = meas.Voltage;
data1.t = meas.Time;

% 데이터 유형 분류: 'C'는 충전, 'D'는 방전, 'R'은 휴지
data1.type = char(zeros([length(data1.t), 1]));
data1.type(data1.I > 0) = 'C';
data1.type(data1.I == 0) = 'R';
data1.type(data1.I < 0) = 'D';

% 스텝 식별
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

% 각 스텝에 대한 구조 배열 초기화
data_line = struct('V', [], 'I', [], 't', [], 'indx', [], 'type', '', ...
    'steptime', [], 'T', [], 'SOC', [], 'avgI', [], 'Crate', [], 'deltaV', [], 'R', [], 'R0', [], 't_rel', []);
data = repmat(data_line, num_step, 1);

for i_step = 1:num_step
    range = find(data1.step == vec_step(i_step));
    data(i_step).V = data1.V(range);
    data(i_step).I = data1.I(range);
    data(i_step).t = data1.t(range);
    data(i_step).indx = range;
    data(i_step).type = data1.type(range(1));
    data(i_step).steptime = data1.t(range);
    data(i_step).T = zeros(size(range)); % 온도 데이터가 없는 경우
    
    % 시간 초기화 (상대 시간)
    data(i_step).t_rel = data(i_step).t - data(i_step).t(1);
end

%% 3. 충전 및 방전 스텝 식별

% 스텝 인덱스를 저장할 배열 초기화
step_chg = [];
step_dis = [];

for i = 1:length(data)
    if strcmp(data(i).type, 'C')
        step_chg(end+1) = i;
    elseif strcmp(data(i).type, 'D')
        step_dis(end+1) = i;
    end
end

%% 3-1. SOC 값 정의 및 할당

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
    if current_index > length(step_dis)
        break;
    end
end

% step_dis 배열을 사용하여 데이터에 SOC 값 할당
for i = 1:length(step_dis)
    data(step_dis(i)).SOC = SOC(i);
end

% 마지막 스텝의 SOC를 0.05로 설정 (필요한 경우)
if length(step_dis) >= 130
    data(130).SOC = 0.05;
end

%% 4. 평균 전류, C-레이트, Delta V, 저항 계산

% 상수 정의
I_1C = 2.89923594059406; % 1C 전류 (방전 시 음수)

for i = 1:length(data)
    % 평균 전류
    data(i).avgI = mean(data(i).I);
    
    % C-레이트
    data(i).Crate = data(i).avgI / I_1C;
    
    % Delta Voltage
    if i == 1
        data(i).deltaV = zeros(size(data(i).V));
    else
        data(i).deltaV = data(i).V - data(i-1).V(end);
    end
    
    % 저항 계산 (R)
    if data(i).avgI == 0
        data(i).R = zeros(size(data(i).V));
    else
        data(i).R = (data(i).deltaV ./ data(i).avgI) .* ones(size(data(i).V));
    end
    
    % R0 계산 및 저장
    if length(data(i).R) >= 1
        data(i).R0 = data(i).R(1); % 첫 번째 R 값을 R0로 저장
    else
        data(i).R0 = 0;
    end
end

%% 5. HPPC 피팅 및 플롯

% 최적화된 파라미터를 저장할 테이블 초기화
optimized_params = table('Size', [0 7], ...
    'VariableTypes', {'double', 'double', 'double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'R0', 'R1', 'R2', 'C1', 'C2', 'Crate', 'SOC'});

for idx = 1:length(step_dis)
    i_step = step_dis(idx);
    
    % 데이터 추출
    I = data(i_step).I;
    V = data(i_step).V;
    t = data(i_step).t_rel; % 상대 시간
    dt = [0; diff(t)]; % 시간 간격 계산
    
    % OCV 설정: 각 스텝의 첫 번째 전압을 사용
    OCV = V(1);
    
    % R0 설정: data(i_step).R0 사용
    R0 = data(i_step).R0;
    
    % 초기 파라미터 설정 [R1, R2, C1, C2]
    initial_params = [0.01, 0.01, 1000, 3000];
    
    % 코스트 함수 정의 (RMSE 계산)
    cost_function = @(params) V - ECM_2RC(params, OCV, I, dt, R0);
    
    % 파라미터 경계 설정 (음수를 방지)
    lb = [0, 0, 0, 0];
    ub = [Inf, Inf, Inf, Inf];
    
    % 최적화 옵션 설정
    options = optimoptions('lsqnonlin', 'Display', 'off', 'MaxIterations', 1000, 'MaxFunctionEvaluations', 2000);
    
    % 피팅 수행
    [fitted_params, resnorm, residual] = lsqnonlin(cost_function, initial_params, lb, ub, options);
    
    % 모델 전압 계산
    V_model = ECM_2RC(fitted_params, OCV, I, dt, R0);
    
    % 결과 플롯
    figure;
    plot(t, V, 'b', 'LineWidth', 1.5); hold on;
    plot(t, V_model, 'r--', 'LineWidth', 1.5);
    xlabel('시간 (s)');
    ylabel('전압 (V)');
    legend('실제 전압', '모델 전압');
    title(sprintf('스텝 %d 피팅 결과', i_step));
    grid on;
    
    % 피팅된 파라미터 출력
    fprintf('스텝 %d 피팅된 파라미터:\n', i_step);
    fprintf('R0 = %.6f Ω (고정값)\n', R0);
    fprintf('R1 = %.6f Ω\n', fitted_params(1));
    fprintf('R2 = %.6f Ω\n', fitted_params(2));
    fprintf('C1 = %.6f F\n', fitted_params(3));
    fprintf('C2 = %.6f F\n', fitted_params(4));
    fprintf('Crate = %.6f\n', data(i_step).Crate);
    fprintf('SOC = %.2f\n', data(i_step).SOC);
    fprintf('\n');
    
    % 최적화된 파라미터를 테이블에 추가
    new_row = {R0, fitted_params(1), fitted_params(2), ...
               fitted_params(3), fitted_params(4), data(i_step).Crate, data(i_step).SOC};
    optimized_params = [optimized_params; new_row];
end

% 열 이름 다시 설정
optimized_params.Properties.VariableNames = {'R0', 'R1', 'R2', 'C1', 'C2', 'Crate', 'SOC'};

%% 최적화된 파라미터 테이블 출력
disp('최적화된 파라미터 테이블 (각 행은 스텝에 해당):');
disp(optimized_params);

%% 최적화된 파라미터를 CSV 파일로 저장 (선택 사항)
% writetable(optimized_params, 'optimized_params.csv');

%% Function

function V_model = ECM_2RC(params, OCV, I, dt, R0)
    % 파라미터 분리
    R1 = params(1);
    R2 = params(2);
    C1 = params(3); 
    C2 = params(4); 

    tau1 = R1 * C1;
    tau2 = R2 * C2;

    N = length(I);
    V_R1 = zeros(N,1);
    V_R2 = zeros(N,1);
    V_R0 = R0 * I;
    V_model = zeros(N,1);

    % 초기 전압 설정
    V_model(1) = OCV ; %+ V_R0(1);
    
    for k = 2:N
        V_R1(k) = V_R1(k-1) * exp(-dt(k)/tau1) + I(k) * R1 * (1 - exp(-dt(k)/tau1));
        V_R2(k) = V_R2(k-1) * exp(-dt(k)/tau2) + I(k) * R2 * (1 - exp(-dt(k)/tau2));
        V_model(k) = OCV + V_R0(k) + V_R1(k) + V_R2(k);
    end
end
