clc; clear; close all;

% 데이터 로드
%data = load('C:\Users\USER\Desktop\Panasonic 18650PF Data\Panasonic 18650PF Data\25degC\5 pulse disch\03-11-17_08.47 25degC_5Pulse_HPPC_Pan18650PF.mat');
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

% step 구분
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

% Discharge step 구하기
step_chg = [];
step_dis = [];

for i = 1:length(data)
    % type 필드가 C인지 확인
    if strcmp(data(i).type, 'C')
        % C가 맞으면 idx 1 추가
        step_chg(end+1) = i;
    % type 필드가 D인지 확인
    elseif strcmp(data(i).type, 'D')
        % 맞으면 idx 1 추가
        step_dis(end+1) = i;
    end
end

%% R0, R1, C 추출 

% 평균 전류 구하기
for i = 1:length(data)
    data(i).avgI = mean(data(i).I);
end

% V 변화량 구하기
for i = 1 : length(data)
    if i == 1
       data(i).deltaV = zeros(size(data(i).V));
    else
       data(i).deltaV = data(i).V - data(i-1).V(end);
    end
end

% Resistance 구하기 
for i = 1 : length(data)
    if data(i).avgI == 0
        data(i).R = zeros(size(data(i).V));
    else 
        data(i).R = (data(i).deltaV / data(i).avgI) .* ones(size(data(i).V));
    end
end

% 시간 초기화 및 초기 시간 저장
initialTimes = zeros(length(data), 1); % 초기 시간을 저장할 배열 초기화
for i = 1 : length(data)
    initialTime = data(i).t(1); % 각 스텝의 초기 시간 저장
    initialTimes(i) = initialTime; % 초기 시간을 배열에 저장
    data(i).t = data(i).t - initialTime; % 시간 초기화
end

for i = 1:length(step_dis)
    % 시간의 길이가 5초 이상인 스텝에 대해서만 r1s 값을 계산
    if length(data(step_dis(i)).t) >= 5
       data(step_dis(i)).R001s = data(step_dis(i)).R(1);
       if length(data(step_dis(i)).R) >= 11
           data(step_dis(i)).R1s = data(step_dis(i)).R(11);
       else
           data(step_dis(i)).R1s = data(step_dis(i)).R(end);
       end
       data(step_dis(i)).R0 = data(step_dis(i)).R001s;
       data(step_dis(i)).R1 = data(step_dis(i)).R1s - data(step_dis(i)).R001s;
    else
       data(step_dis(i)).R001s = NaN;
       data(step_dis(i)).R1s = NaN;
       data(step_dis(i)).R0 = NaN;
       data(step_dis(i)).R1 = NaN;
    end
end

%% 63.2% 값을 이용한 tau 및 C 계산

timeAt632 = zeros(1, length(step_dis));  % Initialize timeAt632 as a matrix

for i = 1:length(step_dis)
    % 최소값과 최대값 계산
    minVoltage = min(data(step_dis(i)).V);
    maxVoltage = max(data(step_dis(i)).V);

    % 63.2% 값 계산
    targetVoltage = minVoltage + (1 - 0.632 ) * (maxVoltage - minVoltage);

    % 63.2%에 가장 가까운 값의 인덱스 찾기
    [~, idx] = min(abs(data(step_dis(i)).V - targetVoltage));

    % 해당 시간 찾기
    timeAt632(i) = data(step_dis(i)).t(idx);
    
    % data(step_dis(i)) 구조체에 timeAt632 필드를 추가하고 값 할당
    data(step_dis(i)).timeAt632 = timeAt632(i);
end

% C값 구하기
for i = 1:length(step_dis)
    data(step_dis(i)).C = data(step_dis(i)).timeAt632 / (data(step_dis(i)).R1s - data(step_dis(i)).R001s);
end

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

% step_dis 배열을 사용하여 데이터에 SOC 값 할당
for i = 1:length(step_dis)
    data(step_dis(i)).SOC = SOC(i);
end

% 데이터 길이보다 SOC 값이 부족할 경우 마지막 값으로 채움
if length(step_dis) > length(SOC)
    data(step_dis(end)).SOC = soc_values(end);
end

% 구조체 생성
optimized_params_struct = struct('R0', [], 'R1', [], 'C', [], 'SOC', [], 'avgI', [], 'm', [], 'Crate', []); % 'm' 필드 추가

% 초기 추정값 개수 설정
num_start_points = 10; % 원하는 시작점의 개수 설정

for i = 1:length(step_dis)
    deltaV_exp = data(step_dis(i)).deltaV;
    time_exp = data(step_dis(i)).t;
    avgI = data(step_dis(i)).avgI;  % 각 스텝의 평균 전류 가져오기
        
    % m 값 설정
    m = 1.5;

    % 스텝의 시간 길이 확인
    step_duration = time_exp(end) - time_exp(1);

    % Check if R1 and C are valid numbers
    R1_initial = data(step_dis(i)).R1;
    C_initial = data(step_dis(i)).C;

    % Check for valid initial guesses and data
    if step_duration >= 0 && isfinite(R1_initial) && isfinite(C_initial) && isfinite(data(step_dis(i)).R0)
        % 최적화를 위한 초기 추정값 생성
        initial_guesses = repmat([R1_initial, C_initial], num_start_points, 1);

        % fmincon을 사용하여 최적화 수행
        options = optimoptions('fmincon', 'Display', 'off', 'MaxIterations', 100);
        problem = createOptimProblem('fmincon', 'objective', @(params) cost_function(params, time_exp, deltaV_exp, avgI, m, data(step_dis(i)).R0), ...
            'x0', initial_guesses, 'lb', [0, 0], 'ub', [], 'options', options);
        ms = MultiStart('Display', 'off');

        [opt_params, ~] = run(ms, problem, num_start_points); % 여러 시작점으로 실행

        optimized_params_struct(i).R0 = data(step_dis(i)).R0; % R0 고정된 값 사용
        optimized_params_struct(i).R1 = opt_params(1);
        optimized_params_struct(i).C = opt_params(2);
        optimized_params_struct(i).SOC = mean(data(step_dis(i)).SOC); % 평균 SOC 값을 저장
        optimized_params_struct(i).Crate = avgI / capacity_Ah; % C-rate 계산
        optimized_params_struct(i).avgI = avgI; % 평균 전류 저장
        optimized_params_struct(i).m = m; % 계산된 m 값 저장
    else
        % 유효하지 않은 경우 NaN으로 채우거나 해당 스텝을 건너뜁니다.
        optimized_params_struct(i).R0 = NaN;
        optimized_params_struct(i).R1 = NaN;
        optimized_params_struct(i).C = NaN;
        optimized_params_struct(i).SOC = NaN;
        optimized_params_struct(i).Crate = NaN;
        optimized_params_struct(i).avgI = NaN;
        optimized_params_struct(i).m = NaN;
    end
end

% NaN 값을 가진 구조체 요소를 제거
optimized_params_struct = optimized_params_struct(~isnan([optimized_params_struct.R1]));

%% Figure 추가 1029

figure(100)

% 스텝 63, 64, 65의 원래 시간 복원
data(63).t = data(63).t + initialTimes(63);
data(64).t = data(64).t + initialTimes(64);
data(65).t = data(65).t + initialTimes(65);

% 스텝 63의 마지막 30초 데이터 선택
t63 = data(63).t;
maxTime63 = t63(end);
indices63 = find(t63 >= maxTime63 - 30);
data63_t = t63(indices63);
data63_I = data(63).I(indices63);
data63_V = data(63).V(indices63);

% 스텝 64의 전체 데이터 사용
data64_t = data(64).t;
data64_I = data(64).I;
data64_V = data(64).V;

% 스텝 65의 처음 30초 데이터 선택
t65 = data(65).t;
minTime65 = t65(1);
indices65 = find(t65 <= minTime65 + 30);
data65_t = t65(indices65);
data65_I = data(65).I(indices65);
data65_V = data(65).V(indices65);

% 시간, 전류, 전압 데이터 연결
Time_combined = vertcat(data63_t, data64_t, data65_t);
Current_combined = vertcat(data63_I, data64_I, data65_I);
Voltage_combined = vertcat(data63_V, data64_V, data65_V);

% Define the color order using lines(9)
c_mat = lines(9);

% 전류와 전압 그래프 그리기
plot(Time_combined, Current_combined, 'Color', c_mat(1,:), 'LineWidth', 3);
xlabel('time');

% 왼쪽 y축 설정 (Current)
yyaxis left
ylabel('Current (A)', 'Color', c_mat(1,:)); % Current y축 레이블 색깔 지정
ax = gca;
ax.YColor = c_mat(1,:); % y축 눈금 및 축 색깔 지정
ylim([-3 1]); % Current y축 범위 설정
hold on;

% 오른쪽 y축 설정 (Voltage)
yyaxis right
ylabel('Voltage (V)', 'Color', c_mat(2,:)); % Voltage y축 레이블 색깔 지정
ax.YColor = c_mat(2,:); % y축 눈금 및 축 색깔 지정
plot(Time_combined, Voltage_combined, 'Color', c_mat(2,:), 'LineWidth', 3 );

xlim([46624, 46650.4]);


% 모델 전압 그래프 추가 (스텝 64에 대해서만)
% 스텝 64의 데이터가 step_dis에서 몇 번째에 해당하는지 찾기
step64_index_in_step_dis = find(step_dis == 64);

if ~isempty(step64_index_in_step_dis)
    % 스텝 64의 시간 상대값 계산
    time_relative = data64_t - initialTimes(64);
    
    % 모델 전압 계산
    voltage_model_step64 = model_func(time_relative, optimized_params_struct(step64_index_in_step_dis).R0, ...
        optimized_params_struct(step64_index_in_step_dis).R1, optimized_params_struct(step64_index_in_step_dis).C, data(64).avgI);

    % 모델 전압에 스텝 63의 마지막 전압 값을 더하여 실제 전압으로 변환
    voltage_model_step64_adjusted = voltage_model_step64 + data63_V(end);

    % 모델 전압 그래프 추가 (검은색 점선)
    plot(data64_t, voltage_model_step64_adjusted, '--', 'Color', c_mat(3,:), 'LineWidth', 3);
    xlabel('Time (s)');
    ylabel('Current (A)');
    yyaxis left
    ylabel('Current (A)', 'Color', c_mat(1,:)); % Current y축 레이블 색깔 지정
    yyaxis right 
    ylabel('Voltage (V)');
    legend('Current', 'Voltage', 'Model Voltage');
    title('DCIR');
   


end

% xlabel('Time (s)');
% ylabel('Current (A)');
% yyaxis right 
% ylabel('Voltage (V)');
% legend('Current', 'Voltage', 'Model Voltage');
% title('DCIR');
% grid on;

hold off;

%% 함수

function cost = cost_function(params, time, deltaV, I, m, R0)
    R1 = params(1);
    C = params(2);
    
    % 모델 함수를 사용하여 예측 전압 계산
    voltage_model = model_func(time, R0, R1, C, I);
    
    % 오차 계산
    error = deltaV - voltage_model;
    
    % 시간에 따라 가중치 함수 적용
    % m 값을 사용하여 가중치 함수를 조절
    time_weights = exp(-m * time); 
    
    % 가중 평균 제곱근 오차(RMS 오차) 계산
    weighted_error = error .* time_weights;
    cost = sqrt(mean(weighted_error.^2));
end

function voltage = model_func(time, R0, R1, C, I)
    voltage = I * (R0 + R1 * (1 - exp(-time / (R1 * C))));
end
