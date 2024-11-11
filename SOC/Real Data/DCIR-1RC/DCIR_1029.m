clc; clear; close all;

%% 데이터 로드
% Uncomment the appropriate line based on your file location
% data = load('C:\Users\USER\Desktop\Panasonic 18650PF Data\Panasonic 18650PF Data\25degC\5 pulse disch\03-11-17_08.47 25degC_5Pulse_HPPC_Pan18650PF.mat');
data = load('G:\공유 드라이브\BSL_Data3\HPPC\03-11-17_08.47 25degC_5Pulse_HPPC_Pan18650PF.mat');

%% 시간, 전압, 전류 데이터 추출
time = data.meas.Time;
voltage = data.meas.Voltage;
current = data.meas.Current;

data1.I = current;
data1.V = voltage;
data1.t = time;

%% 전류 상태 구분
data1.type = char(zeros([length(data1.t), 1]));
data1.type(data1.I > 0) = 'C';  % Charging
data1.type(data1.I == 0) = 'R'; % Rest
data1.type(data1.I < 0) = 'D';  % Discharging

%% Step 구분
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

%% 데이터 라인 구조체 생성
data_line = struct('V', zeros(1, 1), 'I', zeros(1, 1), 't', zeros(1, 1), ...
    'indx', zeros(1, 1), 'type', char('R'), 'steptime', zeros(1, 1), ...
    'T', zeros(1, 1), 'SOC', zeros(1, 1));
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

%% 초기 SOC 설정
initial_SOC = 1;
capacity_Ah = 2.9; % 배터리 용량 (Ah)

%% Discharge step 구하기
step_chg = [];
step_dis = [];

for i = 1:length(data)
    if strcmp(data(i).type, 'C')
        step_chg(end+1) = i;
    elseif strcmp(data(i).type, 'D')
        step_dis(end+1) = i;
    end
end

%% R0, R1, C 추출 

% 평균 전류 구하기
for i = 1:length(data)
    data(i).avgI = mean(data(i).I);
end

% V 변화량 구하기
for i = 1:length(data)
    if i == 1
       data(i).deltaV = zeros(size(data(i).V));
    else
       data(i).deltaV = data(i).V - data(i-1).V(end);
    end
end

% Resistance 구하기 
for i = 1:length(data)
    if data(i).avgI == 0
        data(i).R = zeros(size(data(i).V));
    else 
        data(i).R = (data(i).deltaV / data(i).avgI) .* ones(size(data(i).V));
    end
end

% 시간 초기화
for i = 1:length(data)
    initialTime = data(i).t(1); % 초기 시간 저장
    data(i).t = data(i).t - initialTime; % 초기 시간을 빼서 시간 초기화
end

% R0, R1 계산 (Discharge steps only)
for i = 1:length(step_dis)
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
    if isnan(data(step_dis(i)).R1s) || isnan(data(step_dis(i)).R001s)
        data(step_dis(i)).C = NaN;
    else
        data(step_dis(i)).C = data(step_dis(i)).timeAt632 / (data(step_dis(i)).R1s - data(step_dis(i)).R001s);
    end
end

%% SOC 값을 정의된 패턴에 따라 생성
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

% 특정 데이터의 SOC 값 설정 (예: data(130).SOC = 0.05;)
% Ensure that data has at least 130 elements to avoid indexing errors
if length(data) >= 130
    data(130).SOC = 0.05;
end

%% 최적화 파라미터 구조체 생성
optimized_params_struct = struct('R0', [], 'R1', [], 'C', [], 'SOC', [], 'Crate', [], 'm', []); % 'Crate'로 수정

% 초기 추정값 개수 설정
num_start_points = 10; % 원하는 시작점의 개수 설정

for i = 1:length(step_dis)
    deltaV_exp = data(step_dis(i)).deltaV;
    time_exp = data(step_dis(i)).t;
    avgI = data(step_dis(i)).avgI;  % 각 스텝의 평균 전류 가져오기
        
    % m 값 설정 (여기서는 고정값 사용)
    m = 1.05;

    % 스텝의 시간 길이 확인
    step_duration = time_exp(end) - time_exp(1);

    if step_duration >= 0 % 스텝의 시간이 0 이상인 경우에만 저장
        % 최적화를 위한 초기 추정값 생성
        initial_guesses = repmat([data(step_dis(i)).R1, data(step_dis(i)).C], num_start_points, 1);

        % fmincon을 사용하여 최적화 수행
        options = optimoptions('fmincon', 'Display', 'off', 'MaxIterations', 100); % 'iter'에서 'off'로 변경
        problem = createOptimProblem('fmincon', 'objective', @(params) cost_function(params, time_exp, deltaV_exp, avgI, m, data(step_dis(i)).R0), ...
            'x0', initial_guesses(1,:), 'lb', [0, 0], 'ub', [], 'options', options);
        ms = MultiStart('Display', 'off'); % 'iter'에서 'off'로 변경

        [opt_params, ~] = run(ms, problem, num_start_points); % 여러 시작점으로 실행

        optimized_params_struct(i).R0 = data(step_dis(i)).R0; % R0 고정된 값 사용
        optimized_params_struct(i).R1 = opt_params(1);
        optimized_params_struct(i).C = opt_params(2);
        optimized_params_struct(i).SOC = mean(data(step_dis(i)).SOC); % 평균 SOC 값을 저장
        optimized_params_struct(i).Crate = abs(avgI) / capacity_Ah; % 절대값으로 변경 및 배터리 용량 사용
        optimized_params_struct(i).m = m; % 계산된 m 값 저장

        voltage_model = model_func(time_exp, optimized_params_struct(i).R0, opt_params(1), opt_params(2), avgI);

        % Save the model voltage for data(64)
        if step_dis(i) == 64
            data(step_dis(i)).voltage_model = voltage_model;
            data(step_dis(i)).time_exp = time_exp; % Time already starts at 0
        end

        % Optionally, you can store the model voltage for all steps if needed
        % data(step_dis(i)).voltage_model = voltage_model;
    end
end

%% R1 값의 이상치 제거

% R1 값의 평균과 표준 편차 계산
R1_values = [optimized_params_struct.R1];
R1_mean = mean(R1_values, 'omitnan');
R1_std = std(R1_values, 'omitnan');

% 이상치를 감지하기 위한 임계값 설정 (예: 평균 ± 3 표준편차)
threshold = 3;
outliers = abs(R1_values - R1_mean) > threshold * R1_std;

% 이상치가 아닌 데이터만 선택
optimized_params_filtered = optimized_params_struct(~outliers);

% 이상치가 제거된 데이터를 저장
save('optimized_params_filtered_105.mat', 'optimized_params_filtered');
save('optimized_params_struct_105.mat', 'optimized_params_struct');

%% R0, R1, C 시각화

% 이상치가 제거된 데이터로 그래프 그리기
R0_values = [optimized_params_filtered.R0];
R1_values = [optimized_params_filtered.R1];
C_values = [optimized_params_filtered.C];

SOC_values = [optimized_params_filtered.SOC];
Crate_values = [optimized_params_filtered.Crate];

unique_SOC = unique(SOC_values);
unique_Crate = unique(Crate_values);

% R0 시각화
[X, Y] = meshgrid(linspace(min(SOC_values), max(SOC_values), 100), linspace(min(Crate_values), max(Crate_values), 100));
R0_matrix = griddata(SOC_values, Crate_values, R0_values, X, Y, 'cubic');

figure('Name', 'R0 vs SOC and Crate');
surf(X, Y, R0_matrix);
xlabel('SOC');
ylabel('C-rate');
zlabel('R0');
title('R0 vs SOC and C-rate');
shading interp;
grid on;
zlim([0, 0.05]);

figure('Name', 'R0 Contour');
contourf(X, Y, R0_matrix, 20);
xlabel('SOC');
ylabel('C-rate');
title('Contour of R0 vs SOC and C-rate');
colorbar;
grid on;

% R1 시각화
R1_matrix = griddata(SOC_values, Crate_values, R1_values, X, Y, 'cubic');

figure('Name', 'R1 vs SOC and Crate');
surf(X, Y, R1_matrix);
xlabel('SOC');
ylabel('C-rate');
zlabel('R1');
title('R1 vs SOC and C-rate');
shading interp;
grid on;

figure('Name', 'R1 Contour');
contourf(X, Y, R1_matrix, 20);
xlabel('SOC');
ylabel('C-rate');
title('Contour of R1 vs SOC and C-rate');
colorbar;
grid on;

% C 시각화
C_matrix = griddata(SOC_values, Crate_values, C_values, X, Y, 'cubic');

figure('Name', 'C vs SOC and Crate');
surf(X, Y, C_matrix);
xlabel('SOC');
ylabel('C-rate');
zlabel('C');
title('C vs SOC and C-rate');
shading interp;
grid on;

figure('Name', 'C Contour');
contourf(X, Y, C_matrix, 20);
xlabel('SOC');
ylabel('C-rate');
title('Contour of C vs SOC and C-rate');
colorbar;
grid on;

%% R0, R1, C 평균값 계산 (이상치 제거 후)
R0_mean_filtered = mean([optimized_params_filtered.R0], 'omitnan');
R1_mean_filtered = mean([optimized_params_filtered.R1], 'omitnan');
C_mean_filtered = mean([optimized_params_filtered.C], 'omitnan');

% 결과 출력
fprintf('R0의 평균 값 : %.6f Ohm\n', R0_mean_filtered);
fprintf('R1의 평균 값 : %.6f Ohm\n', R1_mean_filtered);
fprintf('C의 평균 값 : %.6f F\n', C_mean_filtered);

%% Figure 추가: SOC 50% 및 1C 펄스에 대한 전류, 전압, 모델 피팅 결과

% Define the index for the SOC 50% and 1C pulse
target_step = 64; % Assuming step 64 corresponds to data(64)

% Find the corresponding index in step_dis
target_index = find(step_dis == target_step, 1);

% Validate the target index
if isempty(target_index) || target_index > length(data)
    error('Target index for SOC 50%% and 1C pulse (data(64)) is invalid.');
end

% Extract data(63) - Rest before the pulse
if target_step - 1 < 1
    error('Data(63) does not exist. Ensure that data has at least 64 steps.');
end
rest_before = data(target_step - 1);
% Take the last 30 seconds of this rest period
duration_before = 30; % seconds
indices_before = rest_before.t >= (rest_before.t(end) - duration_before);
I_before = rest_before.I(indices_before);
V_before = rest_before.V(indices_before);
t_before = rest_before.t(indices_before);

% Adjust time to start from zero
t_before = t_before - t_before(1);

% Extract data(64) - The pulse
pulse = data(target_step);
I_pulse = pulse.I;
V_pulse = pulse.V;
t_pulse = pulse.t;

% Adjust time to continue from the end of t_before
t_pulse = t_pulse + t_before(end);

% Extract data(65) - Rest after the pulse
if target_step + 1 > length(data)
    error('Data(65) does not exist. Ensure that data has at least 65 steps.');
end
rest_after = data(target_step + 1);
% Take the first 30 seconds of this rest period
duration_after = 30; % seconds
indices_after = rest_after.t <= (rest_after.t(1) + duration_after);
I_after = rest_after.I(indices_after);
V_after = rest_after.V(indices_after);
t_after = rest_after.t(indices_after);

% Adjust time to continue from the end of t_pulse
t_after = t_after + t_pulse(end);

% Concatenate all data
I_total = [I_before; I_pulse; I_after];
V_total = [V_before; V_pulse; V_after];
t_total = [t_before; t_pulse; t_after];

% Retrieve the model voltage and time for data(64)
if isfield(pulse, 'voltage_model') && isfield(pulse, 'time_exp')
    voltage_model = pulse.voltage_model;
    t_model = pulse.time_exp + t_before(end); % Adjust time to match concatenated time
else
    error('Model voltage data is not available for data(64). Ensure optimization was successful.');
end

% Adjust the model voltage to align with the actual voltage
% Assuming that voltage_model starts from zero, add V_before(end) as the offset
V_initial = V_before(end);
voltage_model_adjusted = voltage_model + V_initial;

% Create the figure
figure('Position', [100, 100, 1200, 700]);
hold on;

% Plot current on the left y-axis
yyaxis left;
plot(t_total, I_total, 'b-', 'LineWidth', 2);
ylabel('Current (A)', 'FontSize', 12);
ylim([min(I_total) - 0.1, max(I_total) + 0.1]); % Adjust limits if necessary
grid on;

% Plot voltage on the right y-axis
yyaxis right;
plot(t_total, V_total, 'r-', 'LineWidth', 2);
ylabel('Voltage (V)', 'FontSize', 12);
ylim([min(V_total) - 0.1, max(V_total) + 0.1]); % Adjust limits if necessary

% Overlay the model voltage during the pulse
% Find the time range corresponding to the pulse
pulse_start_time = t_before(end);
pulse_end_time = t_before(end) + t_pulse(end);
pulse_indices = t_total >= pulse_start_time & t_total <= pulse_end_time;

% Plot the model voltage only during the pulse
plot(data(64).t, voltage_model_adjusted, 'g--', 'LineWidth', 2);

% Add labels and title
xlabel('Time (s)', 'FontSize', 12);
title('Current and Voltage with Model Fitting at SOC 50% and 1C Pulse', 'FontSize', 14);

% Add legend
legend('Current', 'Voltage', 'Model Voltage (Pulse)', 'Location', 'Best');


% Add text annotations for SOC and C-rate
soc_text = sprintf('SOC: %.0f%%', pulse.SOC * 100);
crate_text = sprintf('C-rate: %.1fC', pulse.Crate);
text(t_total(1) + 5, max(V_total) - 0.1, {soc_text, crate_text}, 'FontSize', 12, 'Color', 'black', 'FontWeight', 'bold');

% Adjust font size and grid
set(gca, 'FontSize', 12);
hold off;

% Optional: Save the figure
% saveas(gcf, 'SOC50_Crate1C_Fitting_Result.png');

%% End of Script

%% 함수 정의

function cost = cost_function(params, time, deltaV, I, m, R0)
    R1 = params(1);
    C = params(2);
    
    % 모델 함수를 사용하여 예측 전압 계산
    voltage_model = model_func(time, R0, R1, C, I);
    
    % 오차 계산
    error = deltaV - voltage_model;
    
    % 시간에 따라 가중치 함수 적용
    time_weights = exp(-m * time); 
    
    % 가중 평균 제곱근 오차(RMS 오차) 계산
    weighted_error = error .* time_weights;
    cost = sqrt(mean(weighted_error.^2));
end

function voltage = model_func(time, R0, R1, C, I)
    voltage = I * (R0 + R1 * (1 - exp(-time / (R1 * C))));
end
