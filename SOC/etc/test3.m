clc; clear; close all;

% 데이터 로드
data = load('C:\Users\deu04\OneDrive\바탕 화면\wykht8y7tg-1\Panasonic 18650PF Data\Panasonic 18650PF Data\25degC\5 pulse disch\03-11-17_08.47 25degC_5Pulse_HPPC_Pan18650PF.mat');
%data = load('C:\Users\김준연\Desktop\wykht8y7tg-1\Panasonic 18650PF Data\Panasonic 18650PF Data\25degC\5 pulse disch\03-11-17_08.47 25degC_5Pulse_HPPC_Pan18650PF.mat');
% 시간, 전압, 전류 데이터 추출
% 시간, 전압, 전류 데이터 추출
time = data.meas.Time;
voltage = data.meas.Voltage;
current = data.meas.Current;

% 전류 상태 파싱 (C, D, R)
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

for i_step = 1:num_step
    range = find(data1.step == vec_step(i_step));
    data(i_step).V = data1.V(range);
    data(i_step).I = data1.I(range);
    data(i_step).t = data1.t(range);
    data(i_step).indx = range;
    data(i_step).type = data1.type(range(1));
    data(i_step).steptime = data1.t(range);
    data(i_step).T = zeros(size(range)); % 온도 데이터가 없으므로 0으로 설정
    
%     % SOC 계산
%     if i_step == 1
%         data(i_step).SOC = initial_SOC + cumtrapz(data(i_step).t, data(i_step).I) / (capacity_Ah * 3600);
%     else
%         data(i_step).SOC = data(i_step-1).SOC(end) + cumtrapz(data(i_step).t, data(i_step).I) / (capacity_Ah * 3600);
%     end
end

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

% 시간 초기화
for i = 1 : length(data)
    initialTime = data(i).t(1); % 초기 시간 저장
    data(i).t = data(i).t - initialTime; % 초기 시간을 빼서 시간 초기화
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

% % 시간 초기화
% for i = 1 : length(data)
%     initialTime = data(i).t(1); % 초기 시간 저장
%     data(i).t = data(i).t - initialTime; % 초기 시간을 빼서 시간 초기화
% end


timeAt632 = zeros(1, length(step_dis));  % Initialize timeAt632 as a matrix

for i = 1:length(step_dis)
    plot(data(step_dis(i)).t, data(step_dis(i)).V);

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

    % 해당 시간에 선 그리기
    line([timeAt632(i), timeAt632(i)], [minVoltage, maxVoltage], 'Color', 'red', 'LineStyle', '--');

    xlabel('Time');
    ylabel('Voltage (V)', 'fontsize', 12);
    title('Voltage - Time Graph');
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

data(130).SOC = 0.5;

% 구조체 생성
optimized_params_struct = struct('R0', [], 'R1', [], 'C', [], 'SOC', [], 'avgI', []);

% 초기 추정값 개수 설정
num_start_points = 10; % 원하는 시작점의 개수 설정

% 구조체 생성
optimized_params_struct = struct('R0', [], 'R1', [], 'C', [], 'SOC', []);

for i = 1:length(step_dis)
    deltaV_exp = data(step_dis(i)).deltaV;
    time_exp = data(step_dis(i)).t;
    avgI = data(step_dis(i)).avgI;  % 각 스텝의 평균 전류 가져오기
    m = 0.2 / data(step_dis(i)).timeAt632; % timeAt632의 역수를 m으로 설정

    m = 0.5 / data(step_dis(i)).timeAt632; % timeAt632의 역수를 m으로 설정

    % 스텝의 시간 길이 확인
    step_duration = time_exp(end) - time_exp(1);

    if step_duration >= 5 % 스텝의 시간이 5초 이상인 경우에만 저장
        % 최적화를 위한 여러 초기 추정값 생성
        initial_guesses = repmat([data(step_dis(i)).R1, data(step_dis(i)).C], num_start_points, 1);

        % multistart
        for k = 1:num_start_points
            initial_guesses(k, 1) = initial_guesses(k, 1) * (1 + 0.1 * (rand - 0.5)); % R1 변동
            initial_guesses(k, 2) = initial_guesses(k, 2) * (1 + 0.1 * (rand - 0.5)); % C 변동
        end
        % 최적화를 위한 초기 추정값 (R0를 제외하고 R1과 C만 포함)
        initial_guess = [data(step_dis(i)).R1, data(step_dis(i)).C];

        % fmincon을 사용하여 최적화 수행
        options = optimoptions('fmincon', 'Display', 'iter', 'MaxIterations', 100);
        problem = createOptimProblem('fmincon', 'objective', @(params) cost_function(params, time_exp, deltaV_exp, avgI, m, data(step_dis(i)).R0), ...
            'x0', initial_guesses, 'lb', [0, 0], 'ub', [], 'options', options);
            'x0', initial_guess, 'lb', [0, 0], 'ub', [], 'options', options);
        ms = MultiStart('Display', 'iter');

        [opt_params, ~] = run(ms, problem, num_start_points); % 여러 시작점으로 실행

        optimized_params_struct(i).R0 = data(step_dis(i)).R0; % R0 고정된 값 사용
        optimized_params_struct(i).R1 = opt_params(1);
        optimized_params_struct(i).C = opt_params(2);
        optimized_params_struct(i).SOC = mean(data(step_dis(i)).SOC); % 평균 SOC 값을 저장
        optimized_params_struct(i).Crate = avgI/data(step_dis(2)).avgI; % 평균 전류 저장

        
        voltage_model = model_func(time_exp, optimized_params_struct(i).R0, opt_params(1), opt_params(2), avgI);

        figure('Position', [0 0 800 600]);

        lw = 2;  % Desired line width
        msz = 10;  % Marker size

        color1 = [0, 0.4470, 0.7410];  % Blue
        color2 = [0.8500, 0.3250, 0.0980];  % Orange
        % Create a subplot with two rows and one column
        subplot(3, 1, [1 2]); % Larger subplot for data and model results

        % Plot the data with blue solid line
        plot(time_exp, deltaV_exp, 'b-', 'LineWidth', lw, 'Color', color1);
        hold on;

        % Plot the model results with orange dashed line
        plot(time_exp, voltage_model, 'r--', 'LineWidth', lw, 'Color', color2);
        
        % 63.2% 시간에 대한 수직선 추가
        timeAt632_plot = data(step_dis(i)).timeAt632;
        line([timeAt632_plot, timeAt632_plot], [min(deltaV_exp), max(deltaV_exp)], 'Color', 'green', 'LineStyle', '--');

        % SOC 값 표시
        soc_text = sprintf('SOC: %.2f%%', optimized_params_struct(i).SOC * 100);
        text(time_exp(1) + (time_exp(end) - time_exp(1)) * 0.05, max(deltaV_exp) * 0.9, soc_text, 'FontSize', 12, 'Color', 'black', 'FontWeight', 'bold');
        % avgI 값 표시
        avgI_text = sprintf('avgI: %.2fA', avgI);
        text(time_exp(1) + (time_exp(end) - time_exp(1)) * 0.05, max(deltaV_exp) * 0.9, avgI_text, 'FontSize', 12, 'Color', 'black', 'FontWeight', 'bold');

        legend('실험 데이터', '모델 결과', '63.2% 시간');
        xlabel('시간 (sec)');
        ylabel('전압 (V)');
        title('실험 데이터와 모델 결과');

        % Set font size and line width for the axis
        set(gca, 'FontSize', 16, 'LineWidth', 2);

        % Create a smaller subplot for the weight function
        subplot(3, 1, 3); % Smaller subplot for the weight function

        % Plot the weight function 
        weight_function = exp(-m * time_exp);
        plot(time_exp, weight_function, 'g-', 'LineWidth', lw, 'Color', [0, 1, 0]);

        legend('Weight Function');
        xlabel('시간 (sec)');
        ylabel('가중치');
        title('가중치 함수');

        % Set font size and line width for the axis
        set(gca, 'FontSize', 16, 'LineWidth', 2);

        % Adjust subplot spacing manually by changing the position of subplots
        set(gca, 'Position', [0.13, 0.1, 0.775, 0.25]);
        set(gcf, 'Position', [0, 0, 800, 800]);
    end
end


%% R0 그래프 그리기


% R0, SOC, Crate 값을 추출
R0_values = [optimized_params_struct.R0];
SOC_values = [optimized_params_struct.SOC];
Crate_values = [optimized_params_struct.Crate];

% 고유한 SOC 및 Crate 값을 가져옴
unique_SOC = unique(SOC_values);
unique_Crate = unique(Crate_values);

% 원래 데이터로부터 그리드 생성
[X, Y] = meshgrid(unique_SOC, unique_Crate);

% 원래 데이터를 보간하여 더 많은 데이터 포인트 생성
[Xq, Yq] = meshgrid(linspace(min(unique_SOC), max(unique_SOC), 100), linspace(min(unique_Crate), max(unique_Crate), 100));
R0_matrix = griddata(SOC_values, Crate_values, R0_values, Xq, Yq, 'cubic');

% 3차원 곡면 그래프 그리기
figure;
surf(Xq, Yq, R0_matrix);
xlabel('SOC');
ylabel('Crate');
zlabel('R0');
title('R0 vs SOC and Crate');
shading interp; 
grid on;

% Z축 범위 조정
%zlim([0, 1]); 

% 2차원 등고선 그래프 그리기
figure;
contourf(Xq, Yq, R0_matrix, 20);
xlabel('SOC');
ylabel('Crate');
title('Contour of R0 vs SOC and Crate');
colorbar; 
grid on;



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





