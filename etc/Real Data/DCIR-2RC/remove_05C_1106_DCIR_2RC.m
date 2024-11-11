clc; clear; close all;

%% 데이터 로드
% 1RC 모델 결과 로드
load('optimized_params_struct_final_ver2.mat'); % 1RC 모델 결과 로드

% R1과 C1 값을 배열로 추출
R0_values = [optimized_params_struct_final_ver2.R0];
R1_values = [optimized_params_struct_final_ver2.R1];
C1_values = [optimized_params_struct_final_ver2.C];

%% 원본 데이터 로드
% data_raw = load('C:\Users\USER\Desktop\Panasonic 18650PF Data\Panasonic 18650PF Data\25degC\5 pulse disch\03-11-17_08.47 25degC_5Pulse_HPPC_Pan18650PF.mat');
data_raw = load('G:\공유 드라이브\BSL_Data3\HPPC\03-11-17_08.47 25degC_5Pulse_HPPC_Pan18650PF.mat');

% 수정된 부분: data_raw에서 데이터를 가져옵니다.
time = data_raw.meas.Time;
voltage = data_raw.meas.Voltage;
current = data_raw.meas.Current;

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
    'steptime', zeros(1, 1), 'T', zeros(1, 1), 'SOC', zeros(1, 1), 'avgI', zeros(1, 1));
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
    data(i_step).avgI = mean(data(i_step).I); % 각 스텝의 평균 전류 계산
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

%% deltaV 계산 (전압 변화량)

for i = 1 : length(data)
    if i == 1
       data(i).deltaV = zeros(size(data(i).V));
    else
       data(i).deltaV = data(i).V - data(i-1).V(end);
    end
end

%% 63.2% 값을 이용한 tau 및 C1 계산

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

% C1 값을 구하기 (1RC 모델에서 R1을 사용)
for i = 1:length(step_dis)
    R1 = R1_values(i);
    if R1 ~= 0
        data(step_dis(i)).C1 = data(step_dis(i)).timeAt632 / R1;
    else
        data(step_dis(i)).C1 = NaN;
    end
end

%% SOC 값을 정의된 패턴에 따라 생성
soc_values = [1, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05];
steps_per_level = 5;

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

for i = 1:length(step_dis)
    data(step_dis(i)).SOC = SOC(i);
end


%% 2RC 모델 optmizied struct 생성
optimized_params_struct_final_ver2_2RC = struct('R0', [], 'R1', [], 'C1', [], 'R2', [], 'C2', [], 'SOC', [], 'Crate', []);

num_start_points = 10; 

%% 2RC 모델 최적화 루프
for idx = 1:length(step_dis)
    i = step_dis(idx);
    deltaV_exp = data(i).deltaV;
    time_exp = data(i).t - data(i).t(1); % 시간 0부터 시작하도록 조정
    avgI = data(i).avgI;  % 각 스텝의 평균 전류 가져오기

    % 1RC 모델에서 R0, R1, C1 가져오기
    R0 = R0_values(idx);
    R1 = R1_values(idx);
    C1 = C1_values(idx);

    % 초기 추정값 생성 (R2, C2)
    R2_init = R1 * 1.1; % 초기 추정값 
    C2_init = C1 * 1.1; % 초기 추정값 

    initial_guess = [R2_init, C2_init];

    % fmincon을 사용하여 최적화 수행
    options = optimoptions('fmincon', 'Display', 'none', 'MaxIterations', 1000, 'MaxFunctionEvaluations', 5000);
    problem = createOptimProblem('fmincon', ...
        'objective', @(params) cost_function(params, time_exp, deltaV_exp, avgI, R0, R1, C1), ...
        'x0', initial_guess, ...
        'lb', [0, 0], ...
        'ub', [2.5, inf], ...  % R2의 상한 두기
        'options', options);
    ms = MultiStart('Display', 'off');

    % 최적화 실행
    [opt_params, cost] = run(ms, problem, num_start_points);

    if ~isempty(opt_params)
        % 최적화된 파라미터 저장
        optimized_params_struct_final_ver2_2RC(idx).R0 = R0; % 고정된 값
        optimized_params_struct_final_ver2_2RC(idx).R1 = R1; % 고정된 값
        optimized_params_struct_final_ver2_2RC(idx).C1 = C1; % 고정된 값
        optimized_params_struct_final_ver2_2RC(idx).R2 = opt_params(1);
        optimized_params_struct_final_ver2_2RC(idx).C2 = opt_params(2);
        optimized_params_struct_final_ver2_2RC(idx).SOC = data(i).SOC;
        optimized_params_struct_final_ver2_2RC(idx).Crate = avgI / data(step_dis(2)).avgI;
       
    end
end

%% 플로팅 with Subplots
% Define the number of subplots per figure
plots_per_fig = 9; % 3 rows x 3 columns
num_figures = ceil(length(step_dis) / plots_per_fig);

% Initialize figure counter and subplot index
fig_counter = 1;
subplot_idx = 1;

figure(fig_counter);
set(fig_counter, 'Units', 'pixels', 'Position', [100, 100, 1200, 800]); % [left, bottom, width, height]
sgtitle('Comparison of Experimental Data and 2RC Model Results');

for idx = 1:length(step_dis)
    i = step_dis(idx);
    
    if (data(i).t(end) - data(i).t(1)) >= 0
        if subplot_idx > plots_per_fig
            fig_counter = fig_counter + 1;
            figure(fig_counter);
            set(fig_counter, 'Units', 'pixels', 'Position', [100, 100, 1200, 800]); % [left, bottom, width, height]
            sgtitle('Comparison of Experimental Data and 2RC Model Results');
            subplot_idx = 1;
        end

        subplot(3, 3, subplot_idx);
        hold on;

        deltaV_exp = data(i).deltaV;
        time_exp = data(i).t - data(i).t(1); 
        avgI = data(i).avgI;
        optimized_R0 = optimized_params_struct_final_ver2_2RC(idx).R0;
        optimized_R1 = optimized_params_struct_final_ver2_2RC(idx).R1;
        optimized_C1 = optimized_params_struct_final_ver2_2RC(idx).C1;
        optimized_R2 = optimized_params_struct_final_ver2_2RC(idx).R2;
        optimized_C2 = optimized_params_struct_final_ver2_2RC(idx).C2;
        soc = data(i).SOC; 
        crate = optimized_params_struct_final_ver2_2RC(idx).Crate;


        % Generate model voltage
        voltage_model = model_func(time_exp, optimized_R0, optimized_R1, optimized_R2, optimized_C1, optimized_C2, avgI);

        % Plot experimental data
        plot(time_exp, deltaV_exp, 'b-', 'LineWidth', 1.5, 'DisplayName', '실험 데이터');

        % Plot model data
        plot(time_exp, voltage_model, 'r--', 'LineWidth', 1.5, 'DisplayName', '2RC 모델 결과');
       
        if exist('xline', 'file')
            xline(data(i).timeAt632 - data(i).t(1), 'k--', 'LineWidth', 1.0, 'DisplayName', '63.2% 시간');
        else
            
            ylim_current = ylim;
            line([data(i).timeAt632 - data(i).t(1), data(i).timeAt632 - data(i).t(1)], ylim_current, 'Color', 'k', 'LineStyle', '--', 'DisplayName', '63.2% 시간');
        end
       
        soc_text = sprintf('SOC: %.2f%%', soc * 100);
        crate_text = sprintf('C-rate: %.2f', crate);
        text(time_exp(1) + 0.05*(time_exp(end)-time_exp(1)), ...
             max(deltaV_exp)*0.9, ...
             {soc_text, crate_text}, 'FontSize', 8, 'Color', 'k', 'FontWeight', 'bold');

        xlabel('시간 (sec)', 'FontSize', 8);
        ylabel('전압 강하 (V)', 'FontSize', 8);
        title(sprintf('Discharge Step %d', idx), 'FontSize', 10);
        legend('Location', 'best', 'FontSize', 6);
        grid on;

        hold off;

        subplot_idx = subplot_idx + 1;
    end
end

%% R0, R1, R2, C1, C2 Plot
% R0, R1, R2, C1, C2 (SOC, C-rate) 3D and Contour Plots

SOC = [optimized_params_struct_final_ver2_2RC.SOC]';
Crate = [optimized_params_struct_final_ver2_2RC.Crate]';
R0 = [optimized_params_struct_final_ver2_2RC.R0]';
R1 = [optimized_params_struct_final_ver2_2RC.R1]';
R2 = [optimized_params_struct_final_ver2_2RC.R2]';
C1 = [optimized_params_struct_final_ver2_2RC.C1]';
C2 = [optimized_params_struct_final_ver2_2RC.C2]';

num_grid = 100; 
SOC_grid = linspace(min(SOC), max(SOC), num_grid);
Crate_grid = linspace(min(Crate), max(Crate), num_grid);
[SG, CG] = meshgrid(SOC_grid, Crate_grid);

R0_grid = griddata(SOC, Crate, R0, SG, CG, 'cubic');
R1_grid = griddata(SOC, Crate, R1, SG, CG, 'cubic');
R2_grid = griddata(SOC, Crate, R2, SG, CG, 'cubic');
C1_grid = griddata(SOC, Crate, C1, SG, CG, 'cubic');
C2_grid = griddata(SOC, Crate, C2, SG, CG, 'cubic');

%% R0, R1, R2, C1, C2 Plot 생성
create_plots(SG, CG, R0_grid, 'R0');
create_plots(SG, CG, R1_grid, 'R1');
create_plots(SG, CG, R2_grid, 'R2');
create_plots(SG, CG, C1_grid, 'C1');
create_plots(SG, CG, C2_grid, 'C2');

%% function

% 모델 함수 정의 (2RC 모델)
function voltage = model_func(time, R0, R1, R2, C1, C2, I)
    voltage = I * (R0 ...
        + R1 * (1 - exp(-time / (R1 * C1))) ...
        + R2 * (1 - exp(-time / (R2 * C2))));
end

% cost 함수 정의 (2RC 모델)
function cost = cost_function(params, time, deltaV, I, R0, R1, C1)
    R2 = params(1);
    C2 = params(2);

    voltage_model = model_func(time, R0, R1, R2, C1, C2, I);

    error = deltaV - voltage_model;

    cost = sqrt(mean(error.^2));
end

%% pPlot
function create_plots(SG, CG, Param_grid, Param_name)
    figure();
    % 3D Surface Plot
    surf(SG, CG, Param_grid, 'EdgeColor', 'none');
    xlabel('SOC');
    ylabel('C-rate');
    zlabel(Param_name);
    title([Param_name ' vs SOC and C-rate']);
    colorbar;
    view(45, 30); 
    grid on;

    % Contour Plot
    figure();
    contourf(SG, CG, Param_grid, 20, 'LineColor', 'none');
    xlabel('SOC');
    ylabel('C-rate');
    title([Param_name ' Contour Plot']);
    colorbar;
    grid on;
end

