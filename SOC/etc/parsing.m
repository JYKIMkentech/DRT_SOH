clc; clear; close all;

% Load UDDS data
load('C:\Users\USER\Desktop\Panasonic 18650PF Data\Panasonic 18650PF Data\25degC\Drive cycles\03-21-17_00.29 25degC_UDDS_Pan18650PF.mat');
udds_current = meas.Current; % UDDS 전류 데이터
udds_voltage = meas.Voltage; % UDDS 전압 데이터
udds_time = meas.Time; % UDDS 시간 데이터

% Load SOC-OCV structure
load('soc_ocv.mat', 'soc_ocv');
soc_values = soc_ocv(:, 1);
ocv_values = soc_ocv(:, 2);

% 초기 OCV 값을 이용하여 초기 SOC를 설정
initial_ocv = udds_voltage(1);
initial_soc = interp1(ocv_values, soc_values, initial_ocv, 'linear', 'extrap');

% 파싱 및 SOC 계산을 위한 초기화
initial_capacity = 2.9950; % Ah
SOC = initial_soc; % 초기 SOC 설정
true_SOC = zeros(length(udds_current), 1);
true_SOC(1) = SOC;

% 전류 상태 파싱 (C, D, R)
data1.I = udds_current;
data1.V = udds_voltage;
data1.t = udds_time;

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

    % SOC 계산
    if length(data(i_step).I) == 1
        if i_step > 1
            data(i_step).SOC = data(i_step - 1).SOC;
        else
            data(i_step).SOC = SOC; % 초기 SOC 값
        end
    else
        if data(i_step).type == 'D'
            SOC = SOC - trapz(seconds(data(i_step).steptime), data(i_step).I) / (initial_capacity * 3600);
        elseif data(i_step).type == 'C'
            SOC = SOC + trapz(seconds(data(i_step).steptime), data(i_step).I) / (initial_capacity * 3600);
        end

        % Ensure SOC is within bounds [0, 1]
        SOC = max(0, min(1, SOC));
        data(i_step).SOC = SOC;
    end
end

% Plot SOC profile
soc_profile = [data.SOC];
figure;
plot(soc_profile, 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('SOC');
title('SOC Profile');
grid on;


