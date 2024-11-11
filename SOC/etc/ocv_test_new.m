clc; clear;close all;

% Load C/20 charge/discharge data
load('C:\Users\deu04\OneDrive\바탕 화면\wykht8y7tg-1\Panasonic 18650PF Data\Panasonic 18650PF Data\25degC\C20 OCV and 1C discharge tests_start_of_tests/05-08-17_13.26 C20 OCV Test_C20_25dC.mat');
current = meas.Current; % Current data
voltage = meas.Voltage; % Voltage data
time = meas.Time; % Time data

% Parse current state (C, D, R)
data1.I = current;
data1.V = voltage;
data1.t = time;

% Determine current state
data1.type = char(zeros([length(data1.t), 1]));
data1.type(data1.I > 0) = 'C';
data1.type(data1.I == 0) = 'R';
data1.type(data1.I < 0) = 'D';

% Identify steps
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
    data(i_step).T = zeros(size(range)); % Set temperature data to 0 since it's not available
end

% Calculate capacity and SOC
for j = 1:length(data)
    if length(data(j).t) > 1
        data(j).Q = abs(trapz(data(j).t, data(j).I)) / 3600; %[Ah]
        data(j).cumQ = abs(cumtrapz(data(j).t, data(j).I)) / 3600; %[Ah]
    else
        data(j).Q = 0;
        data(j).cumQ = zeros(size(data(j).t));
    end
end

% Identify OCV charging steps
step_ocv_chg = find(strcmp({data.type}, 'C')); % Find all charging steps
step_ocv_dis = find(strcmp({data.type}, 'D')); % Find all discharging steps

data(2).SOC = 1 - data(2).cumQ/data(2).Q;
data(4).SOC = data(4).cumQ/data(4).Q;

%plot
figure();
plot(data(2).SOC,data(2).V)
hold on
plot(data(4).SOC,data(4).V)
xlabel('SOC')
ylabel('OCV')


% 충전 및 방전 데이터 구조체에서 전압 데이터 추출
V_charge = data(4).V; % 충전 전압
V_discharge = data(2).V; % 방전 전압

% 겹치는 전압 범위 찾기
V_min = max(min(V_charge), min(V_discharge));
V_max = min(max(V_charge), max(V_discharge));

% 해당 범위에서 충전 및 방전 SOC 매칭 및 평균 계산
SOC_avg = [];
V_common = [];
for V = linspace(V_min, V_max, 1000)
    % 충전 및 방전 전압에서 가장 가까운 값을 찾음
    [~, idx_ch] = min(abs(V_charge - V));
    [~, idx_dis] = min(abs(V_discharge - V));
    SOC_avg = [SOC_avg, mean([data(4).SOC(idx_ch), data(2).SOC(idx_dis)])];
    V_common = [V_common, V];
end

% 7차 다항식으로 SOC에 대한 OCV fitting
p = polyfit(SOC_avg, V_common, 7);

% 계수 출력
disp('7차 다항식의 계수:');
disp(p);

% 다항식 형태로 보기 좋게 출력
fprintf('OCV = ');
for i = 1:length(p)
    % 마지막 항이 아닌 경우
    if i < length(p)
        fprintf('%.4f*SOC^%d + ', p(i), length(p)-i);
    else
        % 마지막 항
        fprintf('%.4f\n', p(i));
    end
end


% 평균 SOC 곡선 그리기
figure;
plot(data(2).SOC, data(2).V, 'b'); hold on;
plot(data(4).SOC, data(4).V, 'r'); hold on;
plot(SOC_avg, V_common, 'k--');
xlabel('SOC');
ylabel('OCV');
legend('Discharge', 'Charge', 'New OCV');
title('OCV vs. SOC');







% % Extract SOC and OCV for charging
% SOC_chg = [];
% OCV_chg = [];
% for j = step_ocv_chg'
%     if ~isempty(data(j).cumQ)
%         soc = data(j).cumQ / data(j).Q;
%         SOC_chg = [SOC_chg; soc];
%         OCV_chg = [OCV_chg; data(j).V];
%     end
% end
% 
% % Extract SOC and OCV for discharging
% SOC_dis = [];
% OCV_dis = [];
% for j = step_ocv_dis'
%     if ~isempty(data(j).cumQ)
%         soc = data(j).cumQ / data(j).Q;
%         SOC_dis = [SOC_dis; soc];
%         OCV_dis = [OCV_dis; data(j).V];
%     end
% end
% 
% % Create SOC-OCV structures
% ocv_chg = [SOC_chg OCV_chg];
% ocv_dis = [SOC_dis OCV_dis];
% 
% % Save SOC-OCV structures for later use
% save('soc_ocv_chg.mat', 'ocv_chg');
% save('soc_ocv_dis.mat', 'ocv_dis');
% 
% % Plot data
% figure;
% plot(time, current);
% hold on;
% plot(time, voltage);
% xlabel('Time');
% ylabel('Voltage');
% yyaxis right;
% ylabel('Current');
% legend('Current', 'Voltage');
% title('OCV Test');