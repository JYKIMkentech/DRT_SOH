clear; clc; close all;

%% 0. 폰트 크기 및 색상 매트릭스 설정
% Font size settings
axisFontSize = 14;      % 축의 숫자 크기
titleFontSize = 16;     % 제목의 폰트 크기
legendFontSize = 12;    % 범례의 폰트 크기
labelFontSize = 14;     % xlabel 및 ylabel의 폰트 크기

% Color matrix 설정
c_mat = lines(9);  % 9개의 고유한 색상 정의

%% 1. UDDS 주행 데이터 로드
% UDDS 주행 데이터를 로드합니다.
load('G:\공유 드라이브\BSL_Data3\Driving cycles\03-21-17_00.29 25degC_UDDS_Pan18650PF.mat');
udds_current = meas.Current;  % 전류 데이터 (A)
udds_voltage = meas.Voltage;  % 전압 데이터 (V)
udds_time = meas.Time;        % 시간 데이터 (s)

%% 2. SOC-OCV 데이터 로드
% SOC-OCV 데이터를 로드합니다.
load('soc_ocv.mat', 'soc_ocv');
soc_values = soc_ocv(:, 1);  % SOC 값
ocv_values = soc_ocv(:, 2);  % OCV 값

%% 3. SOC 계산 (벡터화된 방식으로 변경)
% 초기 SOC 설정
SOC_initial = 0.9901;  % 초기 SOC를 설정합니다.

% 배터리 용량 (Ah를 Coulomb로 변환)
Q_battery = 2.9 * 3600;  % 배터리 용량 (2.9Ah)

% 시간 간격 dt 계산
delta_t = [0; diff(udds_time)];  % in seconds

% SOC 계산 using cumulative sum
integrated_current = cumsum(udds_current .* delta_t);  % Coulombs
udds_SOC = SOC_initial + integrated_current / Q_battery;

% SOC를 [0,1] 범위로 제한
udds_SOC = min(max(udds_SOC, 0), 1);

%% 4. UDDS 트립 시작점과 끝점 탐지 (Sequential Approach)
% 초기 설정
initial_trip_start_time = 0;          % 첫 트립 시작 시간 (0초)
next_trip_duration = 1370;            % 다음 트립까지의 예상 시간 (초)
current_at_zero = udds_current(1);    % 시간 0에서의 전류
current_threshold = 0.001;            % 전류 유사성 임계값 (A)
time_window = 10;                     % 목표 시간 주변의 시간 창 (초)
total_time = udds_time(end);          % 총 시간

% 트립 시작 인덱스를 저장할 배열 초기화 (첫 트립 시작점 포함)
trip_start_indices = 1; % 첫 트립 시작점 (인덱스 1)

% 현재 탐색할 트립 시작 시간
current_trip_start_time = initial_trip_start_time;

% 트립 탐색 루프
while true
    % 다음 트립 시작 시간 예상
    t_target = current_trip_start_time + next_trip_duration;

    % t_target이 총 시간을 초과하면 루프 종료
    if t_target > total_time
        break;
    end

    % 시간 창 내의 인덱스 찾기
    indices_in_window = find(abs(udds_time - t_target) <= time_window);

    % 해당 인덱스 중 전류가 시간 0의 전류와 유사한 인덱스 찾기
    indices_with_similar_current = indices_in_window(abs(udds_current(indices_in_window) - current_at_zero) <= current_threshold);

    if ~isempty(indices_with_similar_current)
        % 목표 시간에 가장 가까운 인덱스 선택
        [~, idx_closest_time] = min(abs(udds_time(indices_with_similar_current) - t_target));
        index = indices_with_similar_current(idx_closest_time);

        % 인덱스 저장
        trip_start_indices = [trip_start_indices; index];

        % 다음 탐색을 위해 현재 트립 시작 시간 업데이트
        current_trip_start_time = udds_time(index);
    else
        % 유사한 전류를 찾지 못한 경우, 다음 트립 탐색을 종료
        break;
    end
end

% 트립 시작점을 중복 없이 정렬
trip_start_indices = unique(trip_start_indices, 'sorted');

%% 5. udds_data 구조체 배열 생성
% 각 트립의 데이터를 동일한 필드 이름을 가진 구조체 배열로 저장
num_trips = length(trip_start_indices);

% udds_data 구조체 배열 초기화
udds_data = struct('V', {}, 'I', {}, 't', {}, 'SOC', {});

for i = 1:num_trips
    if i < num_trips
        start_idx = trip_start_indices(i);
        end_idx = trip_start_indices(i+1) - 1;
    else
        start_idx = trip_start_indices(i);
        end_idx = length(udds_time);
    end

    % 각 트립의 데이터를 구조체 배열의 요소로 저장
    udds_data(i).V = udds_voltage(start_idx:end_idx);
    udds_data(i).I = udds_current(start_idx:end_idx);
    udds_data(i).Time_duration = udds_time(start_idx:end_idx);
    udds_data(i).t = udds_time(start_idx:end_idx) - udds_time(start_idx);
    udds_data(i).SOC = udds_SOC(start_idx:end_idx);
end

%% 6. 찾은 트립 시작 시간을 fprintf로 출력
fprintf('Detected Trip Start Times:\n');
fprintf('--------------------------\n');
for i = 1:length(trip_start_indices)
    trip_num = i; % 트립 번호 (Trip 1, Trip 2, ...)
    time_sec = udds_time(trip_start_indices(i)); % 트립 시작 시간 (초)

    % 마지막 트립의 경우 'Trip N.xx'와 같이 특별한 이름을 지정할 수 있습니다.
    if i == length(trip_start_indices)
        fprintf('Trip %d start time: %.2f s (Trip %.2f)\n', trip_num, time_sec, trip_num - 1 + 0.38);
    else
        fprintf('Trip %d start time: %.2f s\n', trip_num, time_sec);
    end
end

%% 7. udds_data 구조체 내용 확인
% 구조체 배열의 각 요소에 접근하여 데이터를 확인할 수 있습니다.
disp('udds_data 구조체 배열의 내용:');
disp('---------------------------');

for i = 1:num_trips
    fprintf('Trip %d:\n', i);
    fprintf('  Time length: %d samples\n', length(udds_data(i).t));
    fprintf('  Voltage length: %d samples\n', length(udds_data(i).V));
    fprintf('  Current length: %d samples\n', length(udds_data(i).I));
    fprintf('  SOC length: %d samples\n', length(udds_data(i).SOC));
end

%% 8 & 9. Plot Current vs SOC and Current vs Voltage with Trip Boundaries
% 두 개의 플롯을 하나의 코드 블록에서 생성합니다.

% --------------------------- Figure 1 ---------------------------
figure(1);
hold on;

% Plot Current on the left y-axis
yyaxis left
h_current = plot(udds_time, udds_current, 'Color', c_mat(1, :), 'LineWidth', 2, 'DisplayName', 'Current (A)');
ylabel('Current (A)', 'FontSize', labelFontSize, 'Color', c_mat(1, :));
ylim([min(udds_current)-0.5, max(udds_current)+0.5]);

% Plot SOC on the right y-axis
yyaxis right
h_soc = plot(udds_time, udds_SOC, 'Color', c_mat(2, :), 'LineWidth', 2, 'DisplayName', 'SOC');
ylabel('SOC', 'FontSize', labelFontSize, 'Color', c_mat(2, :));
ylim([min(udds_SOC)-0.05, max(udds_SOC)+0.05]);

% Initialize arrays to store handles for trip lines
h_trip_lines = [];

% Plot trip boundaries
for i = 1:length(trip_start_indices)
    x = udds_time(trip_start_indices(i));

    % Plot vertical dashed line on left y-axis (검은색으로 변경, 마커 제거)
    yyaxis left
    h_line = plot([x x], ylim, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1, 'Marker', 'none');
    h_trip_lines(end+1) = h_line; 

    % Only add trip number labels for trips 1 to N-1
    if i < length(trip_start_indices)
        midpoint = (udds_time(trip_start_indices(i)) + udds_time(trip_start_indices(i+1))) / 2;

        label = sprintf('Trip %d', i);

        % Add text label above the plot
        text(midpoint, max(udds_current)+0.3, label, 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', 'FontWeight', 'bold', 'FontSize', labelFontSize);
    end
end

% Reset to left y-axis for consistency
yyaxis left

% Create dummy plots for legend entries not directly tied to a single plot (마커 제거)
h_trip_line_dummy = plot(NaN, NaN, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1, 'Marker', 'none', 'DisplayName', 'Trip Lines');

% Combine all legend handles
legend_handles = [h_current, h_soc, h_trip_line_dummy];

% Add legend to the plot
legend(legend_handles, 'Location', 'best', 'FontSize', legendFontSize);

% Finalize plot settings
xlabel('Time (s)', 'FontSize', labelFontSize, 'Color', 'k');
title('UDDS Current and SOC Profile with Trip Boundaries', 'FontSize', titleFontSize);
grid off;
set(gca, 'FontSize', axisFontSize);
hold off;

% --------------------------- Figure 2 ---------------------------
figure(2);
hold on;

% Plot Current on the left y-axis
yyaxis left
h_current_fig2 = plot(udds_time, udds_current, 'Color', c_mat(1, :), 'LineWidth', 2, 'DisplayName', 'Current (A)');
ylabel('Current (A)', 'FontSize', labelFontSize, 'Color', c_mat(1, :));
ylim([min(udds_current)-0.5, max(udds_current)+0.5]);

% Plot Voltage on the right y-axis
yyaxis right
h_voltage_fig2 = plot(udds_time, udds_voltage, 'Color', c_mat(2, :), 'LineWidth', 2, 'DisplayName', 'Voltage (V)');
ylabel('Voltage (V)', 'FontSize', labelFontSize, 'Color', c_mat(2, :));
ylim([min(udds_voltage)-0.5, max(udds_voltage)+0.5]);

% Initialize arrays to store handles for trip lines
h_trip_lines_fig2 = [];

% Plot trip boundaries
for i = 1:length(trip_start_indices)
    x = udds_time(trip_start_indices(i));

    % Plot vertical dashed line on left y-axis (검은색으로 변경, 마커 제거)
    yyaxis left
    h_line_fig2 = plot([x x], ylim, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1, 'Marker', 'none');
    h_trip_lines_fig2(end+1) = h_line_fig2; 

    % Only add trip number labels for trips 1 to N-1
    if i < length(trip_start_indices)
        midpoint = (udds_time(trip_start_indices(i)) + udds_time(trip_start_indices(i+1))) / 2;

        label = sprintf('Trip %d', i);

        % Add text label above the plot
        text(midpoint, max(udds_current)+0.3, label, 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', 'FontWeight', 'bold', 'FontSize', labelFontSize);
    end
end

% Reset to left y-axis for consistency
yyaxis left

% Create dummy plots for legend entries not directly tied to a single plot (마커 제거)
h_trip_line_dummy_fig2 = plot(NaN, NaN, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1, 'Marker', 'none', 'DisplayName', 'Trip Lines');

% Combine all legend handles
legend_handles_fig2 = [h_current_fig2, h_voltage_fig2, h_trip_line_dummy_fig2];

% Add legend to the plot
legend(legend_handles_fig2, 'Location', 'best', 'FontSize', legendFontSize);

% Finalize plot settings
xlabel('Time (s)', 'FontSize', labelFontSize, 'Color', 'k');
title('UDDS Current and Voltage Profile', 'FontSize', titleFontSize);
grid off;
set(gca, 'FontSize', axisFontSize);
hold off;
