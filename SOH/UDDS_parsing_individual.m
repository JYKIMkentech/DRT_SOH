clc; clear; close all;

% 데이터 로드
load('G:\공유 드라이브\BSL_Onori\Cycling_tests\Processed_1\W10.mat');
load('RPT_All_soc_ocv_cap.mat');

col_cell_label = {'W3','W4','W5','W7','W8','W9','W10','G1','V4','V5'};

I_1C = 4.99; % [A]

time = t_full_vec_M1_NMC25degC;
curr = I_full_vec_M1_NMC25degC;                                            
volt = V_full_vec_M1_NMC25degC;
step_arbin = Step_Index_full_vec_M1_NMC25degC;

% data1 구조체 생성
data1.V = volt;
data1.I = curr;
data1.t = time;
data1.step = step_arbin;

% step 변경 지점 찾기
change_indices = [1; find(diff(step_arbin) ~= 0) + 1; length(step_arbin) + 1];
num_segments = length(change_indices) - 1;

% data_line 구조체 템플릿 생성
data_line = struct('V',[],'I',[],'t',[],'indx',[],'step',[]);

% data 구조체 배열 생성
data = repmat(data_line, num_segments, 1);

% 데이터 파싱 및 구조체 채우기
for i = 1:num_segments
    idx_start = change_indices(i);
    idx_end = change_indices(i+1) - 1;
    range = idx_start:idx_end;
    
    data(i).V = data1.V(range);
    data(i).I = data1.I(range);
    data(i).t = data1.t(range);
    data(i).indx = range;
    data(i).step = data1.step(idx_start);  % 해당 구간의 step 번호
end

% Aging Cycle 번호 고정 (14번째)
aging_cycle = 14;

% 해당 cycle에 해당하는 첫 번째 데이터 찾기
indices_with_step = find([data.step] == aging_cycle);

if isempty(indices_with_step)
    error('14번째 Aging cycle에 해당하는 데이터가 없습니다.');
else
    first_index = indices_with_step(1);
    Onori_Cycling_UDDS = data(first_index);
    fprintf('Onori_Cycling_%d_UDDS 데이터를 생성했습니다.\n', aging_cycle);
end

% UDDS 주기 파싱
udds_duration = 2600;          % UDDS 주기 시간 (초)
time_tolerance = 0.5;          % 시간 허용 오차 (초)
current_tolerance = 0.03;      % 전류 허용 오차

% 첫 번째 시간과 전류값
t_start = Onori_Cycling_UDDS.t(1);
I_start = Onori_Cycling_UDDS.I(1);

% 전체 시간 길이
total_time = Onori_Cycling_UDDS.t(end) - Onori_Cycling_UDDS.t(1);

% 초기 설정
trip_start_indices = 1;        % 첫 번째 trip은 항상 시작 인덱스 1
last_trip_time = t_start;      % 마지막 trip의 시작 시간

% 두 번째 trip부터 사용할 기준 전류값
reference_current = 0.001;    % 기준 전류값

% while 루프를 사용하여 trip 시작 지점 찾기
trip_number = 1;  % 현재 trip 번호
while true
    % 예상되는 다음 trip의 시작 시간
    expected_time = last_trip_time + udds_duration;
    
    % 예상 시간이 전체 시간 범위를 벗어나면 종료
    if expected_time > Onori_Cycling_UDDS.t(end)
        break;
    end
    
    % 허용 오차 범위 내의 인덱스 찾기
    time_diff = abs(Onori_Cycling_UDDS.t - expected_time);
    indices_in_tolerance = find(time_diff <= time_tolerance);
    
    if ~isempty(indices_in_tolerance)
        % 현재 trip 번호에 따라 기준 전류값 설정
        if trip_number == 1
            % 첫 번째 trip 이후부터는 reference_current 사용
            current_ref = I_start;
        else
            current_ref = reference_current;
        end
        
        % 전류 값이 기준 전류값과 유사한 인덱스 찾기
        I_in_tolerance = Onori_Cycling_UDDS.I(indices_in_tolerance);
        current_diff = abs(I_in_tolerance - current_ref);
        valid_indices = indices_in_tolerance(current_diff <= current_tolerance);
        
        if ~isempty(valid_indices)
            % 유효한 인덱스 중 전류 값이 최대인 인덱스 선택
            [~, max_current_idx] = max(Onori_Cycling_UDDS.I(valid_indices));
            idx = valid_indices(max_current_idx);
            
            % 다음 trip 시작 인덱스로 추가
            trip_start_indices(end+1) = idx;
            last_trip_time = Onori_Cycling_UDDS.t(idx);  % 마지막 trip 시간 업데이트
            trip_number = trip_number + 1;  % trip 번호 증가
        else
            % 조건을 만족하는 인덱스가 없으면 예상 시간을 업데이트하여 다음으로 진행
            last_trip_time = expected_time;
        end
    else
        % 허용 오차 범위 내에 인덱스가 없을 때, 예상 시간을 업데이트하여 다음으로 진행
        last_trip_time = expected_time;
    end
end

% 중복 제거 및 정렬 (필요 시)
trip_start_indices = unique(trip_start_indices, 'stable');

% 각 trip 데이터 분할 및 time_reset 필드 추가
num_trips_original = length(trip_start_indices);
trips_original = struct('t', [], 'I', [], 'V', [], 'time_reset', []);

for i = 1:num_trips_original
    idx_start = trip_start_indices(i);
    if i < num_trips_original
        idx_end = trip_start_indices(i+1) - 1;
    else
        idx_end = length(Onori_Cycling_UDDS.t);
    end
    range = idx_start:idx_end;
    
    trips_original(i).t = Onori_Cycling_UDDS.t(range);
    trips_original(i).I = Onori_Cycling_UDDS.I(range);
    trips_original(i).V = Onori_Cycling_UDDS.V(range);
    trips_original(i).time_reset = trips_original(i).t - trips_original(i).t(1);  % 시간 초기화
end

%% data(5)와 data(6)을 trips 구조체 형식에 맞게 변환하여 맨 앞에 추가
num_new_trips = 2;
new_trips = struct('t', [], 'I', [], 'V', [], 'time_reset', []);
for i = 1:num_new_trips
    data_idx = i + 4; % data(5)와 data(6)
    new_trips(i).t = data(data_idx).t;
    new_trips(i).I = data(data_idx).I;
    new_trips(i).V = data(data_idx).V;
    new_trips(i).time_reset = new_trips(i).t - new_trips(i).t(1); % 시간 초기화
end

% trips 구조체 배열 생성 (new_trips + trips_original)
trips = [new_trips, trips_original];

% 전체 trip 수 업데이트
num_trips = length(trips);

% 전체 데이터를 위해 Onori_Cycling_Total 생성 (data(5), data(6), Onori_Cycling_UDDS 합침)
Onori_Cycling_Total.t = [new_trips(1).t; new_trips(2).t; Onori_Cycling_UDDS.t];
Onori_Cycling_Total.I = [new_trips(1).I; new_trips(2).I; Onori_Cycling_UDDS.I];
Onori_Cycling_Total.V = [new_trips(1).V; new_trips(2).V; Onori_Cycling_UDDS.V];

% trip_start_indices 업데이트 (전체 데이터 기준으로)
trip_start_indices_total = zeros(1, num_trips);
trip_start_indices_total(1) = 1; % 첫 번째 trip 시작 인덱스

% 각 trip의 시작 인덱스 계산
current_index = 1;
for i = 1:num_trips
    trip_length = length(trips(i).t);
    trip_start_indices_total(i) = current_index;
    current_index = current_index + trip_length;
end

% t_start 업데이트 (전체 데이터의 시작 시간)
t_start = Onori_Cycling_Total.t(1);

% 전류 대비 시간 플롯 및 trip 시작점 표시
figure;
plot(Onori_Cycling_Total.t - t_start, Onori_Cycling_Total.I, 'b', 'DisplayName', '전류 (I)');  % 전체 데이터는 파란색 선으로
hold on;

% 각 trip의 시작 지점 표시 및 레이블 추가
for i = 3:num_trips
    trip_time = Onori_Cycling_Total.t(trip_start_indices_total(i)) - t_start;
    % 빨간색 원 표시
    plot(trip_time, Onori_Cycling_Total.I(trip_start_indices_total(i)), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
    % 수직 검은색 점선 추가
    if exist('xline', 'file') == 2  
        xline(trip_time, 'k--', 'LineWidth', 1.5);
    else
        % 이전 버전 호환을 위해 대체 방법 사용
        plot([trip_time, trip_time], ylim, 'k--', 'LineWidth', 1.5);
    end
    % 레이블 추가 (상단)
    text(trip_time + 15, max(Onori_Cycling_Total.I) * 0.95, sprintf('trip%d', i-2), ...
        'FontSize', 12, 'Color', 'k', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
end

xlabel('시간 (초)');
ylabel('전류 (A)');
title(sprintf('Onori Cycling %d UDDS - 전류 대비 시간 그래프', aging_cycle));
grid on;
hold off;

% trips 구조체 확인 (옵션)
disp('각 trip의 time_reset 필드:');
for i = 1:num_trips
    fprintf('trip%d: 시작 시간 = %.2f 초, 종료 시간 = %.2f 초\n', ...
        i, trips(i).time_reset(1), trips(i).time_reset(end));
end

%% -------------- SOC 계산 추가 --------------

%% 1. 트립 데이터 로드
% 'Trips' 구조체를 로드합니다.
load('G:\공유 드라이브\BSL_Onori\Cycling_tests\Trips_Aging_1_W10.mat');  % 'Trips' 구조체를 로드합니다.

col_cell_label = {'W3','W4','W5','W7','W8','W9','W10','G1','V4','V5'};

%% 2. SOC-OCV 데이터 로드
% 'soc_ocv_cap' 데이터를 로드합니다.
load('RPT_All_soc_ocv_cap.mat', 'soc_ocv_cap');

% load('soc_ocv.mat', 'soc_ocv');
% soc_values = soc_ocv(:, 1);  % SOC 값
% ocv_values = soc_ocv(:, 2);  % OCV 값

% % SOC와 OCV 값 추출
soc_values = soc_ocv_cap{1,7}(:, 1);  % SOC 값 (0 ~ 1)
ocv_values = soc_ocv_cap{1,7}(:, 2);  % OCV 값 (V)

% 배터리 용량 추출 (Capacity 열의 최대값)
Q_batt = max(soc_ocv_cap{1,7}(:, 3));  % Ah 단위

% 추출할 data 인덱스
data_indices = [8, 9, 10, 11];

% 빈 배열 초기화
V_combined = [];
I_combined = [];
t_combined = [];

% 데이터를 순차적으로 결합
for idx = data_indices
    V_combined = vertcat(V_combined, data(idx).V);
    I_combined = vertcat(I_combined, data(idx).I);
    t_combined = vertcat(t_combined, data(idx).t);
end

% 시간 정렬을 위해 t_combined를 기준으로 정렬 (필요 시)
% [t_combined, sort_idx] = sort(t_combined);
% V_combined = V_combined(sort_idx);
% I_combined = I_combined(sort_idx);

% 배터리 용량 (예시: 4.8 A·h)
C =  I_1C; % Q_batt;  % I_1C = 4.8 A

% 시작 SOC 설정
SOC_start = 0.179;

% 시간 간격 계산 (초 단위 -> 시간 단위로 변환)
t_hours = t_combined / 3600;  % 초를 시간으로 변환

% 전류 적분 (A·h)
I_integral = cumtrapz(t_hours, I_combined);

% SOC 계산
SOC = SOC_start + (I_integral / C);  % 충전 시 SOC 증가, 방전 시 SOC 감소

% SOC 플롯 (전류, 전압, SOC를 함께 서브플롯으로 표시)
figure;

% 서브플롯 1: 전류 (I) vs 시간
subplot(3,1,1);
plot(t_combined, I_combined, 'b', 'LineWidth', 1.5);
xlabel('Time (sec)');
ylabel('Current (A)');
title('Current (I) vs Time');
grid on;

% 서브플롯 2: 전압 (V) vs 시간
subplot(3,1,2);
plot(t_combined, V_combined, 'g', 'LineWidth', 1.5);
xlabel('Time (sec)');
ylabel('Voltage (V)');
title('Voltage (V) vs Time');
grid on;

% 서브플롯 3: SOC vs 시간
subplot(3,1,3);
plot(t_combined, SOC, 'r', 'LineWidth', 2);
xlabel('time (sec)');
ylabel('SOC');
title('SOC vs Time');
grid on;

% 전체 플롯 레이아웃 조정
sgtitle('Current,Voltage, SOC vs time');

% SOC 데이터 저장 (옵션)
% Onori_Cycling_Total_SOC.V = V_combined;
% Onori_Cycling_Total_SOC.I = I_combined;
% Onori_Cycling_Total_SOC.t = t_combined;
% Onori_Cycling_Total_SOC.SOC = SOC;
