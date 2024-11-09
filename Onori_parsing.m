clc; clear; close all;

% 데이터 로드
load('G:\공유 드라이브\BSL_Onori\Cycling_tests\Processed_10\G1.mat');


I_1C = 4.8; % [A]

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
plot(Onori_Cycling_Total.t - t_start, Onori_Cycling_Total.I, 'b');  % 전체 데이터는 파란색 선으로
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



% trips 구조체 저장
save_filename = sprintf('processed_10_trips_cycle%d_with_data5_6.mat', aging_cycle);
save(save_filename, 'trips');
fprintf('trips 구조체가 %s 파일로 저장되었습니다.\n', save_filename);
