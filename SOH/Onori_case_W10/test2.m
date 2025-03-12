clc; clear; close all;

%% (0) 로드 경로 지정 및 파일 경로에서 제목 문자열 생성
loadPath = 'G:\공유 드라이브\BSL_Onori\Cycling_tests\Processed_9\G1.mat';
[folderPath, fileName, ext] = fileparts(loadPath);
[~, folderName] = fileparts(folderPath);  % 예: 'Processed_14'
plotTitle = sprintf('%s_%s_UDDS', folderName, fileName);  % 예: 'Processed_14_W10_UDDS'

%% (1) 데이터 로드 및 전처리
load(loadPath);

I_1C = 4.6; % [A]

time       = t_full_vec_M1_NMC25degC;
curr       = I_full_vec_M1_NMC25degC;
volt       = V_full_vec_M1_NMC25degC;
step_arbin = Step_Index_full_vec_M1_NMC25degC;

% data1 구조체 생성
data1.V    = volt;
data1.I    = curr;
data1.t    = time;
data1.step = step_arbin;

% step 변경 지점 찾기
change_indices = [1; find(diff(step_arbin) ~= 0) + 1; length(step_arbin) + 1];
num_segments   = length(change_indices) - 1;

% data_line 구조체 템플릿
data_line = struct('V',[],'I',[],'t',[],'indx',[],'step',[]);

% data 구조체 배열 생성 및 채우기
data = repmat(data_line, num_segments, 1);
for i = 1:num_segments
    idx_start = change_indices(i);
    idx_end   = change_indices(i+1) - 1;
    range     = idx_start:idx_end;
    
    data(i).V    = data1.V(range);
    data(i).I    = data1.I(range);
    data(i).t    = data1.t(range);
    data(i).indx = range;
    data(i).step = data1.step(idx_start);
end

%% (2) 14번째 Aging Cycle 데이터 추출
aging_cycle       = 14;
indices_with_step = find([data.step] == aging_cycle);

if isempty(indices_with_step)
    error('14번째 Aging cycle에 해당하는 데이터가 없습니다.');
else
    first_index        = indices_with_step(1);
    Onori_Cycling_UDDS = data(first_index);
    fprintf('Onori_Cycling_%d_UDDS 데이터를 생성했습니다.\n', aging_cycle);
end

%% (3) data(5), data(6)을 Trip으로 만들어 맨 앞에 추가
num_new_trips = 2;
new_trips = struct('t', [], 'I', [], 'V', [], 'time_reset', []);
for i = 1:num_new_trips
    data_idx          = i + 4;  % 즉, data(5), data(6)
    new_trips(i).t    = data(data_idx).t;
    new_trips(i).I    = data(data_idx).I;
    new_trips(i).V    = data(data_idx).V;
    new_trips(i).time_reset = new_trips(i).t - new_trips(i).t(1);
end

%% (4) 전체 데이터 이어붙임: [data(5), data(6), Onori_Cycling_UDDS]
Onori_Cycling_Total.t = [new_trips(1).t; new_trips(2).t; Onori_Cycling_UDDS.t];
Onori_Cycling_Total.I = [new_trips(1).I; new_trips(2).I; Onori_Cycling_UDDS.I];
Onori_Cycling_Total.V = [new_trips(1).V; new_trips(2).V; Onori_Cycling_UDDS.V];

t_data = Onori_Cycling_Total.t;
I_data = Onori_Cycling_Total.I;
V_data = Onori_Cycling_Total.V;

%% (5) "임시 Trip1" = -1.2A 이하로 내려갔다가 올라올 때까지
start_idx = find(I_data < -1.2, 1, 'first');
if isempty(start_idx)
    error('전류가 -1.2 A 이하로 내려간 지점(임시Trip1 시작)을 못 찾았습니다.');
end

trip1_end_idx_rel = find(I_data(start_idx:end) >= -1.2, 1, 'first');
if isempty(trip1_end_idx_rel)
    error('전류가 다시 -1.2 A 위로 올라오지 않음 => 임시Trip1 종료 불명');
end
trip1_end_idx = (start_idx - 1) + trip1_end_idx_rel;

%% (6) 나머지 구간: "홀수 번째 음의 피크" 기준
range_after_trip1 = trip1_end_idx : length(I_data);
I_after           = I_data(range_after_trip1);

minPeakHeight   = 4.0;   % => 실제 전류가 -4 A 이하인 골만 찾기
minPeakDistance = 200;   % => 골 사이 최소 간격
[negPeaks, local_peaks] = findpeaks(-I_after, ...
    'MinPeakHeight',   minPeakHeight, ...
    'MinPeakDistance', minPeakDistance);

peakLocs_afterTrip1 = local_peaks + (trip1_end_idx - 1);

% 홀수번째 피크만
all_peaks      = peakLocs_afterTrip1(:);
num_all_peaks  = length(all_peaks);
odd_indices    = 1:2:num_all_peaks;         
odd_peaksOnly  = all_peaks(odd_indices);

end_idx = length(I_data);

%% (7) 전체 경계 인덱스 (임시Trip1 포함)
boundary_indices = [start_idx; trip1_end_idx; odd_peaksOnly(:); end_idx];
boundary_indices = unique(boundary_indices, 'stable');

%% (8) "임시 Trip1" 빼기 (원하지 않는 구간 제거)
boundary_indices_noT1 = boundary_indices(2:end);  
% => 실제로는 boundary_indices(1)~(2)가 임시Trip1 범위
%    그걸 제외하고 Trip을 구성

% Trip 개수
num_trips = length(boundary_indices_noT1) - 1;

%% (9) Trip 구조체 (임시Trip1 제외)
trips = struct('t', [], 'I', [], 'V', [], 'time_reset', []);
trips = repmat(trips, num_trips, 1);

for i = 1:num_trips
    s_idx = boundary_indices_noT1(i);
    e_idx = boundary_indices_noT1(i+1);
    rng   = s_idx:e_idx;
    
    trips(i).t = t_data(rng);
    trips(i).I = I_data(rng);
    trips(i).V = V_data(rng);
    trips(i).time_reset = trips(i).t - trips(i).t(1);
end

%% (10) 기본 플롯 (전류 vs. 시간)
figure('Name','Trip1 removed + Vertical Lines');
plot(t_data, I_data, 'b');
hold on; grid on;
xlabel('시간 (초)');
ylabel('전류 (A)');
title(plotTitle, 'Interpreter', 'none');  % 파일 경로에서 추출한 제목 사용

%% (11) 각 Trip 경계점을 빨간 동그라미로 표시
for i = 1:num_trips
    b_ind = boundary_indices_noT1(i);
    plot(t_data(b_ind), I_data(b_ind), 'ro','MarkerSize',8,'LineWidth',2);
end

%% (12) 각 Trip 구간을 "수직 점선(xline)" + 오른쪽 라벨 (Trip1 라벨은 왼쪽 배치)
ylims = ylim;   % y축 범위 가져오기
for i = 1:num_trips
    x_val = t_data(boundary_indices_noT1(i));  
    % 수직 점선
    xline(x_val, 'k--', 'LineWidth', 1.5);
    
    if i == 1
        % Trip1 라벨은 점선 왼쪽에 배치 (예: x_val - 50)
        x_text = x_val - 50;   
        y_text = ylims(2)*0.8;
        text(x_text, y_text, sprintf('Trip%d', i), ...
            'HorizontalAlignment','right', ...
            'VerticalAlignment','middle', ...
            'FontSize', 10, 'Color','k', 'FontWeight','bold');
    else
        % 나머지 Trip은 점선 오른쪽에 배치 (예: x_val + 50)
        x_text = x_val + 50;   
        y_text = ylims(2)*0.8;
        text(x_text, y_text, sprintf('Trip%d', i), ...
            'HorizontalAlignment','left', ...
            'VerticalAlignment','middle', ...
            'FontSize', 10, 'Color','k', 'FontWeight','bold');
    end
end

hold off;

%% (13) Trip 정보 콘솔 출력
disp('===== Trip 구간 정보 (임시Trip1 제외) =====');
for i = 1:num_trips
    fprintf('Trip %d: 시작=%.2f s, 끝=%.2f s, 샘플개수=%d\n', ...
        i, trips(i).time_reset(1), trips(i).time_reset(end), length(trips(i).t));
end

