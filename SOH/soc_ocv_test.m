clc; clear; close all;

% 데이터 로드
load('G:\공유 드라이브\BSL_Onori\diagnostic_tests\_processed_mat\capacity_test.mat');

% cap의 사이즈 얻기
[total_cycles, total_cells] = size(cap);

% soc_ocv_cap을 cap과 같은 사이즈의 cell array로 초기화
soc_ocv_cap = cell(total_cycles, total_cells);

% 각 사이클과 셀에 대해 반복
for cycle_num = 1:total_cycles
    for cell_num = 1:total_cells
        
        % 용량, 전류, 시간, 셀 전압 데이터 추출
        cap_data = cap{cycle_num, cell_num};       % 누적 용량 데이터 (Ah)
        curr_data = curr{cycle_num, cell_num};     % 전류 데이터 (A)
        time_data = time{cycle_num, cell_num};     % 시간 데이터 (초)
        vcell_data = vcell{cycle_num, cell_num};   % 셀 전압 데이터 (V)
        
        % 데이터가 비어있지 않은지 확인
        if ~isempty(cap_data) && ~isempty(vcell_data)
            % 최대 용량 계산 (현재 사이클에서의 실제 용량)
            Q_max = cap_data(end);  % Ah 단위
            
            % SOC 계산 (0에서 1 사이의 값)
            SOC = cap_data / Q_max;          % 누적 용량을 최대 용량으로 나누어 SOC 계산
            SOC = 1 - SOC;                   % SOC 방향 조정 (1에서 0으로 감소하도록)
            cap_data_adj = Q_max - cap_data; % 누적 용량을 최대 용량에서 빼서 역전
            
            % SOC와 OCV, Capacity 데이터를 결합하여 저장
            soc_ocv_cap{cycle_num, cell_num} = [SOC, vcell_data, cap_data_adj];
        else
            % 데이터가 비어있을 경우 빈 배열로 저장
            soc_ocv_cap{cycle_num, cell_num} = [];
        end
    end
end

% SOC-OCV-Capacity 테이블 저장
save_filename = 'RPT_All_soc_ocv_cap.mat';
save(save_filename, 'soc_ocv_cap');
fprintf('SOC-OCV-Capacity 테이블이 %s 파일로 저장되었습니다.\n', save_filename);

%% 특정 셀과 사이클에 대한 SOC vs OCV 플롯 생성

% 셀 라벨 정의
col_cell_label = {'W3','W4','W5','W7','W8','W9','W10','G1','V4','V5'};

% 분석할 셀 이름과 사이클 번호 지정
target_cell_label = 'G1';
target_cycle_num = 10;

% 셀 이름을 기반으로 셀 번호 찾기
cell_num = find(strcmp(col_cell_label, target_cell_label));

if isempty(cell_num)
    error('지정한 셀 이름(%s)이 col_cell_label에 존재하지 않습니다.', target_cell_label);
end

% 지정한 사이클과 셀의 soc_ocv_cap 데이터 추출
target_data = soc_ocv_cap{target_cycle_num, cell_num};

if isempty(target_data)
    error('사이클 %d, 셀 %s에 대한 데이터가 비어있습니다.', target_cycle_num, target_cell_label);
end

% SOC, OCV, Capacity 분리
SOC = target_data(:, 1);
OCV = target_data(:, 2);
Capacity = target_data(:, 3);

% SOC-OCV 관계 그래프 생성
figure;
plot(SOC, OCV, '-', 'LineWidth', 1.5, 'MarkerSize', 6);
xlabel('SOC', 'FontSize', 14);
ylabel('OCV (V)', 'FontSize', 14);
title(sprintf('SOC vs OCV for Cycle %d, Cell %s', target_cycle_num, target_cell_label), 'FontSize', 16);
grid on;

%% save

save('RPT_All_soc_ocv_cap.mat', 'soc_ocv_cap');
