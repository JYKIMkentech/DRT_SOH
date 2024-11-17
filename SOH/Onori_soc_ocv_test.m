clc; clear; close all;

% 데이터 로드
load('G:\공유 드라이브\BSL_Onori\diagnostic_tests\_processed_mat\capacity_test.mat');

% 10번째 사이클의 9번째 셀 데이터 추출
cycle_num = 14;  % 사이클 번호
cell_num = 7;    % 셀 번호

% 용량, 전류, 시간, 셀 전압 데이터 추출
cap_data = cap{cycle_num, cell_num};       % 누적 용량 데이터 (Ah)
curr_data = curr{cycle_num, cell_num};     % 전류 데이터 (A)
time_data = time{cycle_num, cell_num};     % 시간 데이터 (초)
vcell_data = vcell{cycle_num, cell_num};   % 셀 전압 데이터 (V)

% 최대 용량 계산 (현재 사이클에서의 실제 용량)
Q_max = cap_data(end);  % Ah 단위

% SOC 계산 (0에서 1 사이의 값)
SOC = cap_data / Q_max;  % 누적 용량을 최대 용량으로 나누어 SOC 계산
SOC = 1 - SOC;  % SOC 방향 조정 (1에서 0으로 감소하도록)
cap_data = cap_data(end) - cap_data;  % 누적 용량을 최대 용량에서 빼서 역전

% SOC와 OCV, Capacity 데이터를 결합하여 테이블 생성
soc_ocv_cap = [SOC, vcell_data, cap_data];

% 테이블 형식으로 변환 (선택 사항)
soc_ocv_table = array2table(soc_ocv_cap, 'VariableNames', {'SOC', 'OCV', 'Capacity'});

% 결과 확인
disp('SOC-OCV-Capacity Table:');
disp(soc_ocv_table);

% SOC-OCV 관계 그래프 생성 (선택 사항)
figure;
plot(soc_ocv_table.SOC, soc_ocv_table.OCV, '-', 'LineWidth', 1.5, 'MarkerSize', 6);
xlabel('SOC', 'FontSize', 14);
ylabel('OCV (V)', 'FontSize', 14);
title('SOC vs OCV for Cycle 10 Cell 9', 'FontSize', 16);
grid on;

% SOC-OCV-Capacity 테이블 저장
% 파일명에 'RPT_10' 포함
save_filename = sprintf('RPT_%d_soc_ocv_cap.mat', cycle_num);
save(save_filename, 'soc_ocv_cap', 'soc_ocv_table');
fprintf('SOC-OCV-Capacity 테이블이 %s 파일로 저장되었습니다.\n', save_filename);
