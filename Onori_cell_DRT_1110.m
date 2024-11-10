% main_script.m

clc; clear; close all;

%% 0. 폰트 크기 및 색상 매트릭스 설정
% Font size settings
axisFontSize = 14;      % 축의 숫자 크기
titleFontSize = 16;     % 제목의 폰트 크기
legendFontSize = 12;    % 범례의 폰트 크기
labelFontSize = 14;     % xlabel 및 ylabel의 폰트 크기

% Color matrix 설정
c_mat = lines(9);  % 9개의 고유한 색상 정의

%% 1. 셀 및 사이클 설정
cell_list = {'W3', 'W4', 'W5', 'W7', 'W8', 'W9', 'W10', 'G1', 'V4', 'V5'};  % 셀 이름 목록
num_cells = length(cell_list);
num_cycles = 14;  % 사이클 수

% 결과 저장을 위한 변수 초기화
num_combinations = num_cells * num_cycles;
Results = cell(num_cycles, num_cells );
features_all = zeros(num_combinations, 18);  % 140 x 18 크기의 배열
capacity_all = zeros(num_combinations, 1);  % 각 조합에 대한 용량 저장
cell_column = cell(num_combinations, 1);  % 셀 이름 저장
cycle_column = zeros(num_combinations, 1);  % 사이클 번호 저장
combination_idx = 1;  % 조합 인덱스 초기화

%% 2. DRT 추정에 필요한 파라미터 설정
n = 401;  % 이산 요소의 개수
tau_min = 0.1;     % 최소 시간 상수 (초)
tau_max = 2600;    % 최대 시간 상수 (초)

% Theta 및 tau 값 계산
theta_min = log(tau_min);
theta_max = log(tau_max);
theta_discrete = linspace(theta_min, theta_max, n);
tau_discrete = exp(theta_discrete);

% Delta theta 계산
delta_theta = theta_discrete(2) - theta_discrete(1);

% 정규화 파라미터
lambda = 8.29e-4;  % 최적화된 람다 값 (필요에 따라 조정 가능)

% Gamma에 대한 1차 차분 행렬 L_gamma 생성
L_gamma = zeros(n-1, n);
for i = 1:n-1
    L_gamma(i, i) = -1;
    L_gamma(i, i+1) = 1;
end

% R0에 대한 정규화를 피하기 위해 L_aug 생성
L_aug = [L_gamma, zeros(n-1, 1)];

%% 3. 셀과 사이클에 대한 루프
for cell_idx = 1:num_cells
    cell_name = cell_list{cell_idx};
    for cycle_num = 1:num_cycles
        fprintf('Processing Cell %s, Cycle %d...\n', cell_name, cycle_num);
        
        % 데이터 파일명 생성 (예시)
        trips_filename = sprintf('processed_trips_%s_cycle%d.mat', cell_name, cycle_num);
        rpt_filename = sprintf('RPT_%d_soc_ocv_cap_%s.mat', cycle_num, cell_name);
        
        % 데이터 로드
        if exist(trips_filename, 'file') && exist(rpt_filename, 'file')
            load(trips_filename, 'trips');  % 'trips' 구조체 로드
            load(rpt_filename, 'soc_ocv_cap');  % OCV-SOC 데이터 로드
        else
            warning('Data files for Cell %s, Cycle %d not found. Skipping...', cell_name, cycle_num);
            continue;  % 다음 조합으로 넘어감
        end
        
        % SOC와 OCV 값 추출
        soc_values = soc_ocv_cap(:, 1);  % SOC 값 (0 ~ 1)
        ocv_values = soc_ocv_cap(:, 2);  % OCV 값 (V)
        
        % 배터리 용량 추출 (Capacity 열의 최대값)
        Q_batt = max(soc_ocv_cap(:, 3));  % Ah 단위
        
        % 각 트립에 대한 DRT 추정
        num_trips = length(trips);
        gamma_est_all = zeros(num_trips, n);
        R0_est_all = zeros(num_trips, 1);
        soc_start_all = zeros(num_trips, 1);  % 각 트립의 시작 SOC 저장
        
        SOC0 = 0.8;  % 전체 시뮬레이션의 초기 SOC 설정 (필요에 따라 조정)
        
        for s = 1:num_trips
            fprintf('Processing Trip %d/%d...\n', s, num_trips);
            
            % 현재 트립의 데이터 추출
            t = trips(s).time_reset;
            ik = trips(s).I;
            V_sd = trips(s).V;
            
            % 시간 간격 계산
            delta_t = [0; diff(t)];  % 시간 간격 계산 (초)
            delta_t(1) = delta_t(2);  % 첫 번째 값이 0이면 두 번째 값으로 대체
         
            % SOC 계산 (전류 적분)
            Ah_consumed = cumsum(ik .* delta_t) / 3600;  % Ah로 변환
            SOC = SOC0 + Ah_consumed / Q_batt;  % SOC 계산
        
            % 각 트립의 시작 SOC 저장
            soc_start_all(s) = SOC(1);
        
            % OCV 계산
            ocv_over_time = interp1(soc_values, ocv_values, SOC, 'linear', 'extrap');
            
            % DRT 해 계산 (함수 호출)
            [gamma_est, R0_est, V_est] = compute_drt(t, ik, V_sd, ocv_over_time, delta_t, tau_discrete, delta_theta, lambda, L_aug, n);
            
            % 추정값 저장
            gamma_est_all(s, :) = gamma_est';
            R0_est_all(s) = R0_est;
            
            % 다음 트립을 위한 SOC0 업데이트
            SOC0 = SOC(end);  % 현재 트립의 마지막 SOC로 업데이트
        end
        
        % 각 조합에 대한 특징 추출 및 저장
        % SOC 구간 정의 (low, mid, high)
        soc_ranges = [0, 0.33; 0.33, 0.66; 0.66, 1];  % 예시로 3등분
        
        features_combination = zeros(1, 18);  % 현재 조합에 대한 특징 벡터
        
        for soc_idx = 1:size(soc_ranges, 1)
            soc_min = soc_ranges(soc_idx, 1);
            soc_max = soc_ranges(soc_idx, 2);
            
            % 해당 SOC 구간의 gamma_est 선택
            gamma_soc = gamma_est_all(soc_start_all >= soc_min & soc_start_all < soc_max, :);
            
            if isempty(gamma_soc)
                % 해당 SOC 구간에 데이터가 없을 경우, 0으로 채움
                continue;
            end
            
            % gamma_soc의 평균 계산
            gamma_mean = mean(gamma_soc, 1);
            
            % 피크 추출
            [peaks, locs, widths] = findpeaks(gamma_mean, theta_discrete);
            
            % 상위 3개의 피크 선택 (높이 순)
            [sorted_peaks, idx] = sort(peaks, 'descend');
            num_peaks = min(3, length(sorted_peaks));
            top_peaks = sorted_peaks(1:num_peaks);
            top_widths = widths(idx(1:num_peaks));
            
            % 특징 벡터에 저장 (높이와 너비)
            feature_idx = (soc_idx - 1) * 6 + 1;  % 각 SOC 구간마다 6개의 특징 (3개의 피크 * 높이, 너비)
            features_combination(feature_idx:feature_idx+num_peaks*2-1) = [top_peaks; top_widths];
        end
        
        % 용량 데이터 저장
        capacity_all(combination_idx) = Q_batt;
        
        % 특징 벡터 저장
        features_all(combination_idx, :) = features_combination;
        
        % 셀 이름과 사이클 번호 저장
        cell_column{combination_idx} = cell_name;
        cycle_column(combination_idx) = cycle_num;
        
        % 조합 인덱스 증가
        combination_idx = combination_idx + 1;
    end
end

%% 4. 결과 테이블 생성 및 저장
% 테이블로 변환
result_table = array2table(features_all);
% 변수 이름 설정
feature_names = {};
soc_labels = {'Low', 'Mid', 'High'};
for soc_idx = 1:3
    for peak_idx = 1:3
        feature_names{end+1} = sprintf('%s_Peak%d_Height', soc_labels{soc_idx}, peak_idx);
        feature_names{end+1} = sprintf('%s_Peak%d_Width', soc_labels{soc_idx}, peak_idx);
    end
end
result_table.Properties.VariableNames = feature_names;

% 용량, 셀 이름, 사이클 번호 추가
result_table.Capacity = capacity_all;
result_table.Cell = cell_column;
result_table.Cycle = cycle_column;

% % 결과 테이블 저장
% writetable(result_table, 'DRT_Features_Capacity.csv');
% fprintf('DRT 특징 및 용량 데이터가 DRT_Features_Capacity.csv 파일로 저장되었습니다.\n');
