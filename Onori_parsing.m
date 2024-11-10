clc; clear; close all;

% 셀 라벨 정의
col_cell_label = {'W3','W4','W5','W7','W8','W9','W10','G1','V4','V5'};

% 데이터가 저장된 기본 폴더 경로
base_folder = 'G:\공유 드라이브\BSL_Onori\Cycling_tests';

% Processed_* 폴더 목록 가져오기
cycle_folders = dir(fullfile(base_folder, 'Processed_*'));
cycle_folders = cycle_folders([cycle_folders.isdir]); % 폴더만 선택

% 사이클 수 얻기
total_cycles = length(cycle_folders);

% 셀 수 얻기
total_cells = length(col_cell_label);

% All_Cycling_Trips_cell 초기화 (사이즈: [total_cycles x total_cells])
All_Cycling_Trips_cell = cell(total_cycles, total_cells);

% 각 사이클 폴더에 대해 반복
for cycle_idx = 1:total_cycles
    % 현재 사이클 폴더 이름 및 경로
    cycle_folder_name = cycle_folders(cycle_idx).name;
    cycle_folder_path = fullfile(base_folder, cycle_folder_name);
    
    % 현재 사이클 번호 추출 (폴더 이름에서 숫자 부분만 추출)
    tokens = regexp(cycle_folder_name, 'Processed_(\d+)', 'tokens');
    if isempty(tokens)
        warning('폴더 이름 %s에서 사이클 번호를 추출할 수 없습니다.', cycle_folder_name);
        continue;
    end
    cycle_num = str2double(tokens{1}{1});
    
    % 현재 사이클 폴더 내의 .mat 파일 목록 가져오기
    cell_files = dir(fullfile(cycle_folder_path, '*.mat'));
    
    % 각 셀 파일에 대해 반복
    for cell_file_idx = 1:length(cell_files)
        % 현재 셀 파일 이름 및 경로
        cell_file_name = cell_files(cell_file_idx).name;
        cell_file_path = fullfile(cycle_folder_path, cell_file_name);
        
        % 셀 이름 추출 (파일 이름에서 .mat 제거)
        [~, cell_name, ~] = fileparts(cell_file_name);
        
        % 셀 이름을 기반으로 인덱스 찾기
        cell_num = find(strcmp(col_cell_label, cell_name));
        
        if isempty(cell_num)
            warning('셀 이름 %s이 col_cell_label에 존재하지 않습니다.', cell_name);
            continue;
        end
        
        % 데이터 로드
        data_vars = load(cell_file_path);
        
        % 필요한 변수 추출 (변수 이름이 다를 수 있으므로 존재 여부 확인)
        if isfield(data_vars, 't_full_vec_M1_NMC25degC') && isfield(data_vars, 'I_full_vec_M1_NMC25degC') && isfield(data_vars, 'V_full_vec_M1_NMC25degC') && isfield(data_vars, 'Step_Index_full_vec_M1_NMC25degC')
            time = data_vars.t_full_vec_M1_NMC25degC;
            curr = data_vars.I_full_vec_M1_NMC25degC;
            volt = data_vars.V_full_vec_M1_NMC25degC;
            step_arbin = data_vars.Step_Index_full_vec_M1_NMC25degC;
        else
            warning('파일 %s에 필요한 변수가 없습니다.', cell_file_name);
            continue;
        end
        
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
        
        % 해당 사이클 번호 고정 (현재 사이클 번호 사용)
        aging_cycle = cycle_num;
        
        % 해당 cycle에 해당하는 첫 번째 데이터 찾기
        indices_with_step = find([data.step] == aging_cycle);
        
        if isempty(indices_with_step)
            warning('사이클 %d, 셀 %s에 대한 데이터가 없습니다.', aging_cycle, cell_name);
            continue;
        else
            first_index = indices_with_step(1);
            Onori_Cycling_UDDS = data(first_index);
            fprintf('사이클 %d, 셀 %s에 대한 데이터를 처리합니다.\n', aging_cycle, cell_name);
        end
        
        % UDDS 주기 파싱 (기존 코드 재사용)
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
        % data(5)와 data(6)이 존재하는지 확인
        num_new_trips = 0;
        new_trips = struct('t', [], 'I', [], 'V', [], 'time_reset', []);
        if length(data) >= 6
            num_new_trips = 2;
            for i = 1:num_new_trips
                data_idx = i + 4; % data(5)와 data(6)
                new_trips(i).t = data(data_idx).t;
                new_trips(i).I = data(data_idx).I;
                new_trips(i).V = data(data_idx).V;
                new_trips(i).time_reset = new_trips(i).t - new_trips(i).t(1); % 시간 초기화
            end
        end
        
        % trips 구조체 배열 생성 (new_trips + trips_original)
        trips = [new_trips, trips_original];
        
        % 결과를 All_Cycling_Trips_cell에 저장
        All_Cycling_Trips_cell{cycle_num, cell_num} = trips;
        
        % 진행 상황 출력
        fprintf('사이클 %d, 셀 %s에 대한 trips를 저장했습니다.\n', cycle_num, cell_name);
    end
end

% % 결과 저장 (선택 사항)
% save('All_Cycling_Trips_cell.mat', 'All_Cycling_Trips_cell');
% fprintf('All_Cycling_Trips_cell이 저장되었습니다.\n');

