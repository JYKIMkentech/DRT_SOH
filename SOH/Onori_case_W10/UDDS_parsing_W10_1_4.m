clc; clear; close all;

%% 공통 경로 및 변수 설정
common_path = 'G:\공유 드라이브\BSL_Onori\Cycling_tests';
dataset_names = {'Processed_1', 'Processed_14'}; % 처리할 데이터셋 이름
cell_name = 'W10'; % 셀 이름
udds_duration = 2600; % UDDS 사이클 지속 시간 [초]

%% 데이터셋별로 처리
for dataset_idx = 1:length(dataset_names)
    dataset_name = dataset_names{dataset_idx};
    dataset_path = fullfile(common_path, dataset_name);
    
    %% 데이터 로드
    data_file = fullfile(dataset_path, [cell_name, '.mat']);
    if ~isfile(data_file)
        error('데이터 파일 %s이 존재하지 않습니다.', data_file);
    end
    load(data_file);
    
    % 변수 할당 (로드한 데이터의 변수 이름에 따라 조정 필요)
    time_var = who('-regexp', '^t_full_vec_.*');
    curr_var = who('-regexp', '^I_full_vec_.*');
    volt_var = who('-regexp', '^V_full_vec_.*');
    step_var = who('-regexp', '^Step_Index_full_vec_.*');
    
    if isempty(time_var) || isempty(curr_var) || isempty(volt_var) || isempty(step_var)
        error('필요한 변수 중 하나 이상을 찾을 수 없습니다.');
    end
    
    time = eval(time_var{1});
    curr = eval(curr_var{1});
    volt = eval(volt_var{1});
    step_arbin = eval(step_var{1});
    
    % data1 구조체 생성
    data1.V = volt;
    data1.I = curr;
    data1.t = time;
    data1.step = step_arbin;
    
    %% Step 변경 위치 찾기
    change_indices = [1; find(diff(step_arbin) ~= 0) + 1; length(step_arbin) + 1];
    num_segments = length(change_indices) - 1;
    
    % data_line 구조체 템플릿 생성
    data_line = struct('V',[],'I',[],'t',[],'step',[]);
    
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
        data(i).step = data1.step(idx_start);  % 해당 구간의 step 번호
    end
    
    %% Trips 설정
    % 각 데이터셋에 따라 step_identifier 및 Trips 저장 이름 설정
    if strcmp(dataset_name, 'Processed_1')
        step_identifier = 14; % Processed_1의 경우 step 번호 (필요시 조정)
        Trips_var_name = ['Trips_Aging_1_', cell_name];
        save_filename = ['Trips_Aging_1_', cell_name, '.mat'];
    elseif strcmp(dataset_name, 'Processed_14')
        step_identifier = 14; % Processed_14의 경우 step 번호
        Trips_var_name = ['Trips_Aging_14_', cell_name];
        save_filename = ['Trips_Aging_14_', cell_name, '.mat'];
    else
        error('알 수 없는 데이터셋 이름: %s', dataset_name);
    end
    
    % step_identifier에 해당하는 인덱스 찾기
    target_indices = find([data.step] == step_identifier);
    if isempty(target_indices)
        error('지정된 step (%d)을 찾을 수 없습니다.', step_identifier);
    end
    primary_index = target_indices(1);
    
    % Rest, Discharge 단계 설정
    Rest_first_indice = primary_index - 2;
    Discharge_first_indice = primary_index - 1;
    Primary_first_indice = primary_index;
    
    if Rest_first_indice < 1 || Discharge_first_indice < 1
        error('Rest 또는 Discharge 단계의 인덱스가 유효하지 않습니다.');
    end
    
    % Trips 초기화
    Trips(1) = data(Rest_first_indice);
    Trips(2) = data(Discharge_first_indice);
    % Trips(3)부터 UDDS 사이클 데이터가 들어감
    
    %% UDDS 데이터 파싱 및 Trips 구조체 배열 채우기
    % Primary 데이터 가져오기
    primary_data = data(Primary_first_indice);
    
    % 시간 벡터를 0부터 시작하도록 조정
    t_primary = primary_data.t - primary_data.t(1);
    
    % 총 Primary 데이터 길이
    total_duration = t_primary(end);
    
    % UDDS 사이클 수 계산 (부분 사이클 포함)
    num_udds_cycles = ceil(total_duration / udds_duration);
    
    % UDDS 사이클의 시작 인덱스 저장할 배열 초기화
    udds_start_indices = zeros(num_udds_cycles+1, 1);
    
    % 첫 번째 사이클의 시작 인덱스는 1
    udds_start_indices(1) = 1;
    
    % 각 사이클의 시작 인덱스 찾기
    for i = 2:num_udds_cycles+1
        % 현재 사이클의 시작 시간을 계산
        current_start_time = (i-1) * udds_duration;
        
        if current_start_time > total_duration
            % 시작 시간이 전체 시간보다 크면 마지막 인덱스로 설정
            udds_start_indices(i) = length(t_primary);
        else
            % 시작 시간에 가장 가까운 인덱스를 찾음
            [~, idx] = min(abs(t_primary - current_start_time));
            udds_start_indices(i) = idx;
        end
    end
    
    % Trips 구조체 배열에 UDDS 사이클 데이터 저장
    for i = 1:num_udds_cycles
        idx_start = udds_start_indices(i);
        idx_end = udds_start_indices(i+1) - 1;
        
        % 인덱스가 유효한지 확인
        if idx_end < idx_start
            idx_end = idx_start;
        end
        
        % 데이터 분할 및 저장
        Trips(i+2).V = primary_data.V(idx_start:idx_end);
        Trips(i+2).I = primary_data.I(idx_start:idx_end);
        Trips(i+2).t = primary_data.t(idx_start:idx_end);
        Trips(i+2).step = primary_data.step; % step은 동일
        
        % time_reset 필드 추가: 각 Trip의 시작 시간을 0으로 재설정
        Trips(i+2).time_reset = Trips(i+2).t - Trips(i+2).t(1);
    end
    
    %% 파싱 포인트 및 Trip 레이블을 포함한 그래프 그리기
    figure;
    if strcmp(dataset_name, 'Processed_1')
        plot_color = 'g'; % Fresh 데이터는 초록색
    else
        plot_color = 'b'; % Aging 데이터는 파란색
    end
    plot(t_primary, primary_data.I, plot_color);
    hold on;
    
    % Y축 범위 가져오기 (레이블 위치 지정에 사용)
    y_limits = ylim;
    y_max = y_limits(2);
    y_min = y_limits(1);
    
    % 파싱 포인트를 빨간색 동그라미와 세로선으로 표시
    for i = 1:length(udds_start_indices)
        idx = udds_start_indices(i);
        t_point = t_primary(idx);
        I_point = primary_data.I(idx);
        
        % 빨간색 동그라미 표시
        plot(t_point, I_point, 'ro', 'MarkerSize', 8, 'LineWidth', 2);
        
        % 세로선 그리기
        plot([t_point, t_point], [y_min, y_max], 'r--', 'LineWidth', 1.5);
        
        % Trip 레이블 추가
        if i < length(udds_start_indices)
            % 현재 Trip의 중간 지점 계산
            t_next = t_primary(udds_start_indices(i+1));
            x_pos = (t_point + t_next) / 2;
            y_pos = y_max - (y_max - y_min) * 0.05; % 그래프 상단 바로 아래에 위치
            
            text(x_pos, y_pos, ['Trip ' num2str(i)], 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'top', 'FontSize', 12, 'FontWeight', 'bold');
        end
    end
    
    xlabel('Time (s)');
    ylabel('Current (A)');
    title(['Current Profile with UDDS Cycle Parsing Points - ', dataset_name, ' - ', cell_name]);
    hold off;
    
    %% Trips 저장
    assignin('base', Trips_var_name, Trips); % 워크스페이스에 저장
    save(fullfile(common_path, save_filename), 'Trips');
    fprintf('%s 파일이 저장되었습니다.\n', save_filename);
    
    % 그래프 저장 (옵션)
    figure_filename = ['Figure_', dataset_name, '_', cell_name, '.png'];
    saveas(gcf, fullfile(common_path, figure_filename));
    fprintf('%s 그래프가 저장되었습니다.\n', figure_filename);
    
    % 작업 완료 후 변수 초기화 (다음 루프를 위해)
    clear data data1 step_arbin time curr volt Trips
end

