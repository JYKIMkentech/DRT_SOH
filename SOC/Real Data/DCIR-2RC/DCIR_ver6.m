clc; clear; close all;

% 데이터 로드
% data = load('C:\Users\USER\Desktop\Panasonic 18650PF Data\Panasonic 18650PF Data\25degC\5 pulse disch\03-11-17_08.47 25degC_5Pulse_HPPC_Pan18650PF.mat');
data = load('G:\공유 드라이브\BSL_Data3\HPPC\03-11-17_08.47 25degC_5Pulse_HPPC_Pan18650PF.mat');

% 시간, 전압, 전류 데이터 추출
time = data.meas.Time;
voltage = data.meas.Voltage;
current = data.meas.Current;

data1.I = current;
data1.V = voltage;
data1.t = time;

% 전류 상태 구분
data1.type = char(zeros([length(data1.t), 1]));
data1.type(data1.I > 0) = 'C';
data1.type(data1.I == 0) = 'R';
data1.type(data1.I < 0) = 'D';

% Step 구분
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

% Initialize data structure
data_line = struct('V', [], 'I', [], 't', [], 'indx', [], 'type', '', ...
    'steptime', [], 'T', [], 'SOC', []);
data = repmat(data_line, num_step, 1);

for i_step = 1:num_step
    range = find(data1.step == vec_step(i_step));
    data(i_step).V = data1.V(range);
    data(i_step).I = data1.I(range);
    data(i_step).t = data1.t(range);
    data(i_step).indx = range;
    data(i_step).type = data1.type(range(1));
    data(i_step).steptime = data1.t(range);
    data(i_step).T = zeros(size(range)); % 온도 데이터가 없으므로 0으로 설정
end

% 초기 SOC 설정 (1로 가정)
initial_SOC = 1;
capacity_Ah = 2.9; % 배터리 용량 (Ah)

% Discharge step 구하기
step_chg = [];
step_dis = [];

for i = 1:length(data)
    % type 필드가 C인지 확인
    if strcmp(data(i).type, 'C')
        % C가 맞으면 idx 추가
        step_chg(end+1) = i;
    % type 필드가 D인지 확인
    elseif strcmp(data(i).type, 'D')
        % D가 맞으면 idx 추가
        step_dis(end+1) = i;
    end
end

%% R0, R1, C 추출 

% 평균 전류 구하기
for i = 1:length(data)
    data(i).avgI = mean(data(i).I);
end

% V 변화량 구하기
for i = 1 : length(data)
    if i == 1
       data(i).deltaV = zeros(size(data(i).V));
    else
       data(i).deltaV = data(i).V - data(i-1).V(end);
    end
end

% Resistance 구하기 
for i = 1 : length(data)
    if data(i).avgI == 0
        data(i).R = zeros(size(data(i).V));
    else 
        data(i).R = (data(i).deltaV / data(i).avgI) .* ones(size(data(i).V));
    end
end

% 시간 초기화 및 원본 시간 저장
for i = 1 : length(data)
    data(i).t_original = data(i).t;       % 원본 시간 저장
    initialTime = data(i).t(1);           % 초기 시간 저장
    data(i).t_reg = data(i).t - initialTime; % 시간 초기화
end

for i = 1:length(step_dis)
    % 시간의 길이가 5초 이상인 스텝에 대해서만 R1s 값을 계산
    if length(data(step_dis(i)).t_reg) >= 5
       data(step_dis(i)).R001s = data(step_dis(i)).R(1);
       if length(data(step_dis(i)).R) >= 11
           data(step_dis(i)).R1s = data(step_dis(i)).R(11);
       else
           data(step_dis(i)).R1s = data(step_dis(i)).R(end);
       end
       data(step_dis(i)).R0 = data(step_dis(i)).R001s;
       data(step_dis(i)).R1 = data(step_dis(i)).R1s - data(step_dis(i)).R001s;
    else
       data(step_dis(i)).R001s = NaN;
       data(step_dis(i)).R1s = NaN;
       data(step_dis(i)).R0 = NaN;
       data(step_dis(i)).R1 = NaN;
    end
end

%% 63.2% 값을 이용한 tau 및 C 계산

timeAt632 = zeros(1, length(step_dis));  % Initialize timeAt632 as a vector

for i = 1:length(step_dis)
    % 최소값과 최대값 계산
    minVoltage = min(data(step_dis(i)).V);
    maxVoltage = max(data(step_dis(i)).V);

    % 63.2% 값 계산
    targetVoltage = minVoltage + (1 - 0.632 ) * (maxVoltage - minVoltage);

    % 63.2%에 가장 가까운 값의 인덱스 찾기
    [~, idx] = min(abs(data(step_dis(i)).V - targetVoltage));

    % 해당 시간 찾기
    timeAt632(i) = data(step_dis(i)).t_reg(idx); % 초기화된 시간 사용

    % data(step_dis(i)) 구조체에 timeAt632 필드를 추가하고 값 할당
    data(step_dis(i)).timeAt632 = timeAt632(i);
end

% C값 구하기
for i = 1:length(step_dis)
    if ~isnan(data(step_dis(i)).R1) && data(step_dis(i)).R1 ~= 0
        data(step_dis(i)).C = data(step_dis(i)).timeAt632 / data(step_dis(i)).R1;
    else
        data(step_dis(i)).C = NaN;
    end
end

% SOC 값을 정의된 패턴에 따라 생성
soc_values = [1, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05];
steps_per_level = 5;

% SOC 배열 초기화
SOC = zeros(length(step_dis), 1);
current_index = 1;

for i = 1:length(soc_values)
    end_index = min(current_index + steps_per_level - 1, length(step_dis));
    SOC(current_index:end_index) = soc_values(i);
    current_index = end_index + 1;
    if current_index > length(step_dis)
        break;
    end
end

% step_dis 배열을 사용하여 데이터에 SOC 값 할당
for i = 1:length(step_dis)
    data(step_dis(i)).SOC = SOC(i);
end

% 특정 인덱스 SOC 값 설정 (예: 130번째 스텝)
if length(step_dis) >= 130
    data(step_dis(130)).SOC = 0.05;
end

% ECM 1RC 모델 파라미터를 저장할 필드 추가
[data.R0] = deal(NaN);
[data.R1] = deal(NaN);
[data.C] = deal(NaN);
[data.OCV] = deal(NaN);

% Optimization 옵션 설정
options = optimset('Display','off', 'TolX',1e-6, 'TolFun',1e-6, 'MaxIter',1000, 'MaxFunEvals',1000);

% 이전 스텝의 마지막 V_model을 저장할 변수 초기화
last_V_model = NaN;

% 각 discharge step에 대해 피팅 수행
for i = 1:length(step_dis)
    step_idx = step_dis(i);
    step_data = data(step_idx);
    
    % Discharge 단계의 데이터 추출
    V_meas = step_data.V;
    I_meas = step_data.I;
    t_meas = step_data.t_original;  % 원본 시간 사용
    
    % dt 계산: 마지막 dt를 평균으로 대체하지 않음
    dt_diff = diff(t_meas);
    dt = [dt_diff(1); dt_diff]; % 첫 번째 dt를 복사하여 전체 길이 맞춤
    
    % OCV 설정: 스텝 시작 전의 전압 또는 스텝 시작 전의 평균 전압
    if step_idx > 1
        prev_step = step_idx - 1;
        OCV = mean(data(prev_step).V(end-10:end)); % 이전 스텝의 마지막 10개 전압 평균
    else
        OCV = mean(V_meas(1:min(10, end))); % 첫 스텝의 초기 전압 평균
    end
    
    % 피팅에 사용할 전압 초기화는 제거
    % V_model은 compute_model_voltage 함수 내에서 계산됨
    
    % Objective 함수 정의 (모델 전압과 측정 전압의 차이)
    objective = @(params) sum((V_meas - compute_model_voltage(params, I_meas, dt, OCV)).^2);
    
    % 초기 추정값 [R0, R1, C]
    initial_guess = [0.01, 0.1, 100]; % 예: R0=0.01 Ohm, R1=0.1 Ohm, C=100 F
    
    % 파라미터 경계 설정 (R0, R1, C는 양수)
    lb = [0, 0, 0];
    ub = [1, 10, 10000];
    
    % 파라미터 최적화
    [params_opt, ~] = fmincon(objective, initial_guess, [], [], [], [], lb, ub, [], options);
    
    % 최적화된 파라미터 저장
    data(step_idx).R0 = params_opt(1);
    data(step_idx).R1 = params_opt(2);
    data(step_idx).C = params_opt(3);
    data(step_idx).OCV = OCV;
    
    % 모델 전압 계산
    V_model = compute_model_voltage(params_opt, I_meas, dt, OCV);
    data(step_idx).V_model = V_model; % 모델 전압을 구조체에 저장 (추가)
    
    % (옵션) 모델 전압 계산 및 플롯
    figure;
    plot(t_meas, V_meas, 'b', t_meas, V_model, 'r--');
    title(['Step ', num2str(step_idx), ' Voltage Fitting']);
    xlabel('Time (s)');
    ylabel('Voltage (V)');
    legend('Measured', 'Fitted');
    hold off;
    
    % 마지막 V_model을 저장하여 다음 스텝에 사용 (필요시)
    last_V_model = V_model(end);
end

% 모델 전압 계산 함수 수정
function V_model = compute_model_voltage(params, I_meas, dt, OCV)
    R0 = params(1);
    R1 = params(2);
    C = params(3);
    V_model = zeros(size(I_meas));
    
    % 첫 번째 전압을 모델 방정식에 따라 초기화
    V_model(1) = OCV + I_meas(1)*R0 + I_meas(1)*R1 * (1 - exp(-dt(1)/(R1*C)));
    
    for k = 2:length(I_meas)
        V_prev = V_model(k-1);
        I = I_meas(k);
        V_model(k) = OCV + I * R0 + V_prev * exp(-dt(k)/(R1 * C)) + I * R1 * (1 - exp(-dt(k)/(R1 * C)));
    end
end


