clc; clear; close all;

%% 1. 데이터 로드
% 실제 데이터 파일 위치로 경로를 교체하세요
load('G:\공유 드라이브\BSL_Data3\HPPC\03-11-17_08.47 25degC_5Pulse_HPPC_Pan18650PF.mat');

%% 2. 데이터 전처리

% 데이터 구조 초기화
data1.I = meas.Current;
data1.V = meas.Voltage;
data1.t = meas.Time;

% 데이터 유형 분류: 'C'는 충전, 'D'는 방전, 'R'은 휴지
data1.type = char(zeros([length(data1.t), 1]));
data1.type(data1.I > 0) = 'C';
data1.type(data1.I == 0) = 'R';
data1.type(data1.I < 0) = 'D';

% 스텝 식별
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

% 각 스텝에 대한 구조 배열 초기화
data_line = struct('V', [], 'I', [], 't', [], 'indx', [], 'type', '', ...
    'steptime', [], 'T', [], 'SOC', [], 'avgI', [], 'Crate', [], 'deltaV', [], 'R', [], 't_rel', []);
data = repmat(data_line, num_step, 1);

for i_step = 1:num_step
    range = find(data1.step == vec_step(i_step));
    data(i_step).V = data1.V(range);
    data(i_step).I = data1.I(range);
    data(i_step).t = data1.t(range);
    data(i_step).indx = range;
    data(i_step).type = data1.type(range(1));
    data(i_step).steptime = data1.t(range);
    data(i_step).T = zeros(size(range)); % 온도 데이터가 없는 경우
    
    % 시간 초기화 (상대 시간)
    data(i_step).t_rel = data(i_step).t - data(i_step).t(1);
end

%% 3. 충전 및 방전 스텝 식별

% 스텝 인덱스를 저장할 배열 초기화
step_chg = [];
step_dis = [];

for i = 1:length(data)
    if strcmp(data(i).type, 'C')
        step_chg(end+1) = i;
    elseif strcmp(data(i).type, 'D')
        step_dis(end+1) = i;
    end
end

%% 4. 평균 전류, C-레이트, Delta V, 저항 계산

% 상수 정의
I_1C = 2.89923594059406; % 1C 전류 (방전 시 음수)

for i = 1:length(data)
    % 평균 전류
    data(i).avgI = mean(data(i).I);
    
    % C-레이트
    data(i).Crate = data(i).avgI / I_1C;
    
    % Delta Voltage
    if i == 1
        data(i).deltaV = zeros(size(data(i).V));
    else
        data(i).deltaV = data(i).V - data(i-1).V(end);
    end
    
    % 저항 (R0)
    if data(i).avgI == 0
        data(i).R = zeros(size(data(i).V));
    else
        data(i).R = (data(i).deltaV ./ data(i).avgI) .* ones(size(data(i).V));
    end
end

%% 5. HPPC 피팅 및 플롯

for i = 1:length(step_dis)
    idx = step_dis(i);
    % 해당 방전 스텝의 데이터 가져오기
    t_data = data(idx).t_rel; % 초기화된 시간 사용
    I_data = data(idx).I;
    V_data = data(idx).V;
    
    % 시간 간격 계산
    dt = [0; diff(t_data)];
    
    % OCV을 스텝의 첫 번째 전압으로 설정
    OCV = V_data(1);
    
    % 초기 추정 파라미터 [R0, R1, R2, C1, C2]
    initial_params = [data(idx).R(1), 0.01, 0.01, 1000, 1000]; % R0, R1, R2, C1, C2 초기값 설정
    
    % 파라미터의 하한 및 상한 설정
    lb = [0, 0, 0, 0, 0];    % 모든 파라미터는 0 이상
    ub = [inf, inf, inf, inf, inf]; % 상한은 무한대로 설정
    
    % 오류 함수 정의 (OCV은 고정)
    error_func = @(params) V_data - ECM_2RC_fixedOCV(params, OCV, I_data, dt);
    
    % 최적화를 통해 오류 최소화
    options = optimoptions('lsqnonlin','Display','off','TolFun',1e-8,'TolX',1e-8);
    [estimated_params, resnorm] = lsqnonlin(error_func, initial_params, lb, ub, options);
    
    % 추정된 파라미터 저장
    data(idx).params = estimated_params;
    data(idx).resnorm = resnorm;
    
    % 모델 전압 계산
    V_model = ECM_2RC_fixedOCV(estimated_params, OCV, I_data, dt);
    
    % 결과 플롯
    figure;
    plot(t_data, V_data, 'b-', 'LineWidth', 1.5);
    hold on;
    plot(t_data, V_model, 'r--', 'LineWidth', 1.5);
    xlabel('시간 (초)');
    ylabel('전압 (V)');
    title(['스텝 ' num2str(idx) ' 전압 비교']);
    legend('측정된 전압', '모델 전압');
    grid on;
   
end

%% ECM-2RC 모델 함수 정의 (OCV 고정, R0 포함)

function V_model = ECM_2RC_fixedOCV(params, OCV, I, dt)
    % 파라미터 분리
    R0 = params(1);
    R1 = params(2);
    R2 = params(3);
    C1 = params(4); 
    C2 = params(5); 

    tau1 = R1 * C1;
    tau2 = R2 * C2;
    
    N = length(I);
    V_R1 = zeros(N,1);
    V_R2 = zeros(N,1);
    V_R0 = R0 * I;
    V_model = zeros(N,1);
    
    % 초기 전압 설정
    V_model(1) = OCV + V_R0(1) + V_R1(1) + V_R2(1);
    
    for k = 2:N
        V_R1(k) = V_R1(k-1) * exp(-dt(k)/tau1) + I(k) * R1 * (1 - exp(-dt(k)/tau1));
        V_R2(k) = V_R2(k-1) * exp(-dt(k)/tau2) + I(k) * R2 * (1 - exp(-dt(k)/tau2));
        V_model(k) = OCV + V_R0(k) + V_R1(k) + V_R2(k);
    end
end
