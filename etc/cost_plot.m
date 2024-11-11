% 초기 설정
clc;clear;close all

% 데이터 로드
load('C:\Users\deu04\OneDrive\바탕 화면\wykht8y7tg-1\Panasonic 18650PF Data\Panasonic 18650PF Data\25degC\Drive cycles\03-21-17_00.29 25degC_UDDS_Pan18650PF.mat');
udds_current = meas.Current;  % 전류 데이터
udds_voltage = meas.Voltage;  % 전압 데이터
udds_time = meas.Time;        % 시간 데이터

% SOC-OCV 데이터 로드
load('soc_ocv.mat', 'soc_ocv');
soc_values = soc_ocv(:, 1);  % SOC 값
ocv_values = soc_ocv(:, 2);  % OCV 값

initial_soc = 0.9901;  % 초기 SOC 값
fixed_R0 = 0.025426;   % 고정된 R0 값

Config.cap = 2.90;                  % 배터리 용량 (Ah)
Config.coulomb_efficient = 1;       % 쿨롱 효율

% R1과 C1의 범위 설정
R1_range = linspace(0.01, 1, 20);  % R1 값의 범위 (Ohms)
C1_range = linspace(1, 1300, 20);  % C1 값의 범위 (Farads)

results = struct('R0', [], 'R1', [], 'C1', [], 'Vt_est', [], 'Measured_Voltage', [], 'Residuals', [], 'Cost', [] );

% R1과 C1 범위 조합 iteration
idx = 1;
for i = 1:length(R1_range)
    for j = 1:length(C1_range)

        R1 = R1_range(i);
        C1 = C1_range(j);
        params = [fixed_R0, R1, C1];
        
        % residual, estimated 전압 계산
        [residuals, Vt_est] = voltage_error(params, initial_soc, udds_current, udds_voltage, Config, soc_values, ocv_values, udds_time);
        
        % cost 계산
        cost = sum(residuals.^2);
        
        % 결과를 구조체 배열에 저장
        results(idx).R0 = fixed_R0;
        results(idx).R1 = R1;
        results(idx).C1 = C1;
        results(idx).Vt_est = Vt_est;  % estimated 전압 저장
        results(idx).Measured_Voltage = udds_voltage;  % measured 전압 저장
        results(idx).Residuals = residuals;
        results(idx).Cost = cost;
         
        idx = idx + 1;
    end
end

%% plot

R1_values = [results.R1];
C1_values = [results.C1];
Cost_values = [results.Cost];

% R1, C1의 mesh 생성
[R1_mesh, C1_mesh] = meshgrid(R1_range, C1_range);

% 각 조합에 대한 Cost 값을 행렬 변환
Cost_mesh = reshape(Cost_values, length(C1_range), length(R1_range));

% 3D surface 플롯
figure;
surf(R1_mesh, C1_mesh, log10(Cost_mesh));
xlabel('R1 (Ohms)');
ylabel('C1 (Farads)');
zlabel('Cost');
title('Cost Surface Plot');
colorbar;
grid on;

% 2D 등고선 플롯
figure;
contourf(R1_mesh, C1_mesh, log10(Cost_mesh), 20); 
ylabel('C1 (Farads)');
title('Cost Contour Plot');
colorbar;
grid on;

%% function

% cost 함수 정의 ( 안 써도 괜찮을듯?)
function cost = func_cost(params, R0, initial_soc, current, voltage, Config, soc_values, ocv_values, time_data)
    % residual 계산
    [residuals, ~] = voltage_error([R0, params], initial_soc, current, voltage, Config, soc_values, ocv_values, time_data);
    
    % cost 계산
    cost = sum(residuals.^2);
end

% Vt-Vest residual 계산 
function [residuals, Vt_est] = voltage_error(params, initial_soc, current, voltage, Config, soc_values, ocv_values, time_data)
    R0 = params(1);  % 고정된 R0
    R1 = params(2);  % 최적화된 R1
    C1 = params(3);  % 최적화된 C1

    num_samples = length(current);  % 샘플 개수
    SOC_est = zeros(num_samples, 1);  % SOC 추정값
    V1_est = zeros(num_samples, 1);   % V1 추정값
    Vt_est = zeros(num_samples, 1);   % 단자 전압 추정값
    residuals = zeros(num_samples, 1); % residual 벡터 초기화

    SOC_est(1) = initial_soc;  % 초기 SOC 설정
    dt_1 = time_data(2) - time_data(1);  % 첫 번째 샘플의 시간 간격
    V1_est(1) = current(1) * R1 * (1 - exp(-dt_1 / (R1 * C1)));  % 초기 V1 추정
    Vt_est(1) = voltage(1);  % 초기 전압 설정
    residuals(1) = Vt_est(1) - voltage(1);  % 초기 잔차 계산

    for k = 2:num_samples
        ik = current(k);
        dt_k = time_data(k) - time_data(k-1);  % 현재 샘플과 이전 샘플 간의 시간 차이 (mean 값 x)

        SOC_est(k) = SOC_est(k-1) + (dt_k / (Config.cap * 3600)) * Config.coulomb_efficient * ik;
        V1_est(k) = exp(-dt_k / (R1 * C1)) * V1_est(k-1) + (1 - exp(-dt_k / (R1 * C1))) * ik * R1;
        Vt_est(k) = interp1(soc_values, ocv_values, SOC_est(k), 'linear', 'extrap') + V1_est(k) + R0 * ik;
        
        residuals(k) = Vt_est(k) - voltage(k);  % 각 샘플에 대한 잔차 계산
    end
end

