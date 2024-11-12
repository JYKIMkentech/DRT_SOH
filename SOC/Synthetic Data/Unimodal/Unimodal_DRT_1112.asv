clc; clear; close all;

%% Description
 
% True n = 201;

%% Graphic 

axisFontSize = 14;
titleFontSize = 12;
legendFontSize = 12;
labelFontSize = 12;

%% load

file_path = 'G:\공유 드라이브\Battery Software Lab\Projects\DRT\SD\';
mat_files = dir(fullfile(file_path, '*.mat'));

for file = mat_files'
    load(fullfile(file_path, file.name));
end

%% Parameter

AS_structs = {AS1_1per, AS1_2per, AS2_1per, AS2_2per};
AS_names = {'AS1_1per', 'AS1_2per', 'AS2_1per', 'AS2_2per'};
Gamma_structs = {Gamma_unimodal, Gamma_unimodal, Gamma_bimodal, Gamma_bimodal};
lambda_values = [2.56, 2.56, 2.56, 2.56];  % 각 경우에 대한 람다 값
AS_num = numel(AS_structs);
num_scenarios = length(AS1_1per);  % 시나리오 수  = 10개 
c_mat = lines(num_scenarios);  % 시나리오 수에 따라 색상 매트릭스 생성
n = length(Gamma_bimodal.theta);  % RC 요소의 개수 (Base = 201개)
OCV = 0;
R0 = 0.1;
dt = AS1_1per(1).t(2)-AS1_1per(1).t(1);

  % 일차 차분 행렬 L 생성 (정규화에 사용)
L = zeros(n-1, n);
for i = 1:n-1
    L(i, i) = -1;
    L(i, i+1) = 1;
end


%% DRT 

for m = 1:AS_num
    AS_data = AS_structs{m};
    AS_name = AS_names{m};
    Gamma_data = Gamma_structs{m};
    lambda = lambda_values(m);
    t = AS_data(1).t;
    
    % theta 및 tau 설정
    theta_discrete = Gamma_data.theta';
    gamma_discrete_true = Gamma_data.gamma';
    tau_discrete = exp(theta_discrete);
    delta_theta = theta_discrete(2) - theta_discrete(1);
    
    % DRT 추정 결과를 저장할 변수
    gamma_quadprog_all = zeros(num_scenarios, n);
    V_est_all = zeros(num_scenarios, length(t));
    V_sd_all = zeros(num_scenarios, length(t));
    ik_scenarios = zeros(num_scenarios, length(t));
    
    for s = 1:num_scenarios
        fprintf('Processing %s Scenario %d/%d...\n', AS_name, s, num_scenarios);
        
        % 현재 시나리오의 데이터 가져오기
        V_sd = AS_data(s).V;    % 측정 전압 (열 벡터)
        ik = AS_data(s).I;      % 전류 (열 벡터)
        ik_scenarios(s, :) = ik';  % 전류 저장 (행 벡터로 변환)

        % W 행렬 초기화
        W = zeros(length(t), n);
        for k_idx = 1:length(t)
            if k_idx == 1
                for i = 1:n
                    W(k_idx, i) = ik(k_idx) * (1 - exp(-dt / tau_discrete(i))) * delta_theta;
                end
            else
                for i = 1:n
                    W(k_idx, i) = W(k_idx-1, i) * exp(-dt / tau_discrete(i)) + ...
                                  ik(k_idx) * (1 - exp(-dt / tau_discrete(i))) * delta_theta;
                end
            end
        end
        
        % y_adjusted 계산
        y_adjusted = V_sd - OCV - R0 * ik;
        
        % Quadprog를 위한 행렬 및 벡터 구성
        H = 2 * (W' * W + lambda * (L' * L));
        f = -2 * W' * y_adjusted;
        
        % 부등식 제약조건: gamma ≥ 0
        A_ineq = -eye(n);
        b_ineq = zeros(n, 1);
        
        % Quadprog 옵션 설정
        options = optimoptions('quadprog', 'Display', 'off');
        
        % Quadprog를 사용하여 최적화 문제 해결
        gamma_quadprog = quadprog(H, f, A_ineq, b_ineq, [], [], [], [], [], options);
        
        % 결과 저장
        gamma_quadprog_all(s, :) = gamma_quadprog';
        V_est = OCV + R0 * ik + W * gamma_quadprog;
        V_est_all(s, :) = V_est';
        V_sd_all(s, :) = V_sd';
    end
    
    %% 결과 플롯
       
    % DRT 비교 플롯
    figure('Name', [AS_name, ': DRT Comparison'], 'NumberTitle', 'off');
    hold on;
    for s = 1:num_scenarios
        plot(theta_discrete, gamma_quadprog_all(s, :), '--', 'LineWidth', 1.5, ...
            'Color', c_mat(s, :), 'DisplayName', ['Scenario ', num2str(s)]);
    end
    plot(theta_discrete, gamma_discrete_true, 'k-', 'LineWidth', 2, 'DisplayName', 'True γ');
    hold off;
    xlabel('\theta = ln(\tau [s])', 'FontSize', labelFontSize);
    ylabel('\gamma', 'FontSize', labelFontSize);
    title([AS_name, ': Estimated \gamma (\lambda = ', num2str(lambda), ')'], 'FontSize', titleFontSize);
    set(gca, 'FontSize', axisFontSize);
    legend('Location', 'Best', 'FontSize', legendFontSize);
    
    % 선택된 시나리오에 대한 추가 그래프
    selected_scenarios = [6, 7, 8, 9];
      
    % DRT 비교 플롯 (선택된 시나리오)
    figure('Name', [AS_name, ': Selected Scenarios DRT Comparison'], 'NumberTitle', 'off');
    hold on;
    for idx_s = 1:length(selected_scenarios)
        s = selected_scenarios(idx_s);
        plot(theta_discrete, gamma_quadprog_all(s, :), '--', 'LineWidth', 1.5, ...
            'Color', c_mat(s, :), 'DisplayName', ['Scenario ', num2str(s)]);
    end
    plot(theta_discrete, gamma_discrete_true, 'k-', 'LineWidth', 2, 'DisplayName', 'True γ');
    hold off;
    xlabel('\theta = ln(\tau [s])', 'FontSize', labelFontSize);
    ylabel('\gamma', 'FontSize', labelFontSize);
    title([AS_name, ': Estimated \gamma for Selected Scenarios'], 'FontSize', titleFontSize);
    set(gca, 'FontSize', axisFontSize);
    legend('Location', 'Best', 'FontSize', legendFontSize);
end
