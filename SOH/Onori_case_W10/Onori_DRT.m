clc;clear;close all;

%% 0. 폰트 크기 및 색상 매트릭스 설정
% Font size settings
axisFontSize = 14;      % 축의 숫자 크기
titleFontSize = 16;     % 제목의 폰트 크기
legendFontSize = 12;    % 범례의 폰트 크기
labelFontSize = 14;     % xlabel 및 ylabel의 폰트 크기

% Color matrix 설정
c_mat = lines(9);  % 9개의 고유한 색상 정의

%% 1. trips 데이터 로드
% trips 데이터를 로드합니다.
load('G:\공유 드라이브\BSL_Onori\Cycling_tests\Trips_Aging_1_W10.mat');  % 'trips' 구조체를 로드합니다.

col_cell_label = {'W3','W4','W5','W7','W8','W9','W10','G1','V4','V5'};


%% 2. SOC-OCV 데이터 로드
% RPT_10_soc_ocv_cap.mat 파일에서 soc_ocv_cap 데이터를 로드합니다.
%load('RPT_10_soc_ocv_cap.mat', 'soc_ocv_cap');
load('RPT_All_soc_ocv_cap.mat', 'soc_ocv_cap');

% SOC와 OCV 값 추출
soc_values = soc_ocv_cap{14,7}(:, 1);  % SOC 값 (0 ~ 1)
ocv_values = soc_ocv_cap{14,7}(:, 2);  % OCV 값 (V)

% 배터리 용량 추출 (Capacity 열의 최대값)
Q_batt = max(soc_ocv_cap{14,7}(:, 3));  % Ah 단위

%% 3. DRT 추정에 필요한 파라미터 설정
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
lambda = 1.6e-1;  % 최적화된 람다 값 (필요에 따라 조정 가능)
%lambda = 1.2e-2;

% Gamma에 대한 1차 차분 행렬 L_gamma 생성
L_gamma = zeros(n-1, n);
for i = 1:n-1
    L_gamma(i, i) = -1;
    L_gamma(i, i+1) = 1;
end

% R0에 대한 정규화를 피하기 위해 L_aug 생성
L_aug = [L_gamma, zeros(n-1, 1)];

%% 4. 각 트립에 대한 DRT 추정 (quadprog 사용)
num_trips = length(Trips);

% 결과 저장을 위한 배열 사전 할당
gamma_est_all = zeros(num_trips, n);
R0_est_all = zeros(num_trips, 1);
soc_start_all = zeros(num_trips, 1);  % 각 트립의 시작 SOC 저장

% 서브플롯 그리드 크기 계산
num_cols = ceil(sqrt(num_trips));
num_rows = ceil(num_trips / num_cols);

SOC0 = 0.7997;  % 전체 시뮬레이션의 초기 SOC 설정

for s = 3:num_trips
    fprintf('Processing Trip %d/%d...\n', s, num_trips);
    
    % 현재 트립의 데이터 추출
    t = Trips(s).time_reset;
    ik = Trips(s).I;
    V_sd = Trips(s).V;
    
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

    %% 4.1 W 행렬 생성
    % W 행렬 초기화
    W = zeros(length(t), n);
    for k_idx = 1:length(t)
        for i = 1:n
            if k_idx == 1
                W(k_idx, i) = ik(k_idx) * (1 - exp(-delta_t(k_idx) / tau_discrete(i))) * delta_theta;
            else
                W(k_idx, i) = W(k_idx-1, i) * exp(-delta_t(k_idx) / tau_discrete(i)) + ...
                              ik(k_idx) * (1 - exp(-delta_t(k_idx) / tau_discrete(i))) * delta_theta;
            end
        end
    end

    % R0 추정을 위한 W 행렬 확장
    W_aug = [W, ik(:)];  % ik(:)는 ik를 열 벡터로 변환

    %% 4.2 y 벡터 생성
    % y = V_sd - OCV
    y = V_sd - ocv_over_time;
    y = y(:);  % y를 열 벡터로 변환

    %% 4.3 quadprog를 사용한 제약 조건 하의 추정
    % 비용 함수: 0.5 * Theta' * H * Theta + f' * Theta
    H = (W_aug' * W_aug + lambda * (L_aug' * L_aug));
    f = -W_aug' * y;

    % 제약 조건: Theta >= 0 (gamma와 R0는 0 이상)
    A = -eye(n+1);
    b = zeros(n+1, 1);

    % quadprog 옵션 설정
    options = optimoptions('quadprog', 'Display', 'off', 'Algorithm', 'interior-point-convex');

    % quadprog 실행
    [Theta_est, ~, exitflag] = quadprog(H, f, A, b, [], [], [], [], [], options);

    if exitflag ~= 1
        warning('Optimization did not converge for trip %d.', s);
    end

    % gamma와 R0 추정값 추출
    gamma_est = Theta_est(1:n);
    R0_est = Theta_est(n+1);

    % 추정값 저장
    gamma_est_all(s, :) = gamma_est';
    R0_est_all(s) = R0_est;

    %% 4.4 V_est 계산 (검증용)
    % V_RC 및 V_est 초기화
    V_RC = zeros(n, length(t));  % 각 요소의 전압
    V_est = zeros(length(t), 1);
    for k_idx = 1:length(t)
        if k_idx == 1
            for i = 1:n
                V_RC(i, k_idx) = gamma_est(i) * delta_theta * ik(k_idx) * (1 - exp(-delta_t(k_idx) / tau_discrete(i)));
            end
        else
            for i = 1:n
                V_RC(i, k_idx) = V_RC(i, k_idx-1) * exp(-delta_t(k_idx) / tau_discrete(i)) + ...
                                 gamma_est(i) * delta_theta * ik(k_idx) * (1 - exp(-delta_t(k_idx) / tau_discrete(i)));
            end
        end
        % 시간 k_idx에서의 V_est 계산
        V_est(k_idx) = ocv_over_time(k_idx) + R0_est * ik(k_idx) + sum(V_RC(:, k_idx));
    end

    %% 4.5 DRT Gamma 그래프 출력
    % Figure 1: DRT Gamma 서브플롯
    figure(1);
    subplot(num_rows, num_cols, s);
    plot(theta_discrete, gamma_est, 'Color', c_mat(mod(s-1,9)+1, :), 'LineWidth', 1.5);
    xlabel('\theta = ln(\tau)', 'FontSize', labelFontSize);
    ylabel('\gamma', 'FontSize', labelFontSize);
    title(['DRT for Trip ', num2str(s)], 'FontSize', titleFontSize);
    set(gca, 'FontSize', axisFontSize);

    % R0 추정값을 그래프에 텍스트로 추가
    x_text = min(theta_discrete);
    y_text = max(gamma_est);
    text(x_text, y_text, sprintf('R₀ = %.3e Ω', R0_est_all(s)), ...
        'FontSize', 8, 'Color', 'k', 'FontWeight', 'bold', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');

    %% 4.6 전압 비교 그래프 출력
    % Figure 2: Voltage and Current Comparison 서브플롯
    figure(2);
    subplot(num_rows, num_cols, s);
    yyaxis left
    plot(t, V_sd, 'Color', c_mat(mod(s-1,9)+1, :), 'LineWidth', 1, 'DisplayName', 'Measured V_{udds}');
    hold on;
    plot(t, V_est, '--', 'Color', c_mat(mod(s-1,9)+1, :), 'LineWidth', 1, 'DisplayName', 'Estimated V_{est}');
    xlabel('Time (s)', 'FontSize', labelFontSize);
    ylabel('Voltage (V)', 'FontSize', labelFontSize, 'Color', 'k');
    title(['Voltage and Current Comparison for Trip ', num2str(s)], 'FontSize', titleFontSize);
    legend('Location', 'best', 'FontSize', legendFontSize);

    yyaxis right
    plot(t, ik, 'Color', 'g', 'LineWidth', 1, 'DisplayName', 'Current (A)');
    ylabel('Current (A)', 'FontSize', labelFontSize, 'Color', 'g');
    set(gca, 'YColor', 'g');
    legend('Location', 'best', 'FontSize', legendFontSize);
    set(gca, 'FontSize', axisFontSize);
    hold off;

    %% 4.7 Trip 1에 대한 별도의 Gamma 그래프 추가
    % Trip 1의 gamma 그래프를 별도의 큰 그림으로 플롯합니다.
    if s == 1
        figure(5);  % 새로운 figure 생성
        set(gcf, 'Position', [100, 100, 800, 600]);  % Figure 크기 조정
        plot(theta_discrete, gamma_est_all(1, :), 'LineWidth', 3, 'Color', c_mat(1, :));
        xlabel('$\theta = \ln(\tau \, [s])$', 'Interpreter', 'latex', 'FontSize', labelFontSize)
        ylabel('\gamma [\Omega]', 'FontSize', labelFontSize);
        title('Trip 1 : DRT ', 'FontSize', titleFontSize);
        hold on;

        % R0 추정값을 그래프에 텍스트로 추가
        x_text = min(theta_discrete);
        y_text = max(gamma_est_all(1, :));
        text(x_text, y_text, sprintf('R₀ = %.3e Ω', R0_est_all(1)), ...
            'FontSize', 12, 'Color', 'k', 'FontWeight', 'bold', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');

        hold off;

        %% 4.8 Trip 1에 대한 I, V, V_model vs t 그래프 추가
        figure(6);  % 새로운 figure 생성
        set(gcf, 'Position', [150, 150, 800, 600]);  % Figure 크기 조정

        % 왼쪽 Y축: Voltage (V_sd 및 V_est)
        yyaxis left
        plot(t, V_sd, 'Color', c_mat(1, :), 'LineWidth', 3, 'DisplayName', 'Measured V_{udds}');
        hold on;
        plot(t, V_est, '--', 'Color', c_mat(2, :), 'LineWidth', 3, 'DisplayName', 'Estimated V_{est}');
        ylabel('Voltage (V)', 'FontSize', labelFontSize, 'Color', c_mat(1, :));

        % 오른쪽 Y축: Current (ik)
        yyaxis right
        plot(t, ik, 'Color', c_mat(3, :), 'LineWidth', 3, 'DisplayName', 'Current (A)');
        ylabel('Current (A)', 'FontSize', labelFontSize, 'Color', c_mat(3, :));
        set(gca, 'YColor', c_mat(3, :));

        % X축 레이블 설정
        xlabel('Time (s)', 'FontSize', labelFontSize);

        % 제목 설정
        title('I, V, V_{model} vs Time for Trip 1', 'FontSize', titleFontSize);

        % 범례 설정
        legend('Location', 'best', 'FontSize', legendFontSize);

        % 축의 숫자(틱 라벨) 폰트 크기 설정
        set(gca, 'FontSize', axisFontSize);

        hold off;
    end

    % 다음 트립을 위한 SOC0 업데이트
    SOC0 = SOC(end);  % 현재 트립의 마지막 SOC로 업데이트

 
end

%% 5. gamma(soc, theta)를 이용한 3D DRT 그래프 생성

% soc_min과 soc_max를 정의합니다.
soc_min = min(soc_start_all);
soc_max = max(soc_start_all);

% gamma_est_all의 최대값을 이용하여 z축 한계를 설정합니다.
z_threshold = max(gamma_est_all(:)) * 1.1;  % 최대값의 110%로 설정

figure(10);
hold on;

% 각 트립에 대해 3D 그래프를 플롯합니다.
for s = 3:num_trips
    % 데이터 준비
    x_data = soc_start_all(s) * ones(size(theta_discrete));  % SOC 값
    y_data = theta_discrete;                                 % θ 값
    z_data = gamma_est_all(s, :)';                           % gamma 값
    
    % 3D 플롯
    plot3(x_data, y_data, z_data, 'LineWidth', 1.5, 'Color', c_mat(mod(s-1,9)+1, :));
end

% 축 레이블 및 제목 설정
xlabel('SOC', 'FontSize', labelFontSize);
ylabel('$\theta = \ln(\tau \, [s])$', 'Interpreter', 'latex', 'FontSize', labelFontSize);
zlabel('\gamma [\Omega]', 'FontSize', labelFontSize);
title('3D DRT', 'FontSize', titleFontSize);

% 컬러맵 및 컬러바 설정
colormap(jet);
c = colorbar;
c.Label.String = 'SOC';
caxis([soc_min soc_max]);

% 축 한계 설정
xlim([0 1]);
ylim([min(theta_discrete) max(theta_discrete)]);
zlim([0, z_threshold]);

% 뷰 설정
view(135, 30);
grid on;

set(gca, 'FontSize', axisFontSize);
hold off;