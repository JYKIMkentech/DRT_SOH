clear; clc; close all;

%% 0. 폰트 크기 및 색상 매트릭스 설정
% 폰트 크기 설정
axisFontSize = 14;      % 축의 숫자 크기
titleFontSize = 16;     % 제목의 폰트 크기
legendFontSize = 12;    % 범례의 폰트 크기
labelFontSize = 14;     % xlabel 및 ylabel의 폰트 크기

% 색상 매트릭스 설정
c_mat = lines(9);  % 9개의 고유한 색상 정의

%% 1. UDDS 주행 데이터 로드
% UDDS 주행 데이터를 로드합니다.
load('udds_data.mat');  % 'udds_data' 구조체를 로드합니다.

%% 2. SOC-OCV 데이터 로드
% SOC-OCV 데이터를 로드합니다.
load('soc_ocv.mat', 'soc_ocv');
soc_values = soc_ocv(:, 1);  % SOC 값
ocv_values = soc_ocv(:, 2);  % OCV 값

%% 3. DRT 추정에 필요한 파라미터 설정
n = 201;  % 이산 요소의 개수
tau_min = 0.1;     % 최소 시간 상수 (초)
tau_max = 1100;    % 최대 시간 상수 (초)

% Theta 및 tau 값 계산
theta_min = log(tau_min);
theta_max = log(tau_max);
theta_discrete = linspace(theta_min, theta_max, n);
tau_discrete = exp(theta_discrete);

% Delta theta 계산
delta_theta = theta_discrete(2) - theta_discrete(1);

% 정규화 파라미터
lambda = 0.204; 

% Gamma에 대한 1차 차분 행렬 L_gamma 생성
L_gamma = zeros(n-1, n);
for i = 1:n-1
    L_gamma(i, i) = -1;
    L_gamma(i, i+1) = 1;
end

% R0에 대한 정규화를 피하기 위해 L_aug 생성
L_aug = [L_gamma, zeros(n-1, 1)];

%% 4. 각 트립에 대한 DRT 추정 (quadprog 사용)
num_trips = length(udds_data);

% 결과 저장을 위한 배열 사전 할당
gamma_est_all = zeros(num_trips-1, n);  % 마지막 트립 제외
R0_est_all = zeros(num_trips-1, 1);
soc_mid_all = zeros(num_trips-1, 1);  % 각 트립의 중간 SOC 저장

% 추가: 각 트립의 W, V_RC, OCV, V_est를 저장할 셀 배열 생성
W_all = cell(num_trips-1, 1);
V_RC_all = cell(num_trips-1, 1);
OCV_all = cell(num_trips-1, 1);
V_est_all = cell(num_trips-1, 1);

% V_RC_end와 W_end를 초기화 (첫 번째 트립의 시작 시점에는 0으로 설정)
V_RC_end = zeros(n, 1);  % V_RC_end를 zeros로 초기화
W_end = zeros(1, n);     % W_end를 zeros로 초기화

for s = 1:num_trips-1  % 마지막 트립은 데이터가 짧으므로 제외
    fprintf('Processing Trip %d/%d...\n', s, num_trips-1);
    
    % 현재 트립의 데이터 추출
    t = udds_data(s).t;
    ik = udds_data(s).I;
    V_sd = udds_data(s).V;
    SOC = udds_data(s).SOC;
    
    % 각 트립의 중간 SOC 계산 (중간 시간에 해당하는 SOC)
    t_mid = t(end) / 2;  % 트립의 중간 시간
    
    % t와 SOC를 고유한 t에 대해 정렬 및 중복 제거
    [t_unique, idx_unique] = unique(t);
    SOC_unique = SOC(idx_unique);
    
    soc_mid_all(s) = interp1(t_unique, SOC_unique, t_mid, 'linear', 'extrap');  % 중간 시간에 해당하는 SOC
    
    % 시간 간격 dt 계산
    delta_t = [0; diff(t)];
    dt = delta_t;
    if dt(1) == 0  % 첫 번째 dt 값이 0이면
        dt(1) = dt(2);  % 두 번째 dt 값으로 대체
    end
    
    % OCV 계산 (SOC-OCV 테이블 사용)
    ocv_over_time = interp1(soc_values, ocv_values, SOC, 'linear', 'extrap');
    
    %% 4.1 W 행렬 생성
    % W 행렬 초기화
    W = zeros(length(t), n);
    for k_idx = 1:length(t)
        for i = 1:n
            if k_idx == 1
                % 이전 트립의 마지막 W_end를 사용하여 계산
                W(k_idx, i) = W_end(i) * exp(-dt(k_idx) / tau_discrete(i)) + ...
                              ik(k_idx) * (1 - exp(-dt(k_idx) / tau_discrete(i))) * delta_theta;
            else
                W(k_idx, i) = W(k_idx-1, i) * exp(-dt(k_idx) / tau_discrete(i)) + ...
                              ik(k_idx) * (1 - exp(-dt(k_idx) / tau_discrete(i))) * delta_theta;
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
    V_RC(:, 1) = V_RC_end;       % 이전 트립의 마지막 V_RC_end로 초기화
    V_est = zeros(length(t), 1);
    for k_idx = 1:length(t)
        if k_idx == 1
            for i = 1:n
                V_RC(i, k_idx) = V_RC_end(i) * exp(-dt(k_idx) / tau_discrete(i)) + ...
                                 gamma_est(i) * delta_theta * ik(k_idx) * (1 - exp(-dt(k_idx) / tau_discrete(i)));
            end
        else
            for i = 1:n
                V_RC(i, k_idx) = V_RC(i, k_idx-1) * exp(-dt(k_idx) / tau_discrete(i)) + ...
                                 gamma_est(i) * delta_theta * ik(k_idx) * (1 - exp(-dt(k_idx) / tau_discrete(i)));
            end
        end
        % 시간 k_idx에서의 V_est 계산
        V_est(k_idx) = ocv_over_time(k_idx) + R0_est * ik(k_idx) + sum(V_RC(:, k_idx));
    end
    
    % 현재 트립의 마지막 V_RC와 W를 저장하여 다음 트립에서 사용
    V_RC_end = V_RC(:, end);
    W_end = W(end, :);
    
    %% 각 트립의 W, V_RC, OCV, V_est를 저장
    W_all{s} = W;
    V_RC_all{s} = V_RC;
    OCV_all{s} = ocv_over_time;
    V_est_all{s} = V_est;
    
    %% 4.5 DRT Gamma 그래프 출력
    % Figure 1: DRT Gamma 서브플롯 (4x4)
    figure(1);
    subplot(4, 4, s);
    plot(theta_discrete, gamma_est, 'Color', c_mat(mod(s-1,9)+1, :), 'LineWidth', 1.5);
    xlabel('\theta = ln(\tau)', 'FontSize', labelFontSize);
    ylabel('\gamma', 'FontSize', labelFontSize);
    title(['DRT for Trip ', num2str(s)], 'FontSize', titleFontSize);
    legend({'\gamma'}, 'FontSize', legendFontSize, 'Location', 'best');
    set(gca, 'FontSize', axisFontSize);
    % grid on;  % 그리드 제거
    
    % R0 추정값을 그래프에 텍스트로 추가
    x_text = min(theta_discrete);  % x축의 왼쪽 끝
    y_text = max(gamma_est);       % y축의 최댓값
    text(x_text, y_text, sprintf('R₀ = %.3e Ω', R0_est_all(s)), ...
        'FontSize', 8, 'Color', 'k', 'FontWeight', 'bold', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
    
    %% 4.6 전압 비교 그래프 출력
    % Figure 2: Voltage and Current Comparison 서브플롯 (4x4)
    figure(2);
    subplot(4, 4, s);
    yyaxis left  % 왼쪽 Y축 활성화 (전압)
    plot(t, V_sd, 'Color', c_mat(mod(s-1,9)+1, :), 'LineWidth', 1, 'DisplayName', 'Measured V_{udds}');
    hold on;
    plot(t, V_est, '--', 'Color', c_mat(mod(s-1,9)+1, :), 'LineWidth', 1, 'DisplayName', 'Estimated V_{est}');
    xlabel('Time (s)', 'FontSize', labelFontSize);
    ylabel('Voltage (V)', 'FontSize', labelFontSize, 'Color', 'k');  % 왼쪽 Y축 레이블 색상 설정 (검정색)
    title(['Voltage and Current Comparison for Trip ', num2str(s)], 'FontSize', titleFontSize);
    legend('Location', 'best', 'FontSize', legendFontSize);
    
    yyaxis right  % 오른쪽 Y축 활성화 (전류)
    plot(t, ik, 'Color', 'g', 'LineWidth', 1, 'DisplayName', 'Current (A)');
    ylabel('Current (A)', 'FontSize', labelFontSize, 'Color', 'g');  % 오른쪽 Y축 레이블 색상을 초록색으로 설정
    set(gca, 'YColor', 'g');  % 오른쪽 Y축의 눈금 및 값 색상을 초록색으로 설정
    legend('Location', 'best', 'FontSize', legendFontSize);
    set(gca, 'FontSize', axisFontSize);
    hold off;
    
    %% 4.7 Trip 1에 대한 별도의 Gamma 그래프 추가
    if s == 1
        figure(5);  % 새로운 figure 생성
        set(gcf, 'Position', [100, 100, 800, 600]);  % Figure 크기 조정 (가로:800, 세로:600)
        plot(theta_discrete, gamma_est_all(1, :), 'LineWidth', 3, 'Color', c_mat(1, :));
        xlabel('$\theta = \ln(\tau \, [s])$', 'Interpreter', 'latex', 'FontSize', labelFontSize)
        ylabel('\gamma [\Omega]', 'FontSize', labelFontSize);
        title('Trip 1 : DRT ', 'FontSize', titleFontSize);
        % grid on;  % 그리드 제거
        hold on;
        
        % R0 추정값을 그래프에 텍스트로 추가
        x_text = min(theta_discrete);
        y_text = max(gamma_est_all(1, :));
        text(x_text, y_text, sprintf('R₀ = %.3e Ω', R0_est_all(1)), ...
            'FontSize', 12, 'Color', 'k', 'FontWeight', 'bold', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
        
        hold off;
        
        %% 4.8 Trip 1에 대한 I, V, V_model vs t 그래프 추가
        figure(6);  % 새로운 figure 생성
        set(gcf, 'Position', [150, 150, 800, 600]);  % Figure 크기 조정 (가로:800, 세로:600)
        
        % 왼쪽 Y축: Voltage (V_sd 및 V_est)
        yyaxis left
        plot(t, V_sd, 'Color', c_mat(1, :), 'LineWidth', 3, 'DisplayName', 'Measured V_{udds}');
        hold on;
        plot(t, V_est, '--', 'Color', c_mat(2, :), 'LineWidth', 3, 'DisplayName', 'Estimated V_{est}');
        ylabel('Voltage (V)', 'FontSize', labelFontSize, 'Color', c_mat(1, :));  % 'Voltage' 레이블 색상 설정
        
        % 오른쪽 Y축: Current (ik)
        yyaxis right
        plot(t, ik, 'Color', c_mat(3, :), 'LineWidth', 3, 'DisplayName', 'Current (A)');
        ylabel('Current (A)', 'FontSize', labelFontSize, 'Color', c_mat(3, :));  % 'Current' 레이블 색상 설정
        set(gca, 'YColor', c_mat(3, :));  % 오른쪽 Y축 눈금 및 값 색상 설정
        
        % X축 레이블 설정 (왼쪽 Y축 색상과 일치)
        xlabel('Time (s)', 'FontSize', labelFontSize);
        
        % 제목 설정
        title('I, V, V_{model} vs Time for Trip 1', 'FontSize', titleFontSize);
        
        % 범례 설정
        legend('Location', 'best', 'FontSize', legendFontSize);
        
        % 축의 숫자(틱 라벨) 폰트 크기 설정
        set(gca, 'FontSize', axisFontSize);
        
        hold off;
    end

end

%% 5. Gamma(SOC, Theta) 3D 그래프 생성
% 각 트립의 SOC 중간값에 해당하는 Gamma 값을 3차원으로 배열
% soc_mid_all: (num_trips-1) x 1
% gamma_est_all: (num_trips-1) x n

% 정렬: SOC 중간값을 기준으로 오름차순 정렬
[soc_sorted, sort_idx] = sort(soc_mid_all);
gamma_sorted = gamma_est_all(sort_idx, :);

% 그리드 생성
[SOC_grid, Theta_grid] = meshgrid(soc_sorted, theta_discrete);

% Gamma 값을 전치하여 (n x num_trips-1) 행렬로 설정
Gamma_grid = gamma_sorted';

% 3D 서피스 플롯 생성 (색상 매핑 추가)
figure(3);
surf_handle = surf(SOC_grid, Theta_grid, Gamma_grid);  
xlabel('SOC', 'FontSize', labelFontSize);
ylabel('$\theta = \ln(\tau \, [s])$', 'Interpreter', 'latex', 'FontSize', labelFontSize);
zlabel('\gamma [\Omega]', 'FontSize', labelFontSize);
title('Gamma(SOC, \theta) 3D Surface Plot', 'FontSize', titleFontSize);
colormap(jet);    % 원하는 컬러맵 설정
c = colorbar;     % colorbar 핸들을 저장
c.Label.String = 'Gamma [\Omega/s]';  % colorbar 라벨 설정
view(135, 30);    % 시각화 각도 조정
% grid on;  % 그리드 제거

alpha(0.8);
axis tight;

% SOC 축을 0에서 1 사이로 설정
xlim([0 1]);

% 축 폰트 크기 조절
set(gca, 'FontSize', axisFontSize);

%% 5.2 개별 트립에 대한 3D 라인 플롯 생성
% z 축 한계 설정
z_threshold = 0.25;

figure(4);
hold on;  

% 컬러맵 및 색상 인덱스 설정
cmap = jet;  % 컬러맵 설정
num_colors = size(cmap, 1);
soc_min = min(soc_mid_all);
soc_max = max(soc_mid_all);

for s = 1:num_trips-1
    % SOC를 컬러맵에 매핑
    color_idx = round((soc_mid_all(s) - soc_min) / (soc_max - soc_min) * (num_colors - 1)) + 1;
    color_idx = max(1, min(num_colors, color_idx));  % 인덱스 범위 제한
    
    % z_threshold를 적용하여 필터링
    gamma_data_filtered = gamma_est_all(s, :);
    gamma_data_filtered(gamma_data_filtered > z_threshold) = NaN;  % 임계값 이상은 NaN 처리
    
    % 필터링된 데이터 플롯
    plot3(repmat(soc_mid_all(s), size(theta_discrete)), theta_discrete, gamma_data_filtered, ...
          'LineWidth', 1.5, 'Color', cmap(color_idx, :));
end

xlabel('SOC', 'FontSize', labelFontSize);
ylabel('$\theta = \ln(\tau \, [s])$', 'Interpreter', 'latex', 'FontSize', labelFontSize);
zlabel('\gamma [\Omega]', 'FontSize', labelFontSize);
title('3D DRT', 'FontSize', titleFontSize);
colormap(jet);  
c = colorbar;  % Colorbar 설정
c.Label.String = 'SOC';  % Colorbar 라벨 설정
caxis([soc_min soc_max]);  % Colorbar 범위를 SOC 범위로 설정

% 축 설정
xlim([0 1]);
zlim([0, z_threshold]);  % z 축 한계를 0.25로 설정

view(135, 30);
grid on;  % 그리드 제거

set(gca, 'FontSize', axisFontSize);
hold off;


