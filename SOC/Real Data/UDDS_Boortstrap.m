clear; clc; close all;

%% 0. 폰트 크기 및 색상 매트릭스 설정
% Font size settings
axisFontSize = 14;      % 축의 숫자 크기
titleFontSize = 16;     % 제목의 폰트 크기
legendFontSize = 12;    % 범례의 폰트 크기
labelFontSize = 14;     % xlabel 및 ylabel의 폰트 크기

% Color matrix 설정
c_mat = lines(9);  % 9개의 고유한 색상 정의

%% 1. UDDS 주행 데이터 로드
% UDDS 주행 데이터를 로드합니다.
% 'udds_data.mat' 파일은 각 트립에 대한 시간(t), 전류(I), 전압(V), SOC 데이터를 포함하는 구조체 배열이어야 합니다.
load('udds_data.mat');  % 'udds_data' 구조체를 로드합니다.

%% 2. SOC-OCV 데이터 로드
% SOC-OCV 데이터를 로드합니다.
% 'soc_ocv.mat' 파일은 SOC와 OCV 값을 포함하는 2열 행렬이어야 합니다.
load('soc_ocv.mat', 'soc_ocv');
soc_values = soc_ocv(:, 1);  % SOC 값
ocv_values = soc_ocv(:, 2);  % OCV 값

%% 3. DRT 추정에 필요한 파라미터 설정
n = 41;  % 이산 요소의 개수
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
lambda = 0.104; 

% Gamma에 대한 1차 차분 행렬 L_gamma 생성
L_gamma = zeros(n-1, n);
for i = 1:n-1
    L_gamma(i, i) = -1;
    L_gamma(i, i+1) = 1;
end

% R0에 대한 정규화를 피하기 위해 L_aug 생성
L_aug = [L_gamma, zeros(n-1, 1)];

% 부트스트랩 샘플 수
N_resample = 200;

% quadprog 옵션 설정
options = optimoptions('quadprog', 'Display', 'off', 'Algorithm', 'interior-point-convex');

%% 4. 각 트립에 대한 DRT 추정 (quadprog 사용)
num_trips = length(udds_data);

% 결과 저장을 위한 배열 사전 할당
gamma_est_all = zeros(num_trips-1, n);           % 원본 데이터로부터 구한 gamma 저장
gamma_ci_lower_all = zeros(num_trips-1, n);      % 부트스트랩 신뢰구간 하한
gamma_ci_upper_all = zeros(num_trips-1, n);      % 부트스트랩 신뢰구간 상한
R0_est_all = zeros(num_trips-1, 1);
soc_mid_all = zeros(num_trips-1, 1);             % 각 트립의 중간 SOC 저장

% 부등식 제약 조건 변수 정의 (루프 밖에서 정의)
A_ineq = -eye(n+1);
b_ineq = zeros(n+1, 1);

for s = 1:num_trips-1  % 마지막 트립은 데이터가 짧으므로 제외
    fprintf('Processing Trip %d/%d...\n', s, num_trips-1);
    
    %% 4.1 현재 트립의 데이터 추출
    t_original = udds_data(s).t;       % 시간 벡터
    ik_original = udds_data(s).I;     % 전류 벡터
    V_sd_original = udds_data(s).V;   % 전압 벡터
    SOC_original = udds_data(s).SOC;   % SOC 벡터
    
    %% 4.2 각 트립의 중간 SOC 계산
    t_mid = t_original(end) / 2;  % 트립의 중간 시간
    
    % t와 SOC를 고유한 t에 대해 정렬 및 중복 제거
    [t_unique, idx_unique] = unique(t_original);
    SOC_unique = SOC_original(idx_unique);
    
    soc_mid_all(s) = interp1(t_unique, SOC_unique, t_mid, 'linear', 'extrap');  % 중간 시간에 해당하는 SOC
    
    %% 4.3 시간 간격 dt 계산
    delta_t = [0; diff(t_original)];
    dt_original = delta_t;
    if dt_original(1) == 0  % 첫 번째 dt 값이 0이면
        dt_original(1) = dt_original(2);  % 두 번째 dt 값으로 대체
    end
    
    %% 4.4 OCV 계산 (SOC-OCV 테이블 사용)
    ocv_over_time = interp1(soc_values, ocv_values, SOC_original, 'linear', 'extrap');
    
    %% 4.5 원본 데이터로부터 gamma 추정 (DRT Original)
    % W 행렬 초기화
    W_original = zeros(length(t_original), n);
    for k_idx = 1:length(t_original)
        if k_idx == 1
            for i = 1:n
                W_original(k_idx, i) = ik_original(k_idx) * (1 - exp(-dt_original(k_idx) / tau_discrete(i))) * delta_theta;
            end
        else
            for i = 1:n
                W_original(k_idx, i) = W_original(k_idx-1, i) * exp(-dt_original(k_idx) / tau_discrete(i)) + ...
                                       ik_original(k_idx) * (1 - exp(-dt_original(k_idx) / tau_discrete(i))) * delta_theta;
            end
        end
    end
    
    % R0 추정을 위한 W 행렬 확장
    W_aug_original = [W_original, ik_original(:)];  % ik_original을 열 벡터로 변환
    
    %% 4.6 y 벡터 생성
    y_original = V_sd_original - ocv_over_time;
    y_original = y_original(:);  % y를 열 벡터로 변환
    
    %% 4.7 quadprog를 사용한 제약 조건 하의 추정
    H_original = (W_aug_original' * W_aug_original + lambda * (L_aug' * L_aug));
    f_original = -W_aug_original' * y_original;
    
    % quadprog 실행
    [Theta_est_original, ~, exitflag] = quadprog(H_original, f_original, A_ineq, b_ineq, [], [], [], [], [], options);
    
    if exitflag ~= 1
        warning('Optimization did not converge for trip %d.', s);
    end
    
    % gamma와 R0 추정값 추출
    gamma_original = Theta_est_original(1:n);
    R0_est = Theta_est_original(n+1);
    
    % 추정값 저장
    gamma_est_all(s, :) = gamma_original';
    R0_est_all(s) = R0_est;
    
    %% 4.8 부트스트랩을 통한 gamma 불확실성 추정
    gamma_resample = zeros(N_resample, n);  % 부트스트랩으로 추정한 gamma 저장
    
    for b_resample = 1:N_resample
        %% 4.8.1 부트스트랩 샘플링 (복원 추출)
        idx_bootstrap = randsample(length(t_original), length(t_original), true);
        t_bootstrap = t_original(idx_bootstrap);
        ik_bootstrap = ik_original(idx_bootstrap);
        V_sd_bootstrap = V_sd_original(idx_bootstrap);
        ocv_bootstrap = ocv_over_time(idx_bootstrap);
        
        %% 4.8.2 중복된 시간 점 제거 및 정렬
        [t_bootstrap_unique, unique_idx] = unique(t_bootstrap);
        ik_bootstrap_unique = ik_bootstrap(unique_idx);
        V_sd_bootstrap_unique = V_sd_bootstrap(unique_idx);
        ocv_bootstrap_unique = ocv_bootstrap(unique_idx);
        
        [t_bootstrap_sorted, sort_idx] = sort(t_bootstrap_unique);
        ik_bootstrap_sorted = ik_bootstrap_unique(sort_idx);
        V_sd_bootstrap_sorted = V_sd_bootstrap_unique(sort_idx);
        ocv_bootstrap_sorted = ocv_bootstrap_unique(sort_idx);
        
        %% 4.8.3 시간 간격 dt_bootstrap 계산
        if length(t_bootstrap_sorted) > 1
            dt_bootstrap = [t_bootstrap_sorted(1); diff(t_bootstrap_sorted)];
            if dt_bootstrap(1) == 0
                dt_bootstrap(1) = dt_bootstrap(2);
            end
        else
            dt_bootstrap = dt_original(1);  % 기본값으로 설정
        end
        
        %% 4.8.4 W 행렬 구성 (부트스트랩 데이터로)
        W_bootstrap = zeros(length(t_bootstrap_sorted), n);  % W 행렬 초기화
        for k_idx = 1:length(t_bootstrap_sorted)
            if k_idx == 1
                for i = 1:n
                    W_bootstrap(k_idx, i) = ik_bootstrap_sorted(k_idx) * (1 - exp(-dt_bootstrap(k_idx) / tau_discrete(i))) * delta_theta;
                end
            else
                for i = 1:n
                    W_bootstrap(k_idx, i) = W_bootstrap(k_idx-1, i) * exp(-dt_bootstrap(k_idx) / tau_discrete(i)) + ...
                                           ik_bootstrap_sorted(k_idx) * (1 - exp(-dt_bootstrap(k_idx) / tau_discrete(i))) * delta_theta;
                end
            end
        end
        
        % R0 추정을 위한 W 행렬 확장
        W_aug_bootstrap = [W_bootstrap, ik_bootstrap_sorted(:)];  % ik_bootstrap_sorted을 열 벡터로 변환
        
        %% 4.8.5 y_bootstrap 벡터 생성
        y_bootstrap = V_sd_bootstrap_sorted - ocv_bootstrap_sorted;
        y_bootstrap = y_bootstrap(:);  % y를 열 벡터로 변환
        
        %% 4.8.6 quadprog를 사용한 제약 조건 하의 추정
        H_bootstrap = (W_aug_bootstrap' * W_aug_bootstrap + lambda * (L_aug' * L_aug));
        f_bootstrap = -W_aug_bootstrap' * y_bootstrap;
        
        % quadprog 실행
        [Theta_est_bootstrap, ~, exitflag] = quadprog(H_bootstrap, f_bootstrap, A_ineq, b_ineq, [], [], [], [], [], options);
        
        if exitflag ~= 1
            warning('Optimization did not converge for bootstrap sample %d in trip %d.', b_resample, s);
        end
        
        % gamma 추정값 추출
        gamma_bootstrap_b = Theta_est_bootstrap(1:n);
        
        % 부트스트랩으로 구한 gamma 저장
        gamma_resample(b_resample, :) = gamma_bootstrap_b';
    end
    
   %% 4.9 gamma 불확실성 구하기 (수정된 부분)
    %% 4.9 gamma 불확실성 구하기 (수정된 부분)
    % gamma_resample로부터 직접 5% 및 95% 백분위수 계산
    gamma_ci_lower = prctile(gamma_resample, 5, 1);  % 5% 백분위수
    gamma_ci_upper = prctile(gamma_resample, 95, 1); % 95% 백분위수
    
    % 오차막대 계산 (gamma_original과의 차이)
    gamma_diff_lower = gamma_original' - gamma_ci_lower;
    gamma_diff_upper = gamma_ci_upper - gamma_original';
    
    % 불확실성 저장
    gamma_ci_lower_all(s, :) = gamma_diff_lower;
    gamma_ci_upper_all(s, :) = gamma_diff_upper;

    %% 4.10 DRT Gamma 그래프 출력 (errorbar를 사용하여 불확실성 표현)
    % Figure 1: DRT Gamma 서브플롯 (4x4)
    figure(1);
    subplot(4, 4, s);
    hold on;
    
    % gamma_original과 부트스트랩 불확실성을 errorbar로 표현
    errorbar(theta_discrete, gamma_original, -gamma_diff_lower, gamma_diff_upper, ...
        'Color', c_mat(mod(s-1,9)+1, :), 'LineWidth', 1.5, 'DisplayName', '\gamma with CI');
    
    xlabel('\theta = ln(\tau)', 'FontSize', labelFontSize);
    ylabel('\gamma', 'FontSize', labelFontSize);
    title(['DRT for Trip ', num2str(s)], 'FontSize', titleFontSize);
    legend({'\gamma with CI'}, 'FontSize', legendFontSize, 'Location', 'best');
    set(gca, 'FontSize', axisFontSize);
    hold off;
    
    % R0 추정값을 그래프에 텍스트로 추가
    x_text = min(theta_discrete);  % Set to the leftmost part of the x-axis
    y_text = max(gamma_original);   % Set to the top part of the y-axis
    text(x_text, y_text, sprintf('R₀ = %.3e Ω', R0_est), ...
        'FontSize', 8, 'Color', 'k', 'FontWeight', 'bold', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
    
    %% 4.11 전압 비교 그래프 출력 (전류 프로파일 추가 및 레이블 색상 변경)
    % Figure 2: Voltage and Current Comparison 서브플롯 (4, 4)
    figure(2);
    subplot(4, 4, s);
    yyaxis left  % 왼쪽 Y축 활성화 (전압)
    plot(t_original, V_sd_original, 'Color', c_mat(mod(s-1,9)+1, :), 'LineWidth', 1, 'DisplayName', 'Measured V_{udds}');
    hold on;
    
    % V_est 계산 (검증용)
    V_RC = zeros(n, length(t_original));  % 각 요소의 전압
    V_est = zeros(length(t_original), 1);
    for k_idx = 1:length(t_original)
        if k_idx == 1
            for i = 1:n
                V_RC(i, k_idx) = gamma_original(i) * delta_theta * ik_original(k_idx) * (1 - exp(-dt_original(k_idx) / tau_discrete(i)));
            end
        else
            for i = 1:n
                V_RC(i, k_idx) = V_RC(i, k_idx-1) * exp(-dt_original(k_idx) / tau_discrete(i)) + ...
                                 gamma_original(i) * delta_theta * ik_original(k_idx) * (1 - exp(-dt_original(k_idx) / tau_discrete(i)));
            end
        end
        % 시간 k_idx에서의 V_est 계산
        V_est(k_idx) = ocv_over_time(k_idx) + R0_est * ik_original(k_idx) + sum(V_RC(:, k_idx));
    end
    
    plot(t_original, V_est, '--', 'Color', c_mat(mod(s-1,9)+1, :), 'LineWidth', 1, 'DisplayName', 'Estimated V_{est}');
    xlabel('Time (s)', 'FontSize', labelFontSize);
    ylabel('Voltage (V)', 'FontSize', labelFontSize, 'Color', 'k');  % 왼쪽 Y축 레이블 색상 설정 (검정색)
    title(['Voltage and Current Comparison for Trip ', num2str(s)], 'FontSize', titleFontSize);
    legend({'Measured V_{udds}', 'Estimated V_{est}'}, 'FontSize', legendFontSize, 'Location', 'best');
    
    yyaxis right  % 오른쪽 Y축 활성화 (전류)
    plot(t_original, ik_original, 'Color', 'g', 'LineWidth', 1, 'DisplayName', 'Current (A)');
    ylabel('Current (A)', 'FontSize', labelFontSize, 'Color', 'g');  % 오른쪽 Y축 레이블 색상을 초록색으로 설정
    set(gca, 'YColor', 'g');  % 오른쪽 Y축의 눈금 및 값 색상을 초록색으로 설정
    legend({'Current (A)'}, 'FontSize', legendFontSize, 'Location', 'best');
    set(gca, 'FontSize', axisFontSize);
    hold off;
    
    %% 4.12 Trip 1에 대한 별도의 Gamma 그래프 및 I-V 비교 그래프 추가
    if s == 1
        % Figure 5: Trip 1의 DRT Gamma 그래프 (errorbar를 사용하여 불확실성 표현)
        figure(5);  % 새로운 figure 생성
        set(gcf, 'Position', [100, 100, 800, 600]);  % Figure 크기 조정 (가로:800, 세로:600)
        hold on;
        
        % gamma_original과 부트스트랩 불확실성을 errorbar로 표현
        errorbar(theta_discrete, gamma_est_all(1, :), -gamma_ci_lower_all(1, :), gamma_ci_upper_all(1, :), ...
            'LineWidth', 2, 'Color', c_mat(1, :), 'DisplayName', '\gamma with CI');
        
        xlabel('$\theta = \ln(\tau \, [s])$', 'Interpreter', 'latex', 'FontSize', labelFontSize);
        ylabel('\gamma [\Omega]', 'FontSize', labelFontSize);
        title('Trip 1 : DRT with Error Bars', 'FontSize', titleFontSize);
        legend({'\gamma with CI'}, 'FontSize', legendFontSize, 'Location', 'best');
        set(gca, 'FontSize', axisFontSize);
        hold off;
        
        % R0 추정값을 그래프에 텍스트로 추가
        x_text = min(theta_discrete);  % Set to the leftmost part of the x-axis
        y_text = max(gamma_est_all(1, :));   % Set to the top part of the y-axis
        text(x_text, y_text, sprintf('R₀ = %.3e Ω', R0_est_all(s)), ...
            'FontSize', 12, 'Color', 'k', 'FontWeight', 'bold', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
        
        %% Figure 6: Trip 1의 I, V, V_model vs t 그래프
        figure(6);  % 새로운 figure 생성
        set(gcf, 'Position', [150, 150, 800, 600]);  % Figure 크기 조정 (가로:800, 세로:600)
        
        % 왼쪽 Y축: Voltage (V_sd 및 V_est)
        yyaxis left
        plot(t_original, V_sd_original, 'Color', c_mat(1, :), 'LineWidth', 2, 'DisplayName', 'Measured V_{udds}');
        hold on;
        plot(t_original, V_est, '--', 'Color', c_mat(2, :), 'LineWidth', 2, 'DisplayName', 'Estimated V_{est}');
        ylabel('Voltage (V)', 'FontSize', labelFontSize, 'Color', c_mat(1, :));  % 'Voltage' 레이블 색상 설정
        
        % 오른쪽 Y축: Current (ik)
        yyaxis right
        plot(t_original, ik_original, 'Color', c_mat(3, :), 'LineWidth', 2, 'DisplayName', 'Current (A)');
        ylabel('Current (A)', 'FontSize', labelFontSize, 'Color', c_mat(3, :));  % 'Current' 레이블 색상 설정
        set(gca, 'YColor', 'g');  % 오른쪽 Y축의 눈금 및 값 색상을 초록색으로 설정
        
        % X축 레이블 설정
        xlabel('Time (s)', 'FontSize', labelFontSize);
        
        % 제목 설정
        title('I, V, V_{model} vs Time for Trip 1', 'FontSize', titleFontSize);
        
        % 범례 설정
        legend({'Measured V_{udds}', 'Estimated V_{est}', 'Current (A)'}, 'FontSize', legendFontSize, 'Location', 'best');
        
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

alpha(0.8);
axis tight;

% SOC 축을 0에서 1 사이로 설정
xlim([0 1]);

% 축 폰트 크기 조절
set(gca, 'FontSize', axisFontSize);

%% 5.2 개별 트립에 대한 3D 라인 플롯 생성
% Set the z threshold
z_threshold = 0.25;

figure(4);
hold on;  

% Use colormap and color index as before
cmap = jet;  % 컬러맵 설정
num_colors = size(cmap, 1);
soc_min = min(soc_mid_all);
soc_max = max(soc_mid_all);

for s = 1:num_trips-1
    % Map SOC to colormap
    color_idx = round((soc_mid_all(s) - soc_min) / (soc_max - soc_min) * (num_colors - 1)) + 1;
    color_idx = max(1, min(num_colors, color_idx));  % 인덱스 범위 제한
    
    % Apply z threshold filter
    gamma_data_filtered = gamma_est_all(s, :);
    gamma_data_filtered(gamma_data_filtered > z_threshold) = NaN;  % Filter values above threshold
    
    % Plot filtered data
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

% SOC axis setting
xlim([0 1]);
zlim([0, z_threshold]);  % z axis limit to 0.25

view(135, 30);
grid on;  % 그리드 표시

set(gca, 'FontSize', axisFontSize);
hold off;

%% 6. 결과 저장
save('gamma_data.mat', 'gamma_sorted', 'soc_sorted', 'theta_discrete', 'R0_est_all', 'soc_mid_all');
% save('soc_ocv_data.mat', 'soc_values', 'ocv_values');  % 필요 시 주석 해제하여 저장

