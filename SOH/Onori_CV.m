clc; clear; close all;

%% 0. 폰트 크기 및 색상 매트릭스 설정
% 폰트 크기 설정
axisFontSize = 14;      % 축의 숫자 크기
titleFontSize = 16;     % 제목의 폰트 크기
legendFontSize = 12;    % 범례의 폰트 크기
labelFontSize = 14;     % xlabel 및 ylabel의 폰트 크기

% 색상 매트릭스 설정
c_mat = lines(9);  % 9개의 고유한 색상 정의

%% 1. 트립 데이터 로드
% 'Trips' 구조체를 로드합니다.
load('G:\공유 드라이브\BSL_Onori\Cycling_tests\Trips_Aging_14_W10.mat');  % 'Trips' 구조체를 로드합니다.

col_cell_label = {'W3','W4','W5','W7','W8','W9','W10','G1','V4','V5'};

%% 2. SOC-OCV 데이터 로드
% 'soc_ocv_cap' 데이터를 로드합니다.
load('RPT_All_soc_ocv_cap.mat', 'soc_ocv_cap');

% SOC와 OCV 값 추출
soc_values = soc_ocv_cap{14,7}(:, 1);  % SOC 값 (0 ~ 1)
ocv_values = soc_ocv_cap{14,7}(:, 2);  % OCV 값 (V)

% 배터리 용량 추출 (Ah 단위)
Q_batt = 4.47;  % 배터리 용량
SOC0 = 0.73;    % 초기 SOC

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

% 정규화 파라미터 람다 값 범위 설정
lambda_values = logspace(-14, -4 , 10);  % 필요에 따라 조정 가능

% Gamma에 대한 1차 차분 행렬 L_gamma 생성
L_gamma = zeros(n-1, n);
for i = 1:n-1
    L_gamma(i, i) = -1;
    L_gamma(i, i+1) = 1;
end

% R0에 대한 정규화를 피하기 위해 L_aug 생성
L_aug = [L_gamma, zeros(n-1, 1)];

%% 4. 각 사이클의 데이터 준비
num_cycles = length(Trips);

for s = 4:num_cycles
    % 현재 사이클의 데이터 추출
    t = Trips(s).t;
    ik = Trips(s).I;
    V_sd = Trips(s).V;   
end


