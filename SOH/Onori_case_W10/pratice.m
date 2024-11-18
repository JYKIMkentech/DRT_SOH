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
load('G:\공유 드라이브\BSL_Onori\Cycling_tests\Trips_Aging_1_W10.mat');  % 'Trips' 구조체를 로드합니다.

col_cell_label = {'W3','W4','W5','W7','W8','W9','W10','G1','V4','V5'};

%% 2. SOC-OCV 데이터 로드
% 'soc_ocv_cap' 데이터를 로드합니다.
load('RPT_All_soc_ocv_cap.mat', 'soc_ocv_cap');

% load('soc_ocv.mat', 'soc_ocv');
% soc_values = soc_ocv(:, 1);  % SOC 값
% ocv_values = soc_ocv(:, 2);  % OCV 값

% % SOC와 OCV 값 추출
soc_values = soc_ocv_cap{1,7}(:, 1);  % SOC 값 (0 ~ 1)
ocv_values = soc_ocv_cap{1,7}(:, 2);  % OCV 값 (V)

% 배터리 용량 추출 (Capacity 열의 최대값)
Q_batt = max(soc_ocv_cap{1,7}(:, 3));  % Ah 단위

% OCV 값 정렬
[sorted_ocv, sortIdx] = sort(ocv_values);
sorted_soc = soc_values(sortIdx);

% 고유한 OCV 값 (첫 번째 발생값 선택)
[unique_ocv, uniqueIdx] = unique(sorted_ocv, 'first');
unique_soc = sorted_soc(uniqueIdx);

% 또는 마지막 발생값 선택
% [unique_ocv, uniqueIdx] = unique(sorted_ocv, 'last');
% unique_soc = sorted_soc(uniqueIdx);

% 보간 수행
soc0 = interp1(unique_ocv, unique_soc, Trips(1).V(end), 'linear', 'extrap');
