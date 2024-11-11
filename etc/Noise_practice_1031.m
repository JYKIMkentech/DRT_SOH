clc; clear; close all;

%% 1. 데이터 로드
% udds_data.mat 파일이 현재 작업 디렉토리에 있어야 합니다.
load('udds_data.mat');

I = udds_data(1).I;        % 전류 (A)
V = udds_data(1).V;        % 전압 (V)
t = udds_data(1).t;        % 시간 (초)
CC_SOC = udds_data(1).SOC; % Coulomb 카운팅을 통한 SOC (%)

%% 2. 노이즈 상태 벡터 구성

% 2.1 전류 데이터의 평균 절대값 계산
I_mean = mean(abs(I)); % 평균 전류의 절대값 (A)

% 2.2 평균 전류의 1% 계산하여 노이즈 최대값 설정
n_max = I_mean * 0.1; % 노이즈 최대값 (A)

% 2.3 노이즈 상태 벡터 정의
noise_states = linspace(-n_max, n_max, 5); % 노이즈 상태 (-n_max ~ n_max 사이의 5개 값)

% 2.5 노이즈의 표준편차 설정 (노이즈 상태 내의 작은 변동)
sigma = n_max * 0.1; % 노이즈 최대값의 10%

%% 3. 전이 확률 행렬 정의

% 전이 확률 행렬 정의 (5x5 행렬)
P = [0.7, 0.15, 0.1, 0.05, 0;
     0.15, 0.7, 0.1, 0.05, 0;
     0.1, 0.1, 0.6, 0.1, 0.1;
     0.05, 0.1, 0.1, 0.7, 0.05;
     0, 0.05, 0.1, 0.15, 0.7];

% 전이 확률 행렬이 유효한지 확인 (각 행의 합이 1인지 확인)
assert(all(abs(sum(P,2) - 1) < 1e-6), '전이 확률 행렬의 각 행의 합이 1이 아닙니다.');

%% 4. 노이즈 생성

% 4.1 노이즈 벡터 초기화
n = zeros(size(I));

% 4.2 초기 상태 설정 (노이즈 상태 벡터의 인덱스)
current_state = 3; % 초기 상태는 0 A (인덱스 3)

% 4.3 노이즈 생성 루프 (PDF 기반 샘플링)
for k = 1:length(I)
    % 현재 상태의 전이 확률 추출
    transition_prob = P(current_state, :);
    
    % 다음 상태 샘플링 (PDF 기반)
    next_state = randsample(noise_states, 1, true, transition_prob);
    
    % 노이즈 계산 (상태 값에 작은 변동 추가)
    n(k) = next_state + sigma * randn();
    
    % 다음 상태의 인덱스 찾기
    current_state = find(noise_states == next_state, 1);
end

% 4.4 노이즈가 적용된 전류 생성
I_noisy = I + n;

%% 5. SOC 계산

% 5.1 파라미터 설정
C_nom = 2.9 * 3600 ;                   % 배터리의 정격 용량 (Ah)
t_hours = t ;            % 시간을 시간 단위로 변환 (초 -> 시간)
SOC_initial = 0.9901;          % 초기 SOC (비율, 1은 100%)

% 5.2 SOC 계산 (cumtrapz 함수 사용)
% 충전 시 SOC가 증가하고 방전 시 SOC가 감소하도록 수식을 수정
% 방전 시 전류가 양수일 경우 SOC 감소, 충전 시 음수일 경우 SOC 증가
SOC_original = SOC_initial + cumtrapz(t_hours, I) / C_nom;
SOC_noisy = SOC_initial + cumtrapz(t_hours, I_noisy) / C_nom;

%% 6. 결과 시각화

% 6.1 전류 비교
figure;
subplot(3,1,1); % 첫 번째 subplot을 첫 번째 위치에 설정
plot(t, I, 'b', 'LineWidth', 1.5); hold on;
plot(t, I_noisy, 'r', 'LineWidth', 1.5);
xlabel('시간 (초)');
ylabel('전류 (A)');
title('원래 전류 vs. Markov noise가 적용된 전류');
legend('원래 전류', 'Markov noise 적용 전류');
grid on;

% 6.2 SOC 비교
subplot(3,1,2); % 두 번째 subplot을 두 번째 위치에 설정
%plot(t, CC_SOC, 'k', 'LineWidth', 1.5); hold on; % CC_SOC를 그대로 사용
hold on
plot(t, SOC_original , 'b-', 'LineWidth', 1.5); % 비율을 퍼센트로 변환
plot(t, SOC_noisy , 'r--', 'LineWidth', 1.5);
xlabel('시간 (초)');
ylabel('SOC (%)');
title('True SOC vs. Markov noise가 적용된 CC SOC');
legend( 'True SOC', 'Markov noise 적용한 CC SOC');
grid on;

% 6.3 SOC 차이 그래프 추가
subplot(3,1,3); % 세 번째 subplot을 세 번째 위치에 설정
SOC_difference = (SOC_original - SOC_noisy ) * 100; % 퍼센트 단위로 변환
plot(t, SOC_difference, 'm', 'LineWidth', 1.5);
xlabel('시간 (초)');
ylabel('SOC 차이 (%)');
title('Markov noise가 적용된 SOC와 원래 SOC의 차이');
legend('SOC 차이');
grid on;

% % 6.4 노이즈 히스토그램 (선택 사항)
% figure;
% histogram(n, 50);
% xlabel('노이즈 값 (A)');
% ylabel('빈도');
% title('마르코프 체인 노이즈 분포');
% grid on;
