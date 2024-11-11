clc; clear; close all;

% 원본 데이터
a = [1, 2, 3];

% 시드 설정
rng(0);  % 시드를 0으로 고정

% 노이즈 추가
noise = randn(1, 3) * 0.1;  % 노이즈 레벨 0.1
noisy_a = a + noise;

% 출력
disp('노이즈를 추가한 a:');
disp(noisy_a);
