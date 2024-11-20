clc; clear; close all;

load('G:\공유 드라이브\BSL_Onori\Cycling_tests\Processed_1\W10.mat');
load('G:\공유 드라이브\BSL_Onori\diagnostic_tests\_processed_mat\capacity_test.mat');

chg_cap = ch_cap_full_vec_M1_NMC25degC;
dis_cap = dis_cap_full_vec_M1_NMC25degC;
t = t_full_vec_M1_NMC25degC / 3600; % 시간 단위를 초에서 시간으로 변환
I = I_full_vec_M1_NMC25degC;
V = V_full_vec_M1_NMC25degC;

figure(1)

% Charging Capacity
subplot(4,1,1)
plot(t, chg_cap);
xlim([0 15]);
title('Charging Capacity over Time');
xlabel('Time (hours)');
ylabel('Charging Capacity (Ah)');
grid on;

% Discharging Capacity
subplot(4,1,2)
plot(t, dis_cap);
xlim([0 15]);
title('Discharging Capacity over Time');
xlabel('Time (hours)');
ylabel('Discharging Capacity (Ah)');
grid on;

% Voltage
subplot(4,1,3)
plot(t, V);
xlim([0 15]);
title('Voltage over Time');
xlabel('Time (hours)');
ylabel('Voltage (V)');
grid on;

% Current
subplot(4,1,4)
plot(t, I);
xlim([0 15]);
title('Current over Time');
xlabel('Time (hours)');
ylabel('Current (A)');
grid on;




