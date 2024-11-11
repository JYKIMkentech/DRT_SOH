clc;clear;close all

load('optimized_params_struct_final.mat')

soc_values = [1, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05];
R0_values = [optimized_params_struct.R0];
R1_values = [optimized_params_struct.R1];
C_values = [optimized_params_struct.C];

SOC_values = [optimized_params_struct.SOC];
Crate_values = [optimized_params_struct.Crate];

unique_SOC = unique(SOC_values);
unique_Crate = unique(Crate_values);

[X, Y] = meshgrid(unique_SOC, unique_Crate);
[Xq, Yq] = meshgrid(linspace(min(unique_SOC), max(unique_SOC), 100), linspace(min(unique_Crate), max(unique_Crate), 100));
R0_matrix = griddata(SOC_values, Crate_values, R0_values, Xq, Yq, 'cubic');

figure;
surf(Xq, Yq, R0_matrix);
xlabel('SOC');
ylabel('Crate');
zlabel('R0');
title('R0 vs SOC and Crate');
shading interp;
grid on;

zlim([0, 0.05]);

figure;
contourf(Xq, Yq, R0_matrix, 20);
xlabel('SOC');
ylabel('Crate');
title('Contour of R0 vs SOC and Crate');
colorbar;
grid on;

[Xq_R1, Yq_R1] = meshgrid(linspace(min(unique_SOC), max(unique_SOC), 100), linspace(min(unique_Crate), max(unique_Crate), 100));
R1_matrix = griddata(SOC_values, Crate_values, R1_values, Xq_R1, Yq_R1, 'cubic');

figure;
surf(Xq_R1, Yq_R1, R1_matrix);
xlabel('SOC');
ylabel('Crate');
zlabel('R1');
title('R1 vs SOC and Crate');
shading interp;
grid on;

figure;
contourf(Xq_R1, Yq_R1, R1_matrix, 20);
xlabel('SOC');
ylabel('Crate');
title('Contour of R1 vs SOC and Crate');
colorbar;
grid on;

[Xq_C, Yq_C] = meshgrid(linspace(min(unique_SOC), max(unique_SOC), 100), linspace(min(unique_Crate), max(unique_Crate), 100));
C_matrix = griddata(SOC_values, Crate_values, C_values, Xq_C, Yq_C, 'cubic');

figure;
surf(Xq_C, Yq_C, C_matrix);
xlabel('SOC');
ylabel('Crate');
zlabel('C');
title('C vs SOC and Crate');
shading interp;
grid on;

figure;
contourf(Xq_C, Yq_C, C_matrix, 20);
xlabel('SOC');
ylabel('Crate');
title('Contour of C vs SOC and Crate');
colorbar;
grid on;

% R0, R1, C 평균값 계산 (이상치 제거 후)
R0_mean = mean([optimized_params_struct.R0]);
R1_mean = mean([optimized_params_struct.R1]);
C_mean = mean([optimized_params_struct.C]);

% 결과 출력
fprintf('R0의 평균 값: %.6f\n', R0_mean);
fprintf('R1의 평균 값: %.6f\n', R1_mean);
fprintf('C의 평균 값 : %.6f\n', C_mean);
