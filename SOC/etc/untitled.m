clc; clear; close all;

% Load UDDS data
load('C:\Users\deu04\OneDrive\바탕 화면\wykht8y7tg-1\Panasonic 18650PF Data\Panasonic 18650PF Data\25degC\Drive cycles\03-21-17_00.29 25degC_UDDS_Pan18650PF.mat');
udds_current = meas.Current; % UDDS 전류 데이터
udds_voltage = meas.Voltage; % UDDS 전압 데이터
udds_time = meas.Time; % UDDS 시간 데이터

% Load SOC-OCV structure
load('soc_ocv.mat', 'soc_ocv'); %c/20에서 가져온 soc-ocv lookup table
soc_values = soc_ocv(:, 1); % soc
ocv_values = soc_ocv(:, 2); % ocv

% Initialization
dt = mean(diff(udds_time));
last_time = udds_time(end);
t = udds_time;
Nsamples = length(t);

Case = "DHG"; % Assuming discharge for now, change if necessary

% Initialize cell data
Cell_Data = Init_Cell(Case, dt);

% Initialize arrays for storing results
Estimation = zeros(Nsamples, 2); % Estimated SOC and V1
EKFVt = zeros(Nsamples, 1); % Predicted Vt
Measurement = zeros(Nsamples, 1); % Measured Vt
ActualSOC = zeros(Nsamples, 1); % Reference SOC
AC_SOC = zeros(Nsamples, 1); % SOC by Ampere Counting
Current_Nominal = zeros(Nsamples, 1); % Nominal Current
Current_Noise = zeros(Nsamples, 1); % Noisy Current

% Initial values
Init = 0;

for k = 1:Nsamples
    if Init == 0
        % Assign initial values
        Estimation(k, :) = Cell_Data.esti_init;
        ActualSOC(k) = Estimation(k, 1);
        AC_SOC(k) = Estimation(k, 1);
        EKFVt(k) = udds_voltage(1);
        Measurement(k) = udds_voltage(1);
        Init = 1;
    else
        % Get experimental data
        Temp = GetExperData(k, Cell_Data, dt, Case);
        Cell_Data.Vt = Temp.Vt;
        Cell_Data.V1 = Temp.V1;
        Cell_Data.ik_noise = Temp.ik_noise;

        % Estimate SOC and V1 using EKF
        [SOC_k, V1_k] = SOCEKF(Cell_Data, dt, Case);
        Estimation(k, :) = [SOC_k V1_k];

        % Calculate actual SOC using Ampere Counting
        ActualSOC(k) = Amphere_Counting(ActualSOC(k-1), dt, Cell_Data, Cell_Data.ik_nominal);
        AC_SOC(k) = Amphere_Counting(AC_SOC(k-1), dt, Cell_Data, Cell_Data.ik_noise);

        % Store predicted Vt
        EKFVt(k) = hx(Cell_Data, SOC_k, Case);
        Measurement(k) = Cell_Data.Vt;
        Current_Nominal(k) = Cell_Data.ik_nominal;
        Current_Noise(k) = Cell_Data.ik_noise;
    end
end

% Plot results
figure;
subplot(2, 2, 1);
hold on;
plot(t, ActualSOC, 'g');
plot(t, Estimation(:, 1), 'r');
plot(t, AC_SOC, 'b');
legend('Actual SOC', 'EKF SOC', 'AC SOC');
xlabel('Time [s]');
ylabel('SOC [%]');
xlim([0 last_time]);

subplot(2, 2, 2);
hold on;
plot(t, EKFVt, 'r');
plot(t, Measurement, 'g');
legend('Estimated Vt', 'Measured Vt');
xlabel('Time [s]');
ylabel('Voltage [V]');
xlim([0 last_time]);

subplot(2, 2, 3);
plot(t, abs(ActualSOC - Estimation(:, 1)) ./ ActualSOC * 100, 'r');
hold on;
plot(t, abs(ActualSOC - AC_SOC) ./ ActualSOC * 100, 'b');
xlabel('Time [s]');
ylabel('Error Rate [%]');
legend('EKF Error Rate', 'AC Error Rate');

subplot(2, 2, 4);
hold on;
plot(t, Current_Noise, 'b');
xlabel('Time [s]');
ylabel('Current [A]');
legend('Current');

% Utility Functions

% Init_Cell function
function cell = Init_Cell(Case, dt)
    coulomb_effi = 1; % Coulombic Efficiency
    
    if Case == "DHG"
        SOC_Initial = OCV_SOC_LUT_0_01C(Discharge_Data_0_1C(1), "DHG", "SOC");
        ik_nominal = 0.41; % Discharge Current
        Cn_Nominal = 4.32019; % Nominal Capacity

        % ECM Parameters
        R0 = 0.001884314;
        R1 = 0.045801322;
        C1 = 4846.080679;

    elseif Case == "CHG"
        SOC_Initial = OCV_SOC_LUT_0_01C(Charge_Data_0_1C(1005), "CHG", "SOC");
        ik_nominal = -0.41; % Charge Current
        Cn_Nominal = 4.07611;

        % ECM Parameters
        R0 = 0.00005884314;
        R1 = 0.01145801322;
        C1 = 4846.080679;
    end

    Cn = (Cn_Nominal * 3600) / 100; % Convert to time domain

    V1_Initial = ik_nominal * R1 * (1 - exp(-dt / (R1 * C1)));

    esti_init = [SOC_Initial V1_Initial];

    cell.SOC_Initial = SOC_Initial;
    cell.V1_Initial = V1_Initial;
    cell.coulomb_effi = coulomb_effi;
    cell.ik_nominal = ik_nominal;
    cell.Cn = Cn;
    cell.R0 = R0;
    cell.R1 = R1;
    cell.C1 = C1;
    cell.esti_init = esti_init;
end

% GetExperData function
function cell = GetExperData(k, Cell_Data, dt, Case)
    persistent ik_nominal ik_before ik_noise V1 V1_before Init_Cell

    if isempty(Init_Cell)
        V1_before = Cell_Data.V1_Initial;
        ik_before = Cell_Data.ik_nominal;
        ik_nominal = ik_before;
        Init_Cell = 1;
    end

    V1 = V1_before * exp(-dt / (Cell_Data.R1 * Cell_Data.C1)) + Cell_Data.R1 * (1 - exp(-dt / (Cell_Data.R1 * Cell_Data.C1))) * ik_before;

    if Case == "CHG"
        cell.Vt = Charge_Data_0_1C(k + 1005);
    elseif Case == "DHG"
        cell.Vt = Discharge_Data_0_1C(k);
    end

    ik_noise = ik_nominal + (1.5 * rand - 0.75);

    cell.V1 = V1;
    cell.ik_nominal = ik_nominal;
    cell.ik_noise = ik_noise;

    V1_before = V1;
    ik_before = ik_noise;
end

% fx function
function x = fx(xk_before, dt, Cell_Data)
    A = [1 0;
         0 exp(-dt / (Cell_Data.R1 * Cell_Data.C1))];
    B = [-Cell_Data.coulomb_effi * dt / Cell_Data.Cn;
          Cell_Data.R1 * (1 - exp(-dt / (Cell_Data.R1 * Cell_Data.C1)))];

    x = A * xk_before + B * Cell_Data.ik_noise;
end

% hx function
function z_predict = hx(Cell_Data, SOC, Case)
    if SOC >= 100
        SOC = 100;
    elseif SOC <= 0
        SOC = 0;
    end

    OCV_LUT = OCV_SOC_LUT_0_01C(SOC, Case, "OCV");

    z_predict = OCV_LUT - Cell_Data.V1 - Cell_Data.R0 * Cell_Data.ik_noise;
end

% Hk function
function H_k = Hk(SOC_predic, SOC_before, Case)
    if SOC_predic >= 100
        SOC_predic = 100;
    elseif SOC_predic <= 0
        SOC_predic = 0;
    end

    OCV_LUT = OCV_SOC_LUT_0_01C(SOC_predic, Case, "OCV");
    OCV_LUT_before = OCV_SOC_LUT_0_01C(SOC_before, Case, "OCV");

    H_k = [(OCV_LUT - OCV_LUT_before) / (SOC_predic - SOC_before) -1];

    if SOC_predic == SOC_before

                H_k = [0 -1];
    end
end

% SOCEKF function
function [SOC_k, V1_k] = SOCEKF(Cell_Data, dt, Case)
    persistent F Q R P H K P_before
    persistent x_esti x_esti_before Init_EKF

    if isempty(Init_EKF)
        F = [1 0;
             0 exp(-dt / (Cell_Data.R1 * Cell_Data.C1))];

        if Case == "CHG"
            Q = [0.0000001 0;
                 0 0.0000001];
            R = 8000.0;
            P_before = [3000 0;
                        0 3000];
        elseif Case == "DHG"
            Q = [0.000001 0;
                 0 0.000001];
            R = 8000.0;
            P_before = [5000 0;
                        0 5000];
        end
        x_esti_before = Cell_Data.esti_init';
        Init_EKF = 1;
    end

    % Predict Step
    x_predic = fx(x_esti_before, dt, Cell_Data);
    P_predic = F * P_before * F' + Q;

    % Get Linearized Measurement Matrix H
    H = Hk(x_predic(1), x_esti_before(1), Case);

    % Update Step
    K = P_predic * H' / (H * P_predic * H' + R);
    x_esti = x_predic + K * (Cell_Data.Vt - hx(Cell_Data, x_predic(1), Case));
    P = P_predic - K * H * P_predic;

    SOC_k = x_esti(1);
    V1_k = x_esti(2);

    x_esti_before = x_esti;
    P_before = P;
end

% OCV_SOC_LUT_0_01C function
function Output = OCV_SOC_LUT_0_01C(Input, Case, Request)
    % Example OCV-SOC LUT data. Replace with actual data from your dataset.
    DHG_OCV = [4.2, 4.1, 4.0, 3.9, 3.8, 3.7, 3.6, 3.5, 3.4, 3.3];
    DHG_SOC = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10];
    
    if strcmp(Request, "OCV")
        if strcmp(Case, "DHG")
            Output = interp1(DHG_SOC, DHG_OCV, Input, 'linear', 'extrap');
        elseif strcmp(Case, "CHG")
            % Add charge data here if different
            Output = interp1(DHG_SOC, DHG_OCV, Input, 'linear', 'extrap');
        end
    elseif strcmp(Request, "SOC")
        if strcmp(Case, "DHG")
            Output = interp1(DHG_OCV, DHG_SOC, Input, 'linear', 'extrap');
        elseif strcmp(Case, "CHG")
            % Add charge data here if different
            Output = interp1(DHG_OCV, DHG_SOC, Input, 'linear', 'extrap');
        end
    end
end

% Amphere_Counting function
function soc = Amphere_Counting(soc_prev, dt, Cell_Data, current)
    soc = soc_prev - (current * dt / (Cell_Data.Cn * 3600));
end

