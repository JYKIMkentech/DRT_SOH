clc; clear; close all;

%% 0. Font Size and Color Matrix Settings
% Font Size Settings
axisFontSize = 14;      % Axis number size
titleFontSize = 16;     % Title font size
legendFontSize = 12;    % Legend font size
labelFontSize = 14;     % xlabel and ylabel font size

% Color Matrix Settings
c_mat = lines(9);  % Define 9 unique colors

%% 1. Load Trip Data
% Load 'Trips' structure
load('G:\공유 드라이브\BSL_Onori\Cycling_tests\Trips_Aging_1_W10.mat');  % 'Trips' 구조체를 로드합니다.

col_cell_label = {'W3','W4','W5','W7','W8','W9','W10','G1','V4','V5'};

%% 2. Load SOC-OCV Data
% Load 'soc_ocv_cap' data
load('RPT_All_soc_ocv_cap.mat', 'soc_ocv_cap');

% Extract SOC and OCV values
soc_values = soc_ocv_cap{14,7}(:, 1);  % SOC values (0 ~ 1) 
ocv_values = soc_ocv_cap{14,7}(:, 2);  % OCV values (V)

% Battery capacity extraction (in Ah)
Q_batt = 4.47;  % Battery capacity
SOC0 = 0.73;    % Initial SOC

% Regularization parameter lambda range
lambda_values = logspace(-5, 2, 20);  % Adjust as needed
num_lambdas = length(lambda_values);

%% 3. Parameters for DRT Estimation
n = 401;  % Number of discrete elements
tau_min = 0.1;     % Minimum time constant (seconds)
tau_max = 2600;    % Maximum time constant (seconds)

% Calculate Theta and tau values
theta_min = log(tau_min);
theta_max = log(tau_max);
theta_discrete = linspace(theta_min, theta_max, n);
tau_discrete = exp(theta_discrete);

% Calculate Delta theta
delta_theta = theta_discrete(2) - theta_discrete(1);



% Create first-order difference matrix L_gamma for Gamma
L_gamma = zeros(n-1, n);
for i = 1:n-1
    L_gamma(i, i) = -1;
    L_gamma(i, i+1) = 1;
end

% Create L_aug to avoid regularizing R0
L_aug = [L_gamma, zeros(n-1, 1)];

%% 4. Cross-Validation Setup
% Trip indices to use (trips 4 to 8)
trip_indices = 4:8;
num_trips = length(trip_indices);

% Generate all combinations of 2 trips for validation
validation_combinations = nchoosek(trip_indices, 2);
num_folds = size(validation_combinations, 1);

% Initialize variable to store CVE for each lambda
CVE = zeros(num_lambdas, 1);

%% 5. Cross-Validation Loop
for lambda_idx = 1:num_lambdas
    lambda = lambda_values(lambda_idx);
    cv_error = 0;  % Initialize cross-validation error for this lambda

    for fold_idx = 1:num_folds
        % Get validation trips for this fold
        val_trips = validation_combinations(fold_idx, :);
        % Training trips are the remaining ones
        train_trips = setdiff(trip_indices, val_trips);

        %% 5.1 Assemble Training Data
        W_train = [];
        y_train = [];

        for s = train_trips
            % Extract data for trip s
            t = Trips(s).t;
            ik = Trips(s).I;
            V_sd = Trips(s).V;

            % Compute SOC over time for trip s
            soc_over_time = compute_SOC_over_time(ik, t, Q_batt, SOC0);
            ocv_over_time = interp1(soc_values, ocv_values, soc_over_time);

            % Compute W_aug and y for trip s
            [W_aug_s, y_s] = compute_W_aug(ik, t, ocv_over_time, V_sd, tau_discrete, delta_theta);

            % Append to training data
            W_train = [W_train; W_aug_s];
            y_train = [y_train; y_s];
        end

        %% 5.2 Solve for gamma_total using quadprog
        % Set up the quadratic programming problem
        H = (W_train' * W_train) + lambda * (L_aug' * L_aug);
        f = -W_train' * y_train;

        % Constraints to enforce gamma >= 0
        Aeq = [];
        beq = [];
        lb = [zeros(n,1); -Inf];  % gamma >= 0, R0 unbounded below
        ub = [];

        % Solve the quadratic programming problem
        options = optimoptions('quadprog', 'Display', 'off');
        theta_est = quadprog(H, f, [], [], Aeq, beq, lb, ub, [], options);
        gamma_est = theta_est(1:end-1);
        R0_est = theta_est(end);

        %% 5.3 Validate on Validation Trips
        for s = val_trips
            % Extract data for trip s
            t = Trips(s).t;
            ik = Trips(s).I;
            V_sd = Trips(s).V;

            % Compute SOC over time for trip s
            soc_over_time = compute_SOC_over_time(ik, t, Q_batt, SOC0);
            ocv_over_time = interp1(soc_values, ocv_values, soc_over_time);

            % Compute V_est for trip s using gamma_est and R0_est
            V_est = compute_V_est(ik, t, ocv_over_time, gamma_est, R0_est, tau_discrete, delta_theta);

            % Compute error for this trip
            error_trip = sum((V_sd - V_est).^2);
            cv_error = cv_error + error_trip;
        end
    end

    % Store average CVE for this lambda
    CVE(lambda_idx) = cv_error / num_folds;
    fprintf('Lambda: %.2e, CVE: %.4f\n', lambda, CVE(lambda_idx));
end

%% 6. Select Optimal Lambda Before Plotting
[~, optimal_idx] = min(CVE);
optimal_lambda = lambda_values(optimal_idx);
fprintf('Optimal Lambda: %.2e\n', optimal_lambda);

%% 7. Plot CVE vs Lambda
figure;
semilogx(lambda_values, CVE, 'b-', 'LineWidth', 1.5); hold on;
semilogx(optimal_lambda, CVE(optimal_idx), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('\lambda (정규화 파라미터)', 'FontSize', labelFontSize);
ylabel('교차 검증 오류 (CVE)', 'FontSize', labelFontSize);
title('로그 스케일 \lambda 에 따른 CVE 그래프', 'FontSize', titleFontSize);
grid on;
legend({'CVE', ['최적 \lambda = ', num2str(optimal_lambda, '%.2e')]}, 'Location', 'best');
ylim([1160 1161])
hold off;


%% Helper Functions
% Function to compute W_aug and y
function [W_aug, y] = compute_W_aug(ik, t, ocv_over_time, V_sd, tau_discrete, delta_theta)
    n = length(tau_discrete);
    dt = [t(1); diff(t)];  % Time step differences

    % Initialize W matrix
    W = zeros(length(t), n);
    for k_idx = 1:length(t)
        for i = 1:n
            if k_idx == 1
                W(k_idx, i) = ik(k_idx) * (1 - exp(-dt(k_idx) / tau_discrete(i))) * delta_theta;
            else
                W(k_idx, i) = W(k_idx-1, i) * exp(-dt(k_idx) / tau_discrete(i)) + ...
                              ik(k_idx) * (1 - exp(-dt(k_idx) / tau_discrete(i))) * delta_theta;
            end
        end
    end

    % Augment W with current ik for R0 term
    W_aug = [W, ik(:)];

    % Compute y
    y = V_sd - ocv_over_time;
    y = y(:);  % Ensure y is a column vector
end

% Function to compute V_est
function V_est = compute_V_est(ik, t, ocv_over_time, gamma_est, R0_est, tau_discrete, delta_theta)
    n = length(tau_discrete);
    dt = [t(1); diff(t)];  % Time step differences
    V_RC = zeros(n, length(t));  % Each element's voltage
    V_est = zeros(length(t), 1);

    for k_idx = 1:length(t)
        if k_idx == 1
            for i = 1:n
                V_RC(i, k_idx) = gamma_est(i) * delta_theta * ik(k_idx) * (1 - exp(-dt(k_idx) / tau_discrete(i)));
            end
        else
            for i = 1:n
                V_RC(i, k_idx) = V_RC(i, k_idx-1) * exp(-dt(k_idx) / tau_discrete(i)) + ...
                                 gamma_est(i) * delta_theta * ik(k_idx) * (1 - exp(-dt(k_idx) / tau_discrete(i)));
            end
        end
        % Compute V_est at time k_idx
        V_est(k_idx) = ocv_over_time(k_idx) + R0_est * ik(k_idx) + sum(V_RC(:, k_idx));
    end
end

% Function to compute SOC over time
function soc_over_time = compute_SOC_over_time(ik, t, Q_batt, SOC0)
    dt = [t(1); diff(t)];  % Time step differences
    soc_over_time = zeros(length(t), 1);
    soc_over_time(1) = SOC0;

    for k_idx = 2:length(t)
        soc_over_time(k_idx) = soc_over_time(k_idx-1) + (ik(k_idx) * dt(k_idx)) / (3600 * Q_batt);
    end
end
