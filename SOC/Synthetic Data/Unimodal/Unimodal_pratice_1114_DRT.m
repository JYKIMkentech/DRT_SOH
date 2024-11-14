clc; clear; close all;

%% Description

% This code performs DRT estimation on selected datasets and types.
% It loads the data, allows you to select a dataset and type, and then
% performs the DRT estimation for each scenario within the selected data.
% The estimated gamma values are plotted and compared with the true gamma
% without interpolating the true gamma values.

%% Graphics settings

axisFontSize = 14;
titleFontSize = 12;
legendFontSize = 12;
labelFontSize = 12;

%% Load data

% Set the file path to the directory containing the .mat files
file_path = 'G:\공유 드라이브\Battery Software Lab\Projects\DRT\SD_new\';

% Get list of .mat files in the directory
mat_files = dir(fullfile(file_path, '*.mat')); % AS1_1per_new, AS1_2per_new, AS2_1per_new, AS2_2per_new, Unimodal_gamma, Bimodal_gamma

% Load the data files
for file = mat_files'
    load(fullfile(file_path, file.name));
end

%% Dataset selection

% List of datasets
datasets = {'AS1_1per_new', 'AS1_2per_new', 'AS2_1per_new', 'AS2_2per_new'};

% Display datasets and allow the user to select one
disp('Select a dataset:');
for i = 1:length(datasets)
    fprintf('%d. %s\n', i, datasets{i});
end
dataset_idx = input('Enter the dataset number: ');
selected_dataset_name = datasets{dataset_idx};
selected_dataset = eval(selected_dataset_name);

%% Type selection

% Extract the list of available types from the selected dataset
types = unique({selected_dataset.type});

% Display the types and allow the user to select one
disp('Select a type:');
for i = 1:length(types)
    fprintf('%d. %s\n', i, types{i});
end
type_idx = input('Enter the type number: ');
selected_type = types{type_idx};

% Extract data for the selected type
type_indices = find(strcmp({selected_dataset.type}, selected_type));
type_data = selected_dataset(type_indices);

% Extract scenario numbers
SN_list = [type_data.SN];

% Display selected dataset, type, and scenario numbers
fprintf('Selected dataset: %s\n', selected_dataset_name);
fprintf('Selected type: %s\n', selected_type);
fprintf('Scenario numbers: ');
disp(SN_list);

%% DRT Estimation

% Load the true gamma(theta) for comparison
% For AS1 datasets, the true gamma is Unimodal_gamma
% For AS2 datasets, the true gamma is Bimodal_gamma

if contains(selected_dataset_name, 'AS1')
    true_gamma_data = Gamma_unimodal;
elseif contains(selected_dataset_name, 'AS2')
    true_gamma_data = Gamma_bimodal;
else
    error('Unknown dataset name');
end

% Number of scenarios
num_scenarios = length(type_data);

% Initialize variables to store results
gamma_est_all = cell(num_scenarios, 1); % To store gamma estimates
V_est_all = cell(num_scenarios, 1); % To store V_est
V_sd_all = cell(num_scenarios, 1); % To store V_sd

% Colors for plotting
c_mat = lines(num_scenarios);

% For each scenario
for s = 1:num_scenarios
    fprintf('Processing Scenario %d/%d...\n', s, num_scenarios);
    
    % Get data for current scenario
    scenario_data = type_data(s);
    
    % Extract necessary data
    t = scenario_data.t; % time vector
    ik = scenario_data.I; % current
    V_sd = scenario_data.V; % measured voltage (Vsd)
    lambda_hat = 0.518; % scenario_data.lambda_hat; % optimal lambda
    n = scenario_data.n; % number of RC elements
    dt = scenario_data.dt; % sampling time
    dur = scenario_data.dur; % duration [tau_min, tau_max]
    
    % Now, set up the DRT estimation problem
    
    % Define theta_discrete and tau_discrete based on dur and n
    tau_min = 0.1;
    tau_max = dur;
    theta_min = log(tau_min);
    theta_max = log(tau_max);
    theta_discrete = linspace(theta_min, theta_max, n)';
    delta_theta = theta_discrete(2) - theta_discrete(1);
    tau_discrete = exp(theta_discrete);
    
    % Note: We will not interpolate gamma_true; we will use it as is
    
    % Set up the W matrix
    % W is a matrix of size length(t) x n
    W = zeros(length(t), n);
    for k_idx = 1:length(t)
        if k_idx == 1
            for i = 1:n
                W(k_idx, i) = ik(k_idx) * (1 - exp(-dt / tau_discrete(i))) * delta_theta;
            end
        else
            for i = 1:n
                W(k_idx, i) = W(k_idx-1, i) * exp(-dt / tau_discrete(i)) + ...
                              ik(k_idx) * (1 - exp(-dt / tau_discrete(i))) * delta_theta;
            end
        end
    end
    
    % Set up other parameters
    OCV = 0; % Open Circuit Voltage
    R0 = 0.1; % Initial resistance (adjust if necessary)
    
    % Adjust y (measured voltage)
    y_adjusted = V_sd - OCV - R0 * ik;
    
    % Regularization matrix L (first-order difference)
    L = zeros(n-1, n);
    for i = 1:n-1
        L(i, i) = -1;
        L(i, i+1) = 1;
    end
    
    % Set up the quadratic programming problem
    H = 2 * (W' * W + lambda_hat * (L' * L));
    f = -2 * W' * y_adjusted;
    
    % Inequality constraints: gamma >= 0
    A_ineq = -eye(n);
    b_ineq = zeros(n, 1);
    
    % Solve the quadratic programming problem
    options = optimoptions('quadprog', 'Display', 'off');
    gamma_est = quadprog(H, f, A_ineq, b_ineq, [], [], [], [], [], options);
    
    % Save the results
    gamma_est_all{s} = gamma_est;
    
    % Compute the estimated voltage
    V_est = OCV + R0 * ik + W * gamma_est;
    V_est_all{s} = V_est;
    V_sd_all{s} = V_sd;
end

%% Plotting Results

% Plot the estimated gamma for all scenarios
figure('Name', ['Estimated Gamma for ', selected_dataset_name, ' Type ', selected_type], 'NumberTitle', 'off');
hold on;
for s = 1:num_scenarios
    plot(theta_discrete, gamma_est_all{s}, '--', 'LineWidth', 1.5, ...
        'Color', c_mat(s, :), 'DisplayName', ['Scenario ', num2str(SN_list(s))]);
end
% Plot the true gamma without interpolation
plot(true_gamma_data.theta, true_gamma_data.gamma, 'k-', 'LineWidth', 2, 'DisplayName', 'True \gamma');
hold off;
xlabel('\theta = ln(\tau [s])', 'FontSize', labelFontSize);
ylabel('\gamma', 'FontSize', labelFontSize);
title(['Estimated \gamma for ', selected_dataset_name, ' Type ', selected_type], 'FontSize', titleFontSize);
set(gca, 'FontSize', axisFontSize);
legend('Location', 'Best', 'FontSize', legendFontSize);

% Optionally, select specific scenarios to plot
selected_scenarios = input('Enter scenario numbers to plot (e.g., [1,2,3]): ');

% Plot the estimated gamma for selected scenarios
figure('Name', ['Estimated Gamma for Selected Scenarios in ', selected_dataset_name, ' Type ', selected_type], 'NumberTitle', 'off');
hold on;
for idx = 1:length(selected_scenarios)
    s = find(SN_list == selected_scenarios(idx));
    if ~isempty(s)
        plot(theta_discrete, gamma_est_all{s}, '--', 'LineWidth', 1.5, ...
            'Color', c_mat(s, :), 'DisplayName', ['Scenario ', num2str(SN_list(s))]);
    else
        warning('Scenario %d not found in the data', selected_scenarios(idx));
    end
end
% Plot the true gamma without interpolation
plot(true_gamma_data.theta, true_gamma_data.gamma, 'k-', 'LineWidth', 2, 'DisplayName', 'True \gamma');
hold off;
xlabel('\theta = ln(\tau [s])', 'FontSize', labelFontSize);
ylabel('\gamma', 'FontSize', labelFontSize);
title(['Estimated \gamma for Selected Scenarios in ', selected_dataset_name, ' Type ', selected_type], 'FontSize', titleFontSize);
set(gca, 'FontSize', axisFontSize);
legend('Location', 'Best', 'FontSize', legendFontSize);
