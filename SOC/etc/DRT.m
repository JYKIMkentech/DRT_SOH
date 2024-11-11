%% Parameters
n = 5;  % Number of RC elements used for voltage calculation (this remains as 5)
m = 21;  % Number of discrete tau and R values for distribution (can be different from n)
t = 0:0.01:100;  % Time vector (discrete time)

% Synthetic current data (sum of sine waves)
A = 1; 
T1 = 1;
T2 = 5;
T3 = 20;
I1 = A * sin(2 * pi * t / T1);
I2 = A * sin(2 * pi * t / T2);
I3 = A * sin(2 * pi * t / T3);
ik = I1 + I2 + I3;  % Total current

% Parameters for the normal distribution of tau
mu = 10;
sigma = 5;
tau_discrete = linspace(0.01, 20, m);  % Discrete tau values for RC elements (m values)

% Use normpdf to directly calculate the R_discrete corresponding to tau_discrete
R_discrete = normpdf(tau_discrete, mu, sigma);

% Normalize R_discrete so that the maximum value is 1
R_discrete = R_discrete / max(R_discrete);  % Now the maximum value of R_discrete is 1

% Initialize voltage
V_est = zeros(1, length(t));  % Estimated voltage
R0 = 0.1;  % Internal resistance
OCV = 0;   % Open circuit voltage
V_RC = zeros(n, length(t));  % RC voltages for each RC element (n values)

%% Voltage calculation for each time step (using first n elements from tau_discrete and R_discrete)
for k = 1:length(t)
    if k < length(t)
        dt = t(k+1) - t(k);  % Calculate dynamic time step for each time
    else
        dt = t(k) - t(k-1);  % For the last step, use the previous dt
    end
    
    % Compute RC voltages for each of the n RC elements (use only the first n elements of tau_discrete)
    for i = 1:n
        if k == 1
            % First time step, initial V_RC calculation
            V_RC(i, k) = ik(k) * R_discrete(i) * (1 - exp(-dt / tau_discrete(i)));
        else
            % Subsequent time steps, accumulate V_RC
            V_RC(i, k) = exp(-dt / tau_discrete(i)) * V_RC(i, k-1) + ...
                         R_discrete(i) * (1 - exp(-dt / tau_discrete(i))) * ik(k);
        end
    end

    % Final voltage summation: OCV + R0 * I + sum(V_RC for n RC elements)
    V_est(k) = OCV + R0 * ik(k) + sum(V_RC(:, k));
end

%% Add noise to the voltage
noise_level = 0.01;
V_noisy = V_est + noise_level * randn(size(V_est));

%% Fitting: Ensure m elements for R_fitted
R_initial = ones(1, m);  % Initial guess for R (m elements)
options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'sqp');

R_fitted = fmincon(@(R_fit) cost_function(R_fit, tau_discrete, ik, V_noisy, t, R0, OCV, m, n), ...
                   R_initial, [], [], [], [], zeros(1, m), [], [], options);

%% Plot the results: Current, Voltage, and DRT comparison
figure;

% Subplot for current
subplot(2, 1, 1);
plot(t, ik, 'k-', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Current (A)');
title('Composite Current (Sum of Sine Waves)');
grid on;

% Subplot for voltage
subplot(2, 1, 2);
plot(t, V_noisy, 'r--', 'LineWidth', 1.5); hold on;
V_fitted = V_noisy - residuals(R_fitted, tau_discrete, ik, V_noisy, t, R0, OCV, m, n);
plot(t, V_fitted, 'b-', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Voltage (V)');
legend('Noisy Voltage', 'Fitted Voltage');
title('Estimated and Fitted Voltages');
grid on;

% Subplot for DRT comparison
figure(2);
plot(tau_discrete, R_discrete, 'b-', 'LineWidth', 1.5);  % True DRT
hold on;
stem(tau_discrete, R_discrete, 'r', 'LineWidth', 1.5);  % Vertical lines for True DRT
plot(tau_discrete, R_discrete, 'ro', 'LineWidth', 1.5);  % Points on the curve

plot(tau_discrete, R_fitted, 'g-', 'LineWidth', 1.5);  % Fitted DRT
stem(tau_discrete, R_fitted, 'g', 'LineWidth', 1.5);  % Vertical lines for Fitted DRT
plot(tau_discrete, R_fitted, 'go', 'LineWidth', 1.5);  % Points on the curve
xlabel('\tau (Time Constant)');
ylabel('R (Resistance)');
legend('True DRT', 'Fitted DRT');
title('True vs Fitted Distribution of Relaxation Times (DRT)');
grid on;

%% Function Definitions (Update residuals function for fitting m elements)
function res = residuals(R_fit, tau_discrete, ik, V_noisy, t, R0, OCV, m, n)
    V_est_fit = zeros(1, length(t));
    V_RC_fit = zeros(m, length(t));  % Use m values for fitting

    for k = 1:length(t)
        if k < length(t)
            dt = t(k+1) - t(k);
        else
            dt = t(k) - t(k-1);
        end

        % Calculate RC voltages with fitted R for each of the m values
        for i = 1:m
            if i <= n  % Only calculate voltage for the first n RC elements
                if k == 1
                    V_RC_fit(i, k) = ik(k) * R_fit(i) * (1 - exp(-dt / tau_discrete(i)));
                else
                    V_RC_fit(i, k) = exp(-dt / tau_discrete(i)) * V_RC_fit(i, k-1) + ...
                                     R_fit(i) * (1 - exp(-dt / tau_discrete(i))) * ik(k);
                end
            else
                V_RC_fit(i, k) = 0;  % Ignore RC voltages for elements beyond n
            end
        end
        V_est_fit(k) = OCV + R0 * ik(k) + sum(V_RC_fit(:, k));
    end
    % Calculate residuals (difference between noisy and estimated voltage)
    res = V_noisy - V_est_fit;
end

% Cost function (sum of squared residuals)
function cost = cost_function(R_fit, tau_discrete, ik, V_noisy, t, R0, OCV, m, n)
    res = residuals(R_fit, tau_discrete, ik, V_noisy, t, R0, OCV, m, n);
    cost = sum(res.^2);  % Sum of squared residuals
end

