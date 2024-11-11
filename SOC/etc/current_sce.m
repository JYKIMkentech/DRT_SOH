% Define the time vector
t = linspace(0, 100, 1000); % 0 to 100 seconds with 1000 points

% Amplitudes and periods for each scenario based on your latest images
A = [1, 1, 1;    % Scenario 1
     1.7, 0.6, 0.7;   % Scenario 2
     0.2, 0.5, 2.3;   % Scenario 3
     1.3, 1.1, 0.6; % Scenario 4
     1.7, 1.8, 0.5; % Scenario 5
     1.27, 1.33, 0.4; % Scenario 6
     1.2, 1.6, 0.2; % Scenario 7
     0.9, 0.7, 2.4; % Scenario 8
     1.1, 1.1, 0.8; % Scenario 9
     0.1, 0.1, 2.8]; % Scenario 10

T = [1, 5, 20;   % Scenario 1
     2, 4, 20;   % Scenario 2
     1, 20, 25;  % Scenario 3
     1.5, 5.3, 19.8; % Scenario 4
     2.5, 4.2, 20.5; % Scenario 5
     1.5, 20.9, 24.2; % Scenario 6
     1.3, 6, 19.3; % Scenario 7
     2.2, 4.8, 20.2; % Scenario 8
     2, 20.8, 26.1; % Scenario 9
     1.1, 4.3, 20.1]; % Scenario 10

% Create a figure for the 5x2 subplot
figure;

% Loop through each scenario
for scenario = 1:10
    % Extract amplitudes and periods for the current scenario
    A1 = A(scenario, 1);
    A2 = A(scenario, 2);
    A3 = A(scenario, 3);
    T1 = T(scenario, 1);
    T2 = T(scenario, 2);
    T3 = T(scenario, 3);
    
    % Generate the mixed signal for the current scenario
    I_scenarios = A1 * sin(2 * pi * t / T1) + ...
                  A2 * sin(2 * pi * t / T2) + ...
                  A3 * sin(2 * pi * t / T3);
    
    % Plot the current scenario in a 5x2 subplot
    subplot(5, 2, scenario);
    plot(t, I_scenarios);
    
    % Set title with Amplitude and Periods (in the order: A1, A2, A3, T1, T2, T3)
    title(sprintf('Scenario %d: A1=%.2f, A2=%.2f, A3=%.2f, T1=%.1f, T2=%.1f, T3=%.1f', ...
                  scenario, A1, A2, A3, T1, T2, T3));
    xlabel('Time (s)');
    ylabel('Current (A)');
end

% Adjust the layout
sgtitle('Synthetic data');


