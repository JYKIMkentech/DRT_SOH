%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% Cycling test analysis %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% Gabriele Pozzato (gpozzato@stanford.edu) %%%%%% Date: 2022/01 %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% In case of help, feel free to contact the author %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clear all
% close all
% clc

warning off

%% User input
% Cell selection
% -----------------
% sel_cell -> label
% -----------------
% 1        -> W3
% 2        -> W4
% 3        -> W5
% 4        -> W7
% 5        -> W8
% 6        -> W9
% 7        -> W10
% 8        -> G1
% 9        -> V4
% 10       -> V5
% -----------------
sel_cell = 7; 

%% General
cell_name = {'W3' 'W4' 'W5' 'W7' 'W8' 'W9' 'W10' 'G1' 'V4' 'V5'};

%% Cycling test lot
try
    load([cell_name{sel_cell} '.mat'])

    % Rename signals
    time = t_full_vec_M1_NMC25degC;
    curr = I_full_vec_M1_NMC25degC;                                            
    volt = V_full_vec_M1_NMC25degC;
    step_arbin = Step_Index_full_vec_M1_NMC25degC;
    
    % Plot
    figure; 
    subplot(2,1,1); hold on; box on; title(['\textbf{Cell ' cell_name{sel_cell} '}'])
    plot(time/3600,curr,'linewidth',2,'color',[0.8500 0.3250 0.0980]); 
    ylabel('Current [A]'); xlim([0 inf]); ylim([-inf inf]); 
    subplot(2,1,2); hold on; box on; 
    plot(time/3600,volt,'linewidth',2,'color',[0 0.4470 0.7410]); 
    ylabel('Voltage [V]'); xlabel('Time [h]'); xlim([0 inf]); ylim([-inf inf]); 
    set(findall(gcf,'-property','FontSize'),'FontSize',16);
    set(findall(gcf,'-property','interpreter'),'interpreter','latex')
    set(findall(gcf,'-property','ticklabelinterpreter'),'ticklabelinterpreter','latex')
catch
    disp('No data!')
end