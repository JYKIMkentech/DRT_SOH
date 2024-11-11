function [SOC_true, Vt_true] = battery_model(SOC_prev, V1_prev, ik, Config)
    SOC_true = SOC_prev - (Config.dt / Config.cap) * Config.coulomb_efficient * ik;
    Vt_true = ocv_soc(SOC_true) - Config.R1 * V1_prev - Config.R0 * ik;
end

