function Config = initialize_config(udds_time, udds_current)
    Config.dt = mean(diff(udds_time)); % 평균 시간 간격
    Config.ik = udds_current(1); % 초기 전류
    Config.R0 = 0.001884314;
    Config.R1 = 0.045801322;
    Config.C1 = 4846.080679;
    Config.cap = 2.99 ; % nominal capacity [Ah]
    Config.coulomb_efficient = 1;
end
