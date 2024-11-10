function [gamma_est, R0_est, V_est] = compute_drt(t, ik, V_sd, ocv_over_time, delta_t, tau_discrete, delta_theta, lambda, L_aug, n)
    % compute_drt: DRT 해를 계산하는 함수
    % 입력:
    %   - t: 시간 벡터
    %   - ik: 전류 벡터
    %   - V_sd: 측정된 전압 벡터
    %   - ocv_over_time: OCV 벡터
    %   - delta_t: 시간 간격 벡터
    %   - tau_discrete: 시간 상수 벡터
    %   - delta_theta: theta의 간격
    %   - lambda: 정규화 파라미터
    %   - L_aug: 정규화 행렬
    %   - n: 이산 요소의 개수
    % 출력:
    %   - gamma_est: 추정된 gamma 벡터
    %   - R0_est: 추정된 R0 값
    %   - V_est: 추정된 전압 벡터

    %% W 행렬 생성
    W = zeros(length(t), n);
    for k_idx = 1:length(t)
        for i = 1:n
            if k_idx == 1
                W(k_idx, i) = ik(k_idx) * (1 - exp(-delta_t(k_idx) / tau_discrete(i))) * delta_theta;
            else
                W(k_idx, i) = W(k_idx-1, i) * exp(-delta_t(k_idx) / tau_discrete(i)) + ...
                              ik(k_idx) * (1 - exp(-delta_t(k_idx) / tau_discrete(i))) * delta_theta;
            end
        end
    end

    % R0 추정을 위한 W 행렬 확장
    W_aug = [W, ik(:)];  % ik(:)는 ik를 열 벡터로 변환

    %% y 벡터 생성
    % y = V_sd - OCV
    y = V_sd - ocv_over_time;
    y = y(:);  % y를 열 벡터로 변환

    %% quadprog를 사용한 제약 조건 하의 추정
    % 비용 함수: 0.5 * Theta' * H * Theta + f' * Theta
    H = (W_aug' * W_aug + lambda * (L_aug' * L_aug));
    f = -W_aug' * y;

    % 제약 조건: Theta >= 0 (gamma와 R0는 0 이상)
    A = -eye(n+1);
    b = zeros(n+1, 1);

    % quadprog 옵션 설정
    options = optimoptions('quadprog', 'Display', 'off', 'Algorithm', 'interior-point-convex');

    % quadprog 실행
    [Theta_est, ~, exitflag] = quadprog(H, f, A, b, [], [], [], [], [], options);

    if exitflag ~= 1
        warning('Optimization did not converge.');
    end

    % gamma와 R0 추정값 추출
    gamma_est = Theta_est(1:n);
    R0_est = Theta_est(n+1);

    %% V_est 계산
    V_RC = zeros(n, length(t));  % 각 요소의 전압
    V_est = zeros(length(t), 1);
    for k_idx = 1:length(t)
        if k_idx == 1
            for i = 1:n
                V_RC(i, k_idx) = gamma_est(i) * delta_theta * ik(k_idx) * (1 - exp(-delta_t(k_idx) / tau_discrete(i)));
            end
        else
            for i = 1:n
                V_RC(i, k_idx) = V_RC(i, k_idx-1) * exp(-delta_t(k_idx) / tau_discrete(i)) + ...
                                 gamma_est(i) * delta_theta * ik(k_idx) * (1 - exp(-delta_t(k_idx) / tau_discrete(i)));
            end
        end
        % 시간 k_idx에서의 V_est 계산
        V_est(k_idx) = ocv_over_time(k_idx) + R0_est * ik(k_idx) + sum(V_RC(:, k_idx));
    end
end
