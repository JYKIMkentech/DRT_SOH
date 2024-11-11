function [noisy_I, states] = add_markov_noise(I_original, n, sigma, noise_scale, initial_state)
    % add_markov_noise - 전류 데이터에 마르코프 체인 기반 노이즈를 추가하는 함수
    %
    % 입력:
    %   I_original    - 원본 전류 데이터 (벡터)
    %   n             - 마르코프 체인의 상태 수 (정수)
    %   sigma         - 노이즈의 표준 편차 (실수)
    %   noise_scale   - 노이즈의 스케일링 계수 (실수)
    %   initial_state - 초기 상태 (1부터 n 사이의 정수)
    %
    % 출력:
    %   noisy_I       - 노이즈가 추가된 전류 데이터 (벡터)
    %   states        - 각 시간에서의 마르코프 상태 (벡터)

    % 노이즈 벡터 생성
    noise_vector = linspace(min(I_original) * noise_scale, max(I_original) * noise_scale, n);

    % 전이 확률 행렬 생성
    P = zeros(n);

    % 각 상태에 대한 전이 확률 계산
    for i = 1:n
        probabilities = normpdf(noise_vector, noise_vector(i), sigma);
        P(i, :) = probabilities / sum(probabilities);
    end

    % 노이즈 추가 및 상태 기록을 위한 초기화
    noisy_I = zeros(size(I_original));
    states = zeros(size(I_original));
    current_state = initial_state;

    for idx = 1:length(I_original)
        % 현재 상태의 노이즈를 전류에 추가
        noisy_I(idx) = I_original(idx) + noise_vector(current_state);

        % 상태 기록
        states(idx) = current_state;

        % 다음 상태로 전이
        current_state = randsample(1:n, 1, true, P(current_state, :));
    end
end
