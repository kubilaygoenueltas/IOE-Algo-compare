function [q_est, b_est, a_est, P_est] = CMKMC_ESKF(yG, yA, dt, sigma, num_steps)
    % Compact Multi-Kernel Maximum Correntropy-Based Error State Kalman Filter (CMKMC-ESKF)
    %
    % Inputs:
    %   yG - Gyroscope measurements (3xT)
    %   yA - Accelerometer measurements (3xT)
    %   dt - Sampling time
    %   sigma - Kernel bandwidth for non-Gaussian acceleration disturbances
    %   num_steps - Number of iterations (T)
    %
    % Outputs:
    %   q_est - Estimated quaternion (4xT)
    %   b_est - Estimated gyroscope bias (3xT)
    %   a_est - Estimated external acceleration (3xT)
    %   P_est - Estimated covariance (9x9xT)

    % Initialization
    q_est = zeros(4, num_steps);
    b_est = zeros(3, num_steps);
    a_est = zeros(3, num_steps);
    P_est = zeros(9, 9, num_steps);
    
    % Initial estimates
    q_est(:,1) = [1; 0; 0; 0]; % Identity quaternion
    P_est(:,:,1) = eye(9); % Initial covariance
    
    % System matrices
    Q = diag([1e-6 * ones(1,3), 1e-4 * ones(1,3), 1e-2 * ones(1,3)]); % Process noise
    R = diag([1e-2 * ones(1,3)]); % Measurement noise

    % Indicator matrices
    Lambda_p = blkdiag(eye(3), eye(3), zeros(3)); % Non-Gaussian acceleration
    Lambda_r = eye(3); % Gaussian measurement noise
    
    for k = 2:num_steps
        % Prediction Step
        omega = yG(:, k) - b_est(:, k-1);
        q_pred = quaternion_update(q_est(:, k-1), omega, dt);
        b_pred = b_est(:, k-1);
        a_pred = 0.95 * a_est(:, k-1); % Acceleration Markov assumption
        
        x_pred = [q_pred(2:4); b_pred; a_pred]; % Quaternion as vector part only
        P_pred = P_est(:,:,k-1) + Q;
        
        % Decomposition
        [Bp,~] = chol(P_pred, 'lower');
        [Br,~] = chol(R, 'lower');
        
        % Error state
        e_p = Bp \ (x_pred - [q_est(2:4, k-1); b_est(:, k-1); a_est(:, k-1)]);
        e_r = Br \ (yA(:,k) - rotate_vector(q_pred, [0; 0; -9.81]));
        
        % Compute kernel matrices
        M_p = Lambda_p + (eye(9) - Lambda_p) .* diag(kernel_gaussian(e_p, sigma));
        M_r = Lambda_r + (eye(3) - Lambda_r) .* diag(kernel_gaussian(e_r, sigma));

        % Kalman Gain
        K_tilde = (Bp * M_p^(-1) * Bp') * eye(3) / (eye(3) * (Bp * M_p^(-1) * Bp') * eye(3)' + Br * M_r^(-1) * Br');

        % Update Error State
        x_update = x_pred + K_tilde * (yA(:, k) - rotate_vector(q_pred, [0; 0; -9.81]));

        % Update Quaternion, Bias, Acceleration
        q_update = quaternion_update(q_pred, -x_update(1:3), 1);
        b_est(:, k) = b_pred - x_update(4:6);
        a_est(:, k) = a_pred - x_update(7:9);

        % Save quaternion as full form
        q_est(:, k) = q_update;

        % Update Covariance
        P_est(:,:,k) = (eye(9) - K_tilde * eye(3)) * P_pred * (eye(9) - K_tilde * eye(3))' + K_tilde * R * K_tilde';
    end
end

function q_next = quaternion_update(q, omega, dt)
    % Quaternion Update using Angular Velocity
    omega_norm = norm(omega);
    if omega_norm > 1e-6
        dq = [cos(0.5 * omega_norm * dt);
              sin(0.5 * omega_norm * dt) * (omega / omega_norm)];
    else
        dq = [1; 0; 0; 0];
    end
    q_next = quaternion_multiply(q, dq);
    q_next = q_next / norm(q_next); % Normalize
end

function q_mult = quaternion_multiply(q1, q2)
    % Quaternion Multiplication
    w1 = q1(1); v1 = q1(2:4);
    w2 = q2(1); v2 = q2(2:4);
    q_mult = [w1*w2 - dot(v1, v2);
              w1*v2 + w2*v1 + cross(v1, v2)];
end

function v_rot = rotate_vector(q, v)
    % Rotate Vector Using Quaternion
    vq = [0; v];
    v_rot = quaternion_multiply(quaternion_multiply(q, vq), quaternion_conjugate(q));
    v_rot = v_rot(2:4);
end

function q_conj = quaternion_conjugate(q)
    % Quaternion Conjugate
    q_conj = [q(1); -q(2:4)];
end

function K = kernel_gaussian(e, sigma)
    % Gaussian Kernel Function
    K = exp(-0.5 * (e ./ sigma).^2);
end
