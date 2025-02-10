function plot_quaternion_angles(time, q1_data, q2_data)
    % Number of samples
    num_samples = size(q1_data, 1);
    
    % Preallocate angle array
    angles = zeros(num_samples, 1);
    
    for i = 1:num_samples
        q1 = q1_data(i, :);  % Quaternion from sensor 1 at time step i
        q2 = q2_data(i, :);  % Quaternion from sensor 2 at time step i

        % Normalize quaternions
        q1 = q1 / norm(q1);
        q2 = q2 / norm(q2);

        % Compute relative quaternion
        q_relative = conj(q1) * q2;

        % Extract the scalar part of quaternion (w-component)
        w = parts(q_relative); % parts() returns [w, x, y, z], so we need the first element
        w=w(1); % Extract only the first element (real part of quaternion)
        
        % Clamp w to avoid numerical issues in acos
        w = max(-1, min(1, double(w))); 
        
        % Compute the angle in radians
        angles(i) = 2 * acos(abs(w));
    end
    
    % Convert to degrees
    angles = rad2deg(angles);
    
    % Plot the results
    figure;
    plot(time, angles, 'b', 'LineWidth', 2);
    xlabel('Time (s)');
    ylabel('Angle (degrees)');
    title('Knee Angle over Time');
    grid on;
end

