function X = SimulateTrajectory(phi0,vtraj,Blist,Klist,vlist)

% Parameters for different scenarios
dt = 0.04; % s
top_rate = 200; % degrees/s
deadzone_rate = 20; % degrees/s
max_roll = 45; % degrees
u = 0; % Set handle to zero degrees as initial


% Predefined parameters
A = [1.0151, 0.0707;
     0.4318, 1.0151];
x = [phi0;0];
N = length(vtraj);

% Step through for each velocity
for i = 1:N
    
    % Find the index values for the closest velocities
    index1 = find(vlist==round(vtraj(i),1));
    vx = vlist(index1)-vtraj(i);
    
    if vx < 0
        index2 = index1 + 1;
        relation = 1;
    elseif vx > 0
        index2 = index1 - 1;
        relation = 2;
    else
        index2 = index1;
        relation = 0;
    end
    
    % Interpolate to find the corresponding B and K
    B = [interpol(Blist(1,index1),Blist(1,index2),vx,relation);
         interpol(Blist(2,index1),Blist(2,index2),vx,relation)];
     
    K = [interpol(Klist(index1,1),Klist(index2,1),vx,relation), ...
         interpol(Klist(index1,2),Klist(index2,2),vx,relation)];
    
    % Calcuate the new input signal, u
    u(i+1) = -K*x(:,i); 
    
    % Limit the rate to top_rate*dt
    if u(i+1) - u(i) > top_rate*dt
        u(i+1) = u(i) + top_rate*dt;
    end

    if u(i+1) - u(i) < - top_rate*dt
        u(i+1) = u(i) - top_rate*dt;
    end

    
    
    % If the change between u(i+1)-u(i) less than zero
    if abs(u(i+1) - u(i)) <= deadzone_rate*dt
        u(i+1) = u(i); 
    end
   
    
    % Clip the signal between -pi/2 -- pi/2
    u(i+1) = max(-max_roll,u(i+1));
    u(i+1) = min(max_roll,u(i+1));
    
    % Include deadzones for the input signal
    %if abs(u(i)) < 3
    %    u(i) = 0;
    %end
    % Calculate the new state, x
    
    x(:,i+1) = A*x(:,i) + B*u(i+1);
    
end

% Set the output as X
X = x;
figure()
x_time = 0:0.04:length(vtraj)*0.04;
plot(x_time(:),u(:))
title('Control signal')

figure()
plot(x_time(2:end), diff(u(:)*1/0.04))
title('Delta_u')

end