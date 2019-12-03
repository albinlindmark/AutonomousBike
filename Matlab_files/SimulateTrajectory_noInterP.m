function X = SimulateTrajectory_noInterP(phi0,vtraj,Blist,Klist,vlist)

% Predefined parameters
A = [1.0151, 0.0707;
     0.4318, 1.0151];
x = [phi0;0];
N = length(vtraj);

% Step through for each velocity
for i = 1:N
    
    % Find the index values for the closest velocities
    index = find(vlist==round(vtraj(i),1));
    
    % Interpolate to find the corresponding B and K
    B = [Blist(1,index); Blist(2,index)];
    K = [Klist(index,1), Klist(index,2)];
    
    % Calcuate the new input signal, u
    u(i) = -K*x(:,i); 
    
    % Clip the signal between -pi/2 -- pi/2
    u(i) = max(-90,u(i));
    u(i) = min(90,u(i));
    
    
    % Calculate the new state, x
    x(:,i+1) = A*x(:,i) + B*u(i);
    
end

% Set the output as X
X = x;

end