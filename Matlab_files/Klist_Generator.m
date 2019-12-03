clc, clearvars
close all

r_wheel = 0.311; % Radius of the wheel [m]
h = 0.2085 + r_wheel;           % height of center of mass [m]
b = 1.095;            % length between wheel centers [m]
c = 0.06;             % length between front wheel contact point and the 
                           % extention of the fork axis [m]
lambda = deg2rad(70); % angle of the fork axis [deg]
a = 0.4964;          % distance from rear wheel to frame's center of mass [m]
IMUheight = 0.45;   % IMU height [m]
m = 44.2; % Mass of the Bike [kg]
g = 9.81;  % gravity [m/s^2]
Ts = 0.04; % Sampling Time [s]
J = m*h^2; % Inertia [kg m^2]
D_inertia = m*a*h; % Inertia [kg m^2]


vlist = linspace(0.5,10,96);

for i = 1:length(vlist)
    
    v = vlist(i); % m/s

    % Choosing the state as x = [phi, b*h*dphi - a*v*delta], results in the following state
    % space model, equivalent to G = (a*v*s + v^2)/(b*h*s^2 - b*g), apart from
    % that not just phi is measured, but also the b*h*dphi-a*v*delta state element:
    Ac = [0 1/(b*h); b*g 0];
    Bc = [a*v/(b*h); v^2];
    % Measure both states. If Cc would have been [1 0] (i.e. if we only
    % measured phi instead of both states), then the plant would be equivalent
    %to phi(s) = (a*v*s + v^2)/(b*h*s^2 - b*g) * delta(s):
    Cc = eye(2);
    plant = ss(Ac, Bc, Cc, []);
    plant = c2d(plant, Ts); % Discretize model
    A = plant.A; B(:,i) = plant.B; C = plant.C;

    M = [1 0]; % phi = M*x
    Q_phi = 1e1; 
    Q = M'*Q_phi*M; % i.e. only care about deviations from the phi state
    R = 1;
    [K(i,:),~,~] = lqr(plant,Q,R); % Performs DARE and returns gain K
end

%% Perform the trajectory for v

% Test 1 
%vtraj1 = linspace(0.5,10,100);
vtraj1 = 0.5:0.01:5;
vtraj1 = [10*ones(1,100)];
X = SimulateTrajectory(10,vtraj1,B,K,vlist);
X_no = SimulateTrajectory_noInterP(10,vtraj1,B,K,vlist);

figure()
plot(X(1,:)) 
hold on 
plot(X_no(1,:),'--')
