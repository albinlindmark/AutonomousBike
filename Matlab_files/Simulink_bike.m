clear
clc

r_wheel = 0.311;
a = 0.4964;
b = 1.095; 
h = 0.2085 + r_wheel;
g = 9.81;

vlist = linspace(0.5,10,96);

for i = 1:length(vlist)
Gs_b = [0 a*vlist(i) vlist(i)^2];

Gs_a = [b*h 0 -g*b];

[A,B,C,D] = tf2ss(Gs_b,Gs_a);
Blist(:,i) = B;

Q = 10000*eye(2);

R = 1;

[Kpre,S,P] = lqr(A,B,Q,R);
K(i,:) = Kpre;
end