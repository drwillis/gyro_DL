clear all
close all
clc

h=0.0001;  %simulation time step
t=0:h:2.5;   %simulation time vector

%constants
m=0.001;  %mass of vibrating mass
Jzz=.1;  %moment of inertia of vibrating mass about rotational axis
ky=10;  %spring constant in vibrating direction
by=.1;  %viscous friction coefficient in vibrating direction
kx=10;  %spring constant in sensing direction
bx=1;  %vicsous friction coefficient in sensing direction
y0=.1;  %amplitude of excitation signal
wy=500;  %frequency of excitation signal
cy=0.00001;  %Coulombic Friction in excitation direction
cx=0.00001;  %Coulombic Friction in sensing direction
params=[m;Jzz;ky;by;kx;bx;y0;wy;cy;cx];  %Group constants into a vector to send to ODE function


b=zeros(6,length(t));  %preallocate memory for samples of state vector
b(:,1)=[0;0;0;0;0;0];
tau=0.5*square(2*pi*t*(1/2))-.25;  %torque applied to gyroscope.  Modify this signal to obtain different data sets

%4th Order Runge Kutta Integration
for i=1:length(t)-1
    k1=vibratory_mems_gyro(b(:,i),tau(i),t(i),params);
    k2=vibratory_mems_gyro(b(:,i)+k1*h/2,tau(i),t(i),params);
    k3=vibratory_mems_gyro(b(:,i)+k2*h/2,tau(i),t(i),params);
    k4=vibratory_mems_gyro(b(:,i)+k3*h,tau(i),t(i),params);
    b(:,i+1)=b(:,i)+h*(k1/6+k2/3+k3/3+k4/6);
end

subplot(3,1,1)
plot(t,b(1,:))
ylabel('x (m)')
title('Configuration Coordinates')
subplot(3,1,2)
plot(t,b(2,:))
ylabel('y (m)')
subplot(3,1,3)
plot(t,b(3,:))
ylabel('\theta (rad)')
xlabel('t (s)')


[Ax,Axlower]=envelope(b(1,:));  %envelope detector
%compute the magnitude of the anglular velocity
angvel_mag=Ax*(sqrt((ky-m*wy^2)^2+(by*wy)^2)*sqrt((kx-m*wy^2)^2+(bx*wy)^2))/(2*m*wy*y0);
figure
plot(t,angvel_mag);
title('Magnitude of Angular Velocity')
xlabel('t (s)')
ylabel('|d\theta/dt|')

%Detect if phase of x(t) lead or lags y(t) using saturation and D-flip-flop
%.... lead==>pos. velocity, lag==>neg. velocity
xB=sign(b(1,:));
yB=sign(cos(wy*t));
angvel_sign=zeros(size(t));
I=find(diff(yB)>0); %find index of the rising edges
for i=2:length(t)
    if sum(i==I)>0
        angvel_sign(i)=xB(i);
    else
        angvel_sign(i)=angvel_sign(i-1);
    end
end
angvel=angvel_mag.*angvel_sign;

figure
subplot(2,1,1)
plot(t,tau)
ylabel('\tau(t) (Nm)')
title('Strapdown Torque')
subplot(2,1,2)
plot(t,angvel);
hold on
plot(t,b(6,:),'r')
xlabel('t (s)')
ylabel('d\theta/dt (rad/s)')
legend('Sensed','Actual')

%For system identification, the output of the rate gyroscope is used as the input and the
%known angular velocity (i.e. from rate table), is used as the ouput.  In
%this application, we want to identify the inverse dynamics of the model
%above.  This would seem to be true for similar sensor identification.

outputData=b(6,:);
inputData=angvel;