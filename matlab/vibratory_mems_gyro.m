function bdot=vibratory_mems_gyro(b,tau,t,params)

bdot=zeros(6,1);

%state variable assignemnts
x=b(1);
y=b(2);
theta=b(3);
dotx=b(4);
doty=b(5);
dottheta=b(6);

%Configation coordinate vector
gamma=[x;y;theta];
%System Speeds
dotgamma=[dotx;doty;dottheta];

%constants
m=params(1);  %mass of vibrating mass
Jzz=params(2);  %moment of inertia of vibrating mass about rotational axis
ky=params(3);  %spring constant in vibrating direction
by=params(4);  %viscous friction coefficient in vibrating direction
kx=params(5);  %spring constant in sensing direction
bx=params(6);  %vicsous friction coefficient in sensing direction
y0=params(7);  %amplitude of excitation signal
wy=params(8);  %frequency of excitation signal
cy=params(9);  %Coulombic Friction in excitation direction
cx=params(10);  %Coulombic Friction in sensing direction

%System Mass matrix
H=[m 0 -m*y;...
   0 m m*x;...
   -m*y m*x m*(x^2+y^2)+Jzz];

%Coriolis and Centripital force vector
d=[-2*m*doty*dottheta-m*x*dottheta^2;...
    2*m*dotx*dottheta-m*y*dottheta^2;...
    2*m*x*dotx*dottheta+2*m*y*doty*dottheta];

%Spring constant matrix
K=diag([kx;ky;0]);

%Viscous damping constant matrix
B=diag([bx;by;0]);

%Coulombic Friction constant matrix
C=diag([cx;cy;0]);

F=[0;...  %no external force applied in sensing directin (x)
   y0*cos(wy*t);...  %excitation signal applied in vibrating direction (y)
   tau];   %torque applied to the gyroscope (so that it rotates)

%Rate of change of state vector
bdot(1:3)=b(4:6);
bdot(4:6)=H\(F-d-K*gamma-B*dotgamma-C*sign(dotgamma));
aaa = 1;