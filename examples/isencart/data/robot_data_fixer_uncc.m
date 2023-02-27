close all
clear all
%clc 

set=10;

data=importdata(['data_',num2str(set),'.txt']);

Data=data.data;

t=Data(:,1)-Data(1,1);
Pr0=[Data(:,3:5)'];
Pr1=[Data(:,7:9)'];
Pr2=[Data(:,11:13)'];
Pr3=[Data(:,15:17)'];
%set drop outs to previous value
for i=2:length(t)-1
    if Pr0(1,i)==0
        Pr0(1,i)=(Pr0(1,i+1)+Pr0(1,i-1))/2;
    end
    if Pr0(2,i)==0
        Pr0(2,i)=(Pr0(2,i+1)+Pr0(2,i-1))/2;
    end
    if Pr0(3,i)==0
        Pr0(3,i)=(Pr0(3,i+1)+Pr0(3,i-1))/2;
    end
    
    if Pr1(1,i)==0
        Pr1(1,i)=(Pr1(1,i+1)+Pr1(1,i-1))/2;
    end
    if Pr1(2,i)==0
        Pr1(2,i)=(Pr1(2,i+1)+Pr1(2,i-1))/2;
    end
    if Pr1(3,i)==0
        Pr1(3,i)=(Pr1(3,i+1)+Pr1(3,i-1))/2;
    end

    if Pr2(1,i)==0
        Pr2(1,i)=(Pr2(1,i+1)+Pr2(1,i-1))/2;
    end
    if Pr2(2,i)==0
        Pr2(2,i)=(Pr2(2,i+1)+Pr2(2,i-1))/2;
    end
    if Pr2(3,i)==0
        Pr2(3,i)=(Pr2(3,i+1)+Pr2(3,i-1))/2;
    end
    
    if Pr3(1,i)==0
        Pr3(1,i)=(Pr3(1,i+1)+Pr3(1,i-1))/2;
    end
    if Pr3(2,i)==0
        Pr3(2,i)=(Pr3(2,i+1)+Pr3(2,i-1))/2;
    end
    if Pr3(3,i)==0
        Pr3(3,i)=(Pr3(3,i+1)+Pr3(3,i-1))/2;
    end
end

ITP=[-1 0 0;0 0 1;0 1 0];  %transform points so ground plane is x-y plane
Ir0=ITP*Pr0;
Ir1=ITP*Pr1;
Ir2=ITP*Pr2;
Ir3=ITP*Pr3;

%marker locations in body-fixed frame
r1=[-332.6;-26.5;82];
r2=[-332.6;26.5;82];
r3=[28.4;142.5;164];
r0=[28.4;-142.5;164];

%solve Wahba's problem with Markley's solution and find Body-Fixed Frame
%Location in I
q=zeros(4,length(t));
psi=0*t;
rB=zeros(3,length(t));
for i=1:length(t)
    A1=Ir0(:,i)-Ir1(:,i);
    B1=r0-r1;
    A2=Ir0(:,i)-Ir3(:,i);
    B2=r0-r3;
    A3=Ir1(:,i)-Ir3(:,i);
    B3=r1-r3;
    M=A1*B1'+A2*B2'+A3+B3';
    [u,s,v]=svd(M);
    TB=u*diag([1,1,det(u)*det(v)])*v.';
    %Decomposing TB into quaternion of form [cos(theta/2;u*sin(theta/2)]
    q(1,i)=((1+trace(TB))^.5)/2;
    if q(1,i)~=0
        q(2,i)=(TB(3,2)-TB(2,3))/(4*q(1,i));
        q(3,i)=(TB(1,3)-TB(3,1))/(4*q(1,i));
        q(4,i)=(TB(2,1)-TB(1,2))/(4*q(1,i));
    else
        q(2,i)=sqrt((TB(1,1)+1)/2);
        q(3,i)=sqrt((TB(2,2)+1)/2);
        q(4,i)=sqrt((TB(3,3)+1)/2);
    end
    psi(i)=atan2(TB(2,1),TB(1,1));  %z-rotation angle assuming only z-rotation.  
    rB(:,i)=((Ir0(:,i)-TB*r0)+(Ir1(:,i)-TB*r1)+(Ir3(:,i)-TB*r3))/3;
end

%remove jumps in motion capture data
rB(:,1)=rB(:,2);
psi(1)=psi(2);
rBcopy=rB;
psicopy=psi;
for i=1:length(t)-1
    if abs(rBcopy(1,i+1)-rBcopy(1,i))>20
        rB(1,i+1)=rB(1,i);
    end
    if abs(rBcopy(2,i+1)-rBcopy(2,i))>20
        rB(2,i+1)=rB(2,i);
    end
    if abs(rBcopy(3,i+1)-rBcopy(3,i))>20
        rB(3,i+1)=rB(3,i);
    end
    if abs(psicopy(i+1)-psicopy(i))>1
        psi(i+1)=psi(i);
    end
end

x=rB(1,:)/1000;   %motion capture is in mm, need to convert to m
y=rB(2,:)/1000;
z=rB(3,:)/1000;
thetaR=Data(:,18);
thetaL=Data(:,19);
psi=unwrap(psi);
ax=Data(:,25);
ay=Data(:,26);
az=Data(:,27);
dutyR=Data(:,20);
dutyL=Data(:,21);
wx=Data(:,22);
wy=Data(:,23);
wz=Data(:,24);
Vbatt=Data(:,28);

data_array=[t,rB(1,:).',rB(2,:).',rB(3,:).',q(1,:).',q(2,:).',q(3,:).',q(4,:).',wx,wy,wz,ax,ay,az,dutyR,dutyL,Vbatt];
filename = sprintf('fixed_data2_%s.txt',num2str(set));
fid=fopen(filename,'w');
%fprintf(fid,'%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\r\n','t (s)','x (m)','y (m)','z (m)','q0','q1','q2','q3','wx (rad/s)','wy (rad/s)','wz (rad/s)','ax (m/s^2)','ay (m/s^2)','az (m/s^2)','dutyR','dutyL','Vbatt (V)');
fprintf(fid,'%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\r\n','t','x','y','z','q0','q1','q2','q3','wx','wy','wz','ax','ay','az','dutyR','dutyL','Vbatt');
%csvwrite(filename, data_array)
for row=1:size(data_array,1)
    fprintf(fid,'%.17f,%.17f,%.17f,%.17f,%.17f,%.17f,%.17f,%.17f,%.17f,%.17f,%.17f,%.17f,%.17f,%.17f,%.17f,%.17f,%.17f\r\n', ...
    data_array(row,:));
end
fclose(fid);


   