clc; clear all; close all;

%%
load image1.txt;
load image2.txt;
m1 = image1;
m2 = image2;

z = length(m1);
m1=[m1(:,1) m1(:,2) ones(z,1)];   % m1: 16x3
m2=[m2(:,1) m2(:,2) ones(z,1)];   % m2: 16x3
%%
% Get Intrinsic Matrix
img1 = imread('image1.jpg');
img2 = imread('image2.jpg');
[r ,c ,d] = size(img2); 
cx = c/2;
cy = r/2;
f1 = 1000;  % focal length
f2 = 2000;
K = [f1 0 cx; 0 f2 cy; 0 0 1];  % Intrinsic Matrix
%%
% The matrix for normalization(Centroid)
N=[2/c 0 -1;
    0 2/r -1;
    0 0 1];

% Data Centroid
x1=N*m1'; x2=N*m2';
x1=[x1(1,:)' x1(2,:)'];  
x2=[x2(1,:)' x2(2,:)']; 

% Af=0 
A=[x1(:,1).*x2(:,1) x1(:,2).*x2(:,1) x2(:,1) x1(:,1).*x2(:,2) x1(:,2).*x2(:,2) x2(:,2) x1(:,1) x1(:,2), ones(z,1)];
%%
% Fundamental matrix
[UA, SA, VA] = svd(A);   % Sinal Value Decomposition (SVD)
F=reshape(VA(:,9), 3, 3)';   % reshape the 9th column to 3x3 matrix
% rank(E) = rank(F) = 2 
[UF, SF, VF] = svd(F);
F=UF*diag([SF(1,1) SF(2,2) 0])*VF';
% Denormalize
F = N'*F*N;
% Essential Matrix
E=K'*F*K;
%%
% SVD of E
[UE,SE,VE] = svd(E);
W = [0,-1,0;
    1,0,0;
    0,0,1];        
R = UE*W*VE';  % Roatation matrix
T = UE(:,3);   % Transistion matrix
% P matrix (Camera calibration)
P2 = [R, T];
P1 = [eye(3) zeros(3,1)];  % World coordinate system is centered at 1st camera pinhole
%%
%Get 3D Data using Direct Linear Transform(Linear Triangular method)
inX = [m1(1,:)' m2(1,:)'];
Xw = Triangulation(m1',K*P1, m2',K*P2);
xxx=Xw(1,:);
yyy=Xw(2,:);
zzz=Xw(3,:);

figure(1);
plot3(xxx, yyy, zzz, 'R+');



