% Find Point in 3D space from given x1,P1,x2,P2
% [x1] * P1 * Xw = 0
% [x2] * P2 * Xw = 0

function TR = Triangulation_selfdesigned(x1, x2, P1, P2)  
% x1,x2: points in 2D image
% P1,P2: P matrix for the camera

[~,cl] = size(x1);
for i=1:cl
    Lx1 = x1(:,i);
    Lx2 = x2(:,i);
    
    A1 = Lx1(1,1).*P1(3,:) - P1(1,:);
    A2 = Lx1(2,1).*P1(3,:) - P1(2,:);
    A3 = Lx2(1,1).*P2(3,:) - P2(1,:);
    A4 = Lx2(2,1).*P2(3,:) - P2(2,:);
    
    A = [A1;A2;A3;A4];
    [~,~,V] = svd(A);
    
    XYZ = V(:,4);  % coordinate of point in 3D senario is the last column of V
    XYZ = rdivide(XYZ, repmat(XYZ(4,1),4,1));   
    TR(:,i) = XYZ;
    
end
