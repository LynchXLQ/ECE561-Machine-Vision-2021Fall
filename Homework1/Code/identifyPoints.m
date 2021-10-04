clc;
clear all;
close all;

%%
im1 = imread('image1.jpg');
figure(1);
imshow(im1);
axis on;
xlabel x;
ylabel y;
m1 = ginput(16);
% save ('image1.txt', m1, '.txt');
fid1=fopen('image1.txt','wt'); 
[r,c]=size(m1);            
 for i=1:r
  for j=1:c
  fprintf(fid1,'%f\t',m1(i,j));
  end
  fprintf(fid1,'\r');
 end
fclose(fid1);  
%%
im2 = imread('image2.jpg');
figure(2);
imshow(im2);
axis on;
xlabel x;
ylabel y;
m2 = ginput(16);
% save ('image2.txt', m1, '.txt');
fid2=fopen('image2.txt','wt'); 
[r,c]=size(m2);            
 for i=1:r
  for j=1:c
  fprintf(fid2,'%f\t',m2(i,j));
  end
  fprintf(fid2,'\r');
 end
fclose(fid2);  