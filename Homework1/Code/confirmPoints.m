load image1.txt
load image2.txt
%% 
figure(1);
axis on;
xlabel x;
ylabel y;

im1 = imread('image1.jpg');
imshow(im1);
hold on;
plot(m1(:,1), m1(:,2), 'R+', 'LineWidth', 5, 'MarkerSize',20);
hold off;
%%
figure(2);
axis on;
xlabel x;
ylabel y;

im2 = imread('image2.jpg');
imshow(im2);
hold on;
plot(m2(:,1), m2(:,2), 'R+', 'LineWidth', 5, 'MarkerSize',20);
hold off;
