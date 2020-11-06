clear
clc
close all;

T1 = load('output.csv');

time1 = T1(:,3).*1000;
gnpr1 = T1(:,2);

% T2 = load('output2.csv');
% 
% time2 = T2(:,3).*1000;
% gnpr2 = T2(:,2);
% 
% T3 = load('output3.csv');
% 
% time3 = T3(:,3).*1000;
% gnpr3 = T3(:,2);

figure
semilogx(gnpr1,gnpr1./time1,'-sq','LineWidth',2)
% hold on
% semilogx(gnpr2,gnpr2./time2,'-sq','LineWidth',2)
% hold on
% semilogx(gnpr3,gnpr3./time3,'-sq','LineWidth',2)
% hold off;
ylabel('nodes per rank per sec');
xlabel('nodes per rank');
