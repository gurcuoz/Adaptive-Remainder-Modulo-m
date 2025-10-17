function [ stego_image embedded not_used] = stc_embed_29032021(cover_image, secret_message, code, w)
%clc; clear;
%H_hat = [7 5 1];
%H_hat = [7 5 1 13];
%H_hat = [7 5];
%mm=floor(2^18/size(H_hat,2));3;%33000;
% H = create_pcm_from_submatrix(H_hat, mm)
% draw_pcm(H);
%alfa=0.2;%0.4;%0.6;%0.8;

% code = structure with all necesary components
%code = create_code_from_submatrix(H_hat, mm);
%code.shift(1:floor(code.n*alfa))=1;
%w = 2.^[0:1:code.n-1];%ones(code.n,1);
%w = 2.^(mod([0:1:code.n-1],mm));%ones(code.n,1);12
%w = ones(code.n,1);%ones(code.n,1);
%x = ones(code.n,1); %double(rand(code.n,1)<0.5);
%rng(123);
%m = double(rand(sum(code.shift),1)<0.5);
m=secret_message';
stego_image=cover_image;
M=size(cover_image,1); N=size(cover_image,2);
embedded=size(m,1)*size(m,2);
not_used=M*N-size(m,1)*size(m,2);
if embedded==0
    return;
end
mse_stc=[];
mse_lsb=[];
%for i=1:100
 %filename = cat(2, 'D:\Ucheba\papa\Trudy\Gurcu\Iranian\BOSS\',  char(num2str(i)), '.pgm');  
 %img=imread(filename);
 img=cover_image;
 %imshow(img);
 %x = randi([0, 1], code.n, 1); %double(rand(code.n,1)<0.5);
 x=double(reshape(mod(img,2),[],1));
 [y min_cost] = dual_viterbi(code, x, w, m);
 x = x'; %x
 y = y';% y
 m2=calc_syndrome(code,y)';
 m1 = [m' ; m2];% m
 if  sum(m1(1,:)~=m1(2,:))
     disp('incorrect stc embedding');
     return;
 end
 signs=randi([0 1], 1, size(x,2));
 mask=signs==0;
 signs(mask)=-1;
 stego_image=reshape(stego_image,1,[]);
 lsb=mod(stego_image, 2);
 y2=xor(y,lsb);
 y1=y2.*signs;
 mask1=(stego_image==255) & (y1==1);
 y1(mask1)=-1;
 maskm1=(stego_image==0) & (y1==-1);
 y1(maskm1)=1;
 stego_image1=uint8(double(stego_image)+y1);
 stego_image=reshape(stego_image1, M,N);
 
%  mse_stc=[mse_stc; sum((x-y).^2)/size(x,2)];
%  mse_lsb=[mse_lsb; sum(xor((m(:,1))', x(1:size(m,1))))/size(x,2)];
%  min_cost;
%end
% plot(1:100, mse_stc, 1:100, mse_lsb);
% hold on;
% legend('stc','lsb');
% title('MSE of STC and LSB on 100 runs');
% grid on
% mean_stc=mean(mse_stc);
% mean_lsb=mean(mse_lsb);
% mstc(1:100)=mean_stc; mlsb(1:100)=mean_lsb;
% plot(1:100, mstc, 1:100, mlsb);
% legend('stc','lsb','mstc','mlsb');
end
