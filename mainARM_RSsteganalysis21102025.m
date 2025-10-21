clc
clear 
% ml=4;
%  mu=8;
%  T=160;
%T=176; K=3;
%T=256; K=4;
%T=128;K=3;
%R=[[0 179 3]; [180 255 4]];%  Ri2-Ri1>=2^Ri3
%R=[[0 15 4 0]; [16 63 5 0]; [64 127 6 0]; [128 255 7 0]];%  Ri2-Ri1>=2^Ri3 Khodaei
%R=[[0 15 4 0]; [16 63 5 0]; [64 255 6 0]];%  Ri2-Ri1>=2^Ri3 Modified
%R=[[0 15 4 0]; [16 255 5 0]];%  Ri2-Ri1>=2^Ri3 Modified
%R=[[0 15 3 0]; [16 255 4 0]];%  Ri2-Ri1>=2^Ri3 Modified
%R=[[0 15 2 0]; [16 255 5 0]];%  Ri2-Ri1>=2^Ri3 Modified
%R=[[0 15 3 0]; [16 255 5 0]];%  Ri2-Ri1>=2^Ri3 Modified
%R=[[0 15 2 0]; [16 255 4 0]];%  Ri2-Ri1>=2^Ri3 Modified
%R=[[0 15 3 0]; [16 255 4 0]];%  Ri2-Ri1>=2^Ri3 Modified
%R=[[0 127 3 0]; [128 255 4 0]];%  Ri2-Ri1>=2^Ri3 Modified
%R=[[0 63 3 0]; [64 255 4 0]];%  Ri2-Ri1>=2^Ri3 Modified
%R=[[0 31 3 0]; [32 255 4 0];];%  Ri2-Ri1>=2^Ri3 Modified
%R=[[0 31 3 0]; [32 255 5 0]];%  Ri2-Ri1>=2^Ri3 Modified
%R=[[0 31 3 0]; [32 255 6 0]];%  Ri2-Ri1>=2^Ri3 Modified
%R=[[0 15 3 0]; [16 255 4 0]];%  Ri2-Ri1>=2^Ri3 Modified
%R=[[0 7 3 0]; [8 255 4 0]];%  Ri2-Ri1>=2^Ri3 Modified
%R=[[0 15 3 0]; [16 255 5 0]];%  Ri2-Ri1>=2^Ri3 Modified
%R=[[0 15 3 0]; [16 255 5 0]];%  Ri2-Ri1>=2^Ri3 Modified
%R=[[0 15 4 0]; [16 255 5 0]];%  Ri2-Ri1>=2^Ri3 Modified
%R=[[0 31 5 0]; [32 255 6 0]];%  Ri2-Ri1>=2^Ri3 Modified
R=[[0 15 3 0]; [16 31 4 0]; [32 255 5 0]];%  Ri2-Ri1>=2^Ri3 Modified
%R=[[0 15 2 0]; [16 255 5 0]];%  Ri2-Ri1>=2^Ri3 Modified
%R=[[0 7 2 0]; [8 255 3 0]];%  Ri2-Ri1>=2^Ri3 Modified
%R=[[0 7 2 0]; [8 255 4 0]];%  Ri2-Ri1>=2^Ri3 Modified
%R=[[0 7 3 0]; [8 255 4 0]];%  Ri2-Ri1>=2^Ri3 Modified
%R=[[0 7 2 0]; [8 255 5 0]];%  Ri2-Ri1>=2^Ri3 Modified
%R=[[0 255 0 0]];%  Ri2-Ri1>=2^Ri3 Modified
%R=[[0 255 4 0]];%  Ri2-Ri1>=2^Ri3 Modified
%R=[[0 255 2 0]];%  Ri2-Ri1>=2^Ri3 Modified
%R=[[0 255 3 0]];%  Ri2-Ri1>=2^Ri3 Modified
%R=[[0 255 5 0]];%  Ri2-Ri1>=2^Ri3 Modified
%Khodaei
%T=0; K=0;
%T=256; K=3;
%T=224; K=4;
%T=224; K=3;
%T=240; K=3;
%T=160; K=3;
%T=176; K=3;
T=192; K=3;
%T=208; K=3;
%T=224; K=3;
%T=192; K=3;
%T=256; K=2;
%T=256; K=3;
%T=256; K=4;
%T=256; K=5;
%T=256; K=5;
%T=256; K=6;
%T=256; K=1;
%T=128; K=2;
%T=176; K=2;
%T=184; K=2;
%T=192; K=4;
%T=160; K=4;
%T=32; K=3;
%T=96; K=3;
%T=160; K=3;
%T=128; K=3;
%T=160; K=2;
%T=144; K=2;
%T=80;K=3;
%T=128; K=3;
%T=64; K=3;
%T=160; K=4;
%T=192; K=4;
%T=176; K=3;
%T=0; K=3;
%T=0; K=0;
%T=192; K=3;
%T=200; K=2;
%T=208; K=2;
%T=208; K=3;
%T=32; K=2;
%T=128; K=2;
%T=64; K=2;
%T=128; K=2;
%T=176; K=2;
%T=32; K=2;
%T=32; K=5;
%T=48; K=4;
%T=16; K=4;
%T=144; K=3;
%T=184; K=3;
%T=176; K=3;
%T=168; K=3;
%T=0; K=0
%T=64; K=4;
Freq = [0 0 0 0 0];
%mask=[1 0;0 1];
mask=[0 1;1 0];
%mask=[0 1 1 0];
%mask=[1 0 0 1];
%mask = [ 0 1 1 1];
%mask = [0 1 0];
%mask = [0 0 0; 0 1 0; 0 0 0];
%mask = [0 0 1 1; 1 1 0 0];
%mask = [1 1 0 0 ; 0 0 1 1];
%mask = [0 1 0; 1 1 1; 0 1 0];
%mask = [1 1; 0 1];
%mask = [1 1; 1 1];
%mask = [1 0; 0 0];
%mask = [1 1; 0 0];
%mask = [0 0; 1 1];
%mask = [0 0; 0 1];
% cover_image=imread('C:\Users\nor\Desktop\thesis\GRAY SCALE\LSBCT\zeldaBMP.bmp');
pictures = [
    %"46794713-toys-.jpg"; ...
%     "airplane.tif"; ...
     "baboon.bmp"; ...
 %    "BABOON24012018.bmp"; ...3D image Khodaei works correctly on couple
 %   "Baboon24012018.gif"; ...
 %    "baby.png"; ...
    "barbara.bmp"; ...
%     "lena.bmp"; ... Khodaei works correctly on Lena.bmp
 %    "boat.gif"; ...
  %   "boy.tif"; ...
 %    "cameraman.tif"; ...
 %    "couple.tif"; ... Khodaei works co
     "goldhill.bmp"; ...rrectly on couple
     "elaine.bmp"; ... Khodaei works correctly on elain.bmp
     "girlface.dib.bmp"; ...
%     "home.gif"; ...
%     "Jerusalem.gif"; ... Khodaei works correctly on jerusalem.gif
%   "jet.bmp"; ...
%     "lady.tif"; ...  Khodaei works correctly on Lady.tif
%     "lake.gif"; ...
    "lena.bmp"; ... Khodaei works correctly on Lena.bmp
     "peppers.bmp"; ...
%     "tank.tif"; ... Khodaei works correctly on tank.tif
     "tiffany.jpg"; ...3D image Khodaei works correctly on tank.tif
%     "truck.tif"; ...
     "46794713-toys-gray.jpg"; ...
     "zelda.bmp"; ...%     ];    
%              "zeldaBMP.bmp"; ...
%             "peppersBMP.bmp"; ...
%             "lenaBMP.bmp"; ...
%             "lady.tif"; ...
%            "house.tif"; ...
%             "elainBMP.bmp"; ...
%            "camerman.tif"; ...
%             "BOY.tif"; ...
%             "boat.gif"; ...
%             "barbaraBMP.bmp"; ...
%             "BABY.png"; ...
%             "BaboonBMP.bmp"; ...
%             "airplaneBMP.bmp"
           ];
%start_length = 2*524288;
%start_length = 5*2^18;
mdl=input('Modulo value? ');
  n1=input('Block size vertical (rows)? ');
  n2=input('Block size horizontal (cols)? ');
  Threshold1=input('Threshold='); % 0 instead of 5 used earlier
for t=1:1 % 6
result = [];
  %S=sum(cover_image(:,:));
  %avg=sum(S)/(M*N);
%disp('cover_image       Size_secret_data          PSNR             Actual_PSNR             MSNR             MSE             Ml        Mu         T ')
%disp('                                             dB                  dB                   dB')
%p='zeldaBMP.bmp';
%disp('=================================================================================================================================================')
   
  
  %r=0;MSE=0;PSNR=0;Bpp=0;
  resi=[];
  %rng(123);%rng('default');
  avhdif(1:256)=0;
%   for pict=1:size(pictures,1)
% %         rng(123);
% %         for i=1:9
% %        start_length=(i-1)*2^18;%17;%15;%17; %6;
% %        %secret_message= randi([0 2],1,start_length); 
% %        secret_message= randi([0 1],1,start_length);
%        %secret_message= randi([0 1],1,start_length); 
%        filename = cat(2, 'D:\Ucheba\Masters\Hajer\Hajer Thesis\GRAY SCALE\LSBCT\',  char(pictures(pict)));  
%        %filename = cat(2, 'D:\Ucheba\Masters\Hajer\Hajer Thesis\GRAY SCALE\LSBCT\pictures\pictures\',  char(pictures(pict)));  
%        cover_image=imread(filename);
%        %cover_image = randi([0 255],512,512);
%        %cover_image = cover_image(1:10,1:10);
%        %secret_message = secret_message(1:200);
%        [M N L]=size(cover_image);
%        if L>1
%          a=cover_image(:,:,1);
%          cover_image=a;
%        end
%        if mod(N,2)==0
%           N1=N;
%       else
%           N1=N-1;
%       end
%       if mod(M,2)==0
%           M1=M;
%       else
%           M1=M-1;
%       end
%       a=cover_image(1:M1,1:N1);
%       cover_image=a;
%       [M N L]=size(cover_image);
%       dif=abs(single(cover_image(:,1:2:end))-single(cover_image(:,2:2:end)));
%       hdif=histogram(dif,-0.5:1:255.5);
%       avhdif=hdif.Values+avhdif;
%   end
      
  for iter=1:10  
    %secret_message= randi([0 1],1,start_length); 
    for pict=1:9 %1:size(pictures,1)
        rng(123);
        for i=1:9
       %start_length=(i-1)*2^18;%17;%15;%17; %6;
       start_length=(i-1)*2^(18-log2(mdl))*mdl/8;%17;%15;%17; %6;
       %secret_message= randi([0 2],1,start_length); 
       %secret_message= randi([0 1],1,start_length);
       secret_message= randi([0 mdl-1],1,start_length);
       %secret_message= randi([0 1],1,start_length); 
       filename = cat(2, 'D:\Ucheba\Masters\Hajer\Hajer Thesis\GRAY SCALE\LSBCT\',  char(pictures(pict)));  
       %filename = cat(2, 'D:\Ucheba\Masters\Hajer\Hajer Thesis\GRAY SCALE\LSBCT\pictures\pictures\',  char(pictures(pict)));  
       cover_image=imread(filename);
       %cover_image = randi([0 255],512,512);
       %cover_image = cover_image(1:10,1:10);
       %secret_message = secret_message(1:200);
       [M N L]=size(cover_image);
       if L>1
         a=cover_image(:,:,1);
         cover_image=a;
       end
       if mod(N,2)==0
          N1=N;
      else
          N1=N-1;
      end
      if mod(M,2)==0
          M1=M;
      else
          M1=M-1;
      end
      a=cover_image(1:M1,1:N1);
      cover_image=a;
      [M N L]=size(cover_image);
      
       %secret_message = mod(cover_image,2);
       %secret_message = reshape([secret_message secret_message],1,[]);

    % [e r]=size(secret_message);
       %[stego_image r not_used] = Embedding23112017( R,cover_image,secret_message);
       [ stego_image r, MSE] = adaptiveLSB29092023(cover_image, secret_message, mdl, n1, n2, Threshold1);
        r=0; not_used=0;
       %[stego_image r not_used Freqcase1 R] = EmbeddingKhodaei13052018( R,T, K, cover_image,secret_message);
       %[stego_image r not_used Freqcase1 R] = EmbeddingKhodaei08042018( R,T, K, cover_image,secret_message);
       %[stego_image r not_used Freqcase1 R] = EmbeddingKhodaei18042018( R,T, K, cover_image,secret_message);
%        Freq(1) = Freq(1) + Freqcase1;
%        for i1=1:size(R,1)
%          Freq(i1+1) = Freq(i1+1) + R(i1,4);
%        end
       if r>1
       output = extractionKhodaei08042018(R, T, K, stego_image,r-1, secret_message);
% % % disp('=====================================================================================')
%  ii=0;
%  for i=1:r-1
%     e(i)=output(i)-char(secret_message(i)+48);
%     if (e(i)~=0)
%         ii=ii+1;
%     end
%  end
% if ii~=0
%     break;
% end
%resi = [resi ii];
       %ii = output~=char(secret_message(1:r-1)+48);
       ii = output~=char(secret_message(1:size(output,2))+48);
       if sum(ii)>0
           break;
       end
       r = r-1;
       end
%        MSE=0;
%        for i=1:M
%          for j=1:N
%            MSE=MSE+(double(cover_image(i,j))-double(stego_image(i,j)))^2;
%          end
%        end
%        MSE=MSE/(M*N);
%-----------------------PSNR-----------------------
%       MSNR=10*log10(avg*avg/MSE);
       for pxl=1:256
         hc(pxl)=sum(sum(cover_image==pxl-1));
         hs(pxl)=sum(sum(stego_image==pxl-1));
         hcdif(255+pxl)=sum(sum(double(cover_image(:,1:2:N1))-double(cover_image(:,2:2:N1))==pxl-1));
         hsdif(255+pxl)=sum(sum(stego_image(:,1:2:N1)-stego_image(:,2:2:N1)==pxl-1));
         if pxl<256
           hcdif(pxl)=sum(sum(double(cover_image(:,1:2:N1))-double(cover_image(:,2:2:N1))==pxl-256));
           hsdif(pxl)=sum(sum(stego_image(:,1:2:N1)-stego_image(:,2:2:N1)==pxl-256));
         end
       end
       %plot(-150:150, hcdif(106:406),'red',-150:150,hsdif(106:406),'blue');
       maxdifdif(pict)=max(abs(hcdif-hsdif));
       %[RM SM UM RMM SMM UMM] = RSsteganalysis(stego_image);
       [RM SM UM RMM SMM UMM] = RSsteganalysis30042018(stego_image,mask);
       RSres(i,:)=[RM SM UM RMM SMM UMM];
       %MD=fspam(stego_image, 4);
       %MD = fspam06052018(stego_image, 4);
     end
     RSres=RSres/sum(RSres(1,1:3));
x=linspace(0,1,9);
figure;
%subplot(3,1,1);
%subplot(2,1,1);
plot(x,RSres(1:9,1),'k',x,RSres(1:9,2),'k--',x,RSres(1:9,4),'k-o',x,RSres(1:9,5),'k--o');
legend('RM','SM','R-M', 'S-M');
title(pictures(pict));
%figure;
%subplot(3,1,2);
% for pxl=1:256
%   hc(pxl)=sum(sum(cover_image==pxl-1));
%   hs(pxl)=sum(sum(stego_image==pxl-1));
%   hcdif(255+pxl)=sum(sum(abs(cover_image(:,1:2:N)-cover_image(:,2:2:N))==pxl-1));
%   hsdif(255+pxl)=sum(sum(abs(stego_image(:,1:2:N)-stego_image(:,2:2:N))==pxl-1));
%   if pxl<256
%     hcdif(pxl)=sum(sum(abs(double(cover_image(:,1:2:N))-double(cover_image(:,2:2:N)))==pxl-256));
%     hsdif(pxl)=sum(sum(abs(stego_image(:,1:2:N)-stego_image(:,2:2:N))==pxl-256));
%   end
% end
%subplot(3,1,3);
%subplot(2,1,2);
%plot(-150:150, hcdif(106:406),'red',-150:150,hsdif(106:406),'blue');
% hc=histogram(cover_image, 256,'FaceColor','red');
% subplot(3,1,3);
% %hold on;
% hs=histogram(stego_image,256,'FaceColor','blue','LineStyle','--');
%legend('Cover image','Stego image');
% %title(pictures(pict));
maxdifr(pict)=max(abs(hc-hs))/mean(hc);
maxdifdifr(pict)=max(abs(hcdif-hsdif))/mean(hcdif);
maxdif(pict)=max(abs(hc-hs));
maxdifdif(pict)=max(abs(hcdif-hsdif));
hzerodif(pict,:)=[hcdif(256) hsdif(256) hcdif(256)/hsdif(256) pictures(pict)];
       MSE = sum(sum((double(cover_image)-double(stego_image)).^2))/M/N;
       PSNR=10*log10(255^2/(MSE));
       steg_m=sum(sum(double(stego_image)))/M/N;
       cover_m=sum(sum(double(cover_image)))/M/N;
       steg_sig=sum(sum((double(stego_image)-steg_m).^2))/(M*N-1);
       cover_sig=sum(sum((double(cover_image)-cover_m).^2))/(M*N-1);
       cover_stego_cor=sum(sum((double(cover_image)-cover_m).*(double(stego_image)-steg_m)))/(M*N-1);
       Q_cover_stego=4*cover_stego_cor*cover_m*steg_m/(cover_sig+steg_sig)/(cover_m^2+steg_m^2);
       % max_value=max(max(cover_image(:,:)));
       Bpp=r/(M*N);
       result = cat(1, result, [MSE PSNR Bpp Q_cover_stego not_used pictures(pict)]);
    end%pict
  %avgr es = mean(result(:,1:3));
  end %iter

 resi(t,1:4)=mean(str2double(result(:,1:4)));
 end %on t
 ax1=subplot(2,2,1);
 ax2=subplot(2,2,2);
 ax3=subplot(2,2,3);
 ax4=subplot(2,2,4);
 
 plot(ax1,40:40:240, resi(:,1));
 title(ax1,'MSE dependence on threshold, T');
 xlabel(ax1,'Threshold, T');
 ylabel(ax1, 'MSE');
 xlim(ax1, [40 240]);
 grid(ax1,'on');
 plot(ax2,40:40:240, resi(:,2));
 title(ax2,'PSNR dependence on threshold, T');
 xlabel(ax2,'Threshold, T');
 ylabel(ax2,'PSNR');
 xlim(ax2, [40 240]);
 grid(ax2,'on');
 plot(ax3,40:40:240, resi(:,3));
 title(ax3,'BPP dependence on threshold, T');
 xlabel(ax3,'Threshold, T');
 ylabel(ax3,'BPP');
 xlim(ax3, [40 240]);
 grid(ax3,'on');
 plot(ax4,40:40:240, resi(:,4));
 title(ax4,'Image quality, IQ, dependence on threshold, T');
 xlabel(ax4,'Threshold, T');
 ylabel(ax4,'IQ');
 xlim(ax4, [40 240]);
 grid(ax4,'on');
fActual_PSNR=10*log10(double(max_value)*double(max_value)/MSE);   
disp(sprintf('%s            %d               %f          %f           %f           %f          %d         %d         %d' ,p,start_length,PSNR,Actual_PSNR,MSNR,MSE,ml,mu,T));
% %*********************************************************
T=160;
 ml=4;
  mu=8;
   start_length =584288; 
  secret_massege= randi([0 1],1,start_length); 
  [e r]=size(secret_massege);

   stego_image=Embedding( ml,mu,T,cover_image,secret_massege);
   MSE=0;
for i=1:M
    for j=1:N
        
     MSE=MSE+(double(cover_image(i,j))-double(stego_image(i,j)))^2;
    end
end
MSE=MSE/(M*N);
%-----------------------PSNR-----------------------
 MSNR=10*log10(avg*avg/MSE);
PSNR=10*log10(255^2/(MSE));
  max_value=max(max(cover_image(:,:)));
Bpp=(start_length)/(M*N);
Actual_PSNR=10*log10(double(max_value)*double(max_value)/MSE);   
disp(sprintf('%s            %d               %f          %f           %f           %f          %d         %d         %d' ,p,start_length,PSNR,Actual_PSNR,MSNR,MSE,ml,mu,T));
% %*************************************************************************
T=160;
 ml=8;
  mu=16;
   start_length =644288; 
  secret_massege= randi([0 1],1,start_length); 
  [e r]=size(secret_massege);

   stego_image=Embedding( ml,mu,T,cover_image,secret_massege);
   MSE=0;
for i=1:M
    for j=1:N
        
     MSE=MSE+(double(cover_image(i,j))-double(stego_image(i,j)))^2;
    end
end
MSE=MSE/(M*N);
%-----------------------PSNR-----------------------
 MSNR=10*log10(avg*avg/MSE);
PSNR=10*log10(255^2/(MSE));
  max_value=max(max(cover_image(:,:)));
Bpp=(start_length)/(M*N);
Actual_PSNR=10*log10(double(max_value)*double(max_value)/MSE);   
disp(sprintf('%s            %d               %f          %f           %f           %f          %d         %d         %d' ,p,start_length,PSNR,Actual_PSNR,MSNR,MSE,ml,mu,T));
% %**************************************************
T=160;
 ml=8;
  mu=16;
   start_length =704288; 
  secret_massege= randi([0 1],1,start_length); 
  [e r]=size(secret_massege);

   stego_image=Embedding( ml,mu,T,cover_image,secret_massege);
   MSE=0;
for i=1:M
    for j=1:N
        
     MSE=MSE+(double(cover_image(i,j))-double(stego_image(i,j)))^2;
    end
end
MSE=MSE/(M*N);
%-----------------------PSNR-----------------------
 MSNR=10*log10(avg*avg/MSE);
PSNR=10*log10(255^2/(MSE));
  max_value=max(max(cover_image(:,:)));
Bpp=(start_length)/(M*N);
Actual_PSNR=10*log10(double(max_value)*double(max_value)/MSE);   
disp(sprintf('%s            %d               %f          %f           %f           %f          %d         %d         %d' ,p,start_length,PSNR,Actual_PSNR,MSNR,MSE,ml,mu,T));
% %*************************************************************************
T=160;
 ml=8;
  mu=16;
   start_length =734288; 
  secret_massege= randi([0 1],1,start_length); 
  [e r]=size(secret_massege);

   stego_image=Embedding( ml,mu,T,cover_image,secret_massege);
   MSE=0;
for i=1:M
    for j=1:N
        
     MSE=MSE+(double(cover_image(i,j))-double(stego_image(i,j)))^2;
    end
end
MSE=MSE/(M*N);
%-----------------------PSNR-----------------------
 MSNR=10*log10(avg*avg/MSE);
PSNR=10*log10(255^2/(MSE));
  max_value=max(max(cover_image(:,:)));
Bpp=(start_length)/(M*N);
Actual_PSNR=10*log10(double(max_value)*double(max_value)/MSE);   
disp(sprintf('%s            %d               %f          %f           %f           %f          %d         %d         %d' ,p,start_length,PSNR,Actual_PSNR,MSNR,MSE,ml,mu,T));
% %************************************************************************************
T=160;
 ml=8;
  mu=16;
   start_length =764288; 
  secret_massege= randi([0 1],1,start_length); 
  [e r]=size(secret_massege);

   stego_image=Embedding( ml,mu,T,cover_image,secret_massege);
   MSE=0;
for i=1:M
    for j=1:N
        
     MSE=MSE+(double(cover_image(i,j))-double(stego_image(i,j)))^2;
    end
end
MSE=MSE/(M*N);
%-----------------------PSNR-----------------------
 MSNR=10*log10(avg*avg/MSE);
PSNR=10*log10(255^2/(MSE));
  max_value=max(max(cover_image(:,:)));
Bpp=(start_length)/(M*N);
Actual_PSNR=10*log10(double(max_value)*double(max_value)/MSE);   
disp(sprintf('%s            %d               %f          %f           %f           %f          %d         %d         %d' ,p,start_length,PSNR,Actual_PSNR,MSNR,MSE,ml,mu,T));
% %********************************************************************************
T=160;
ml=8;
  mu=16;
   start_length =824288; 
  secret_massege= randi([0 1],1,start_length); 
  [e r]=size(secret_massege);

   stego_image=Embedding( ml,mu,T,cover_image,secret_massege);
   MSE=0;
for i=1:M
    for j=1:N
        
     MSE=MSE+(double(cover_image(i,j))-double(stego_image(i,j)))^2;
    end
end
MSE=MSE/(M*N);
%-----------------------PSNR-----------------------
 MSNR=10*log10(avg*avg/MSE);
PSNR=10*log10(255^2/(MSE));
  max_value=max(max(cover_image(:,:)));
Bpp=(start_length)/(M*N);
Actual_PSNR=10*log10(double(max_value)*double(max_value)/MSE);   
disp(sprintf('%s            %d               %f          %f           %f           %f          %d         %d         %d' ,p,start_length,PSNR,Actual_PSNR,MSNR,MSE,ml,mu,T));
% %   ************************************************************************
figure
subplot(2,2,[1,3]);
 imshow(cover_image);
title('Cover image')
subplot(2,2,[2,4]);
imshow(cover_image);
title('Stego image')

%    output = extraction( stego_image,T,mu,ml,r);
% % % disp('=====================================================================================')
% ii=0;
% for i=1:r
%     e(i)=output(i)-secret_massege(i);
%     if (e(i)~=0)
%         ii=ii+1;
%     end
% end
% ii
%====================================================================================================================================
