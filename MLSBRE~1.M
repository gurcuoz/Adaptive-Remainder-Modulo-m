function [ stego_image embedded MSE] = mlsb_embed_30032021(cover_image, secret_message, n1, n2, Thr)
%n1, n2 - block size, Thr - threshold for differences
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
m=secret_message;
stego_image=cover_image;
M=floor(size(cover_image,1)/n1)*n1; N=floor(size(cover_image,2)/n2)*n2;
embedded=size(m,1)*size(m,2);
%not_used=M*N-embedded;
if embedded==0
    return;
end
% mse_stc=[];
% mse_lsb=[];
%for i=1:100
 %filename = cat(2, 'D:\Ucheba\papa\Trudy\Gurcu\Iranian\BOSS\',  char(num2str(i)), '.pgm');  
 %img=imread(filename);
 img=cover_image(1:M,1:N);
 %split image into n1 x n2-sized blocks
 img1=reshape(img, M,n2,[]);%make vertical strips M x n2 x ( N/n2)
 img2=permute(img1,[2 1 3]);%make strips horizontal n2 x M x (N/n2)
 img3=reshape(img2, n2,n1,[]);%split into n2 x n1 x M*N/n1/n2
 img4=permute(img3, [2 1 3]);% transpose blocks to n1 x n2 x M*N/n1/n2
 img5=img4(:,:,2:2:end);% take even numbered blocks
 img6=permute(img5,[2 1 3]);% transpose odd numbered blocks to consider verticall pairs
 img7=img4;
 img7(:,:,2:2:end)=img6;%blocks are prepared for embedding
 img8=single(reshape(img7, 2, []));%reshape image to pairs
 img7=[]; img6=[];img5=[];img4=[];img3=[];img2=[];img1=[];
 img71=reshape(img8, n1, n2,[]);%return original shape
 img61=img71(:,:,2:2:end);%get even blocks transposed
 img62=permute(img61,[2 1 3]);%transpose transposed blocks
 img72=img71;
 img72(:,:,2:2:end)=img62;%return retransposed back
 img41=permute(img72, [2 1 3]);%transpose blocks to n2 x n1
 img31=reshape(img41, n2, M, []);%get back horizontal strips
 img21=permute(img31, [2 1 3]);%get back vertical strips
 img11=reshape(img21, M, []);%get back original image
 
 m1=reshape(m, 2,[]);%reshape secret to pairs
 szm1=min(size(m1,2),size(img8,2));
 embedded=2*szm1;
 dif=abs(img8(1,1:szm1)-img8(2,1:szm1));
 dif1=sort(dif,'descend');
 T=dif1(szm1);%threshold is the difference of the pair in the sorted array with index equal to the mesage size
 if T>31
     T=31;
 end% max threshold is 31
 mask=dif>=T;
 if sum(mask)<szm1
     disp('Incorrect threshold');
     return;
 end
 indmask=find(mask);
 mask1=mod(img8(1,indmask(1:szm1)),2)==m1(1,1:szm1) & f(img8(1,indmask(1:szm1)), img8(2,indmask(1:szm1)))== m1(2,1:szm1);
 mask2=mod(img8(1,indmask(1:szm1)),2)~=m1(1,1:szm1) & f(img8(1,indmask(1:szm1))-1, img8(2,indmask(1:szm1)))== m1(2,1:szm1);
 mask3=mod(img8(1,indmask(1:szm1)),2)==m1(1,1:szm1) & f(img8(1,indmask(1:szm1)), img8(2,indmask(1:szm1)))~= m1(2,1:szm1);
 mask4=mod(img8(1,indmask(1:szm1)),2)~=m1(1,1:szm1) & f(img8(1,indmask(1:szm1))-1, img8(2,indmask(1:szm1)))~= m1(2,1:szm1);
 if sum(mask2)>0
    indmask2=find(mask2);   
    img8(1,indmask(indmask2))=img8(1,indmask(indmask2))-1;
    mask21=img8(1,indmask(indmask2))<0 & abs(img8(1,indmask(indmask2))-img8(2,indmask(indmask2)))<=34 ;
    if sum(mask21)>0%adjust FOB start
        indmask21=find(mask21);
        img8(1,indmask(indmask2(indmask21)))=img8(1,indmask(indmask2(indmask21)))+4;
        img8(2,indmask(indmask2(indmask21)))=img8(2,indmask(indmask2(indmask21)))+4;
    end
    mask21=img8(1,indmask(indmask2))<0 & abs(img8(1,indmask(indmask2))-img8(2,indmask(indmask2)))>34 ;
    if sum(mask21)>0%adjust FOB start
        indmask21=find(mask21);
        img8(1,indmask(indmask2(indmask21)))=img8(1,indmask(indmask2(indmask21)))+4;
    end
 end
 if sum(mask4)>0
    indmask4=find(mask4);   
    img8(1,indmask(indmask4))=img8(1,indmask(indmask4))+1;
    mask41=img8(1,indmask(indmask4))>255 & abs(img8(1,indmask(indmask4))-img8(2,indmask(indmask4)))<=34 ;;
    if sum(mask41)>0%adjust FOB start
        indmask41=find(mask41);
        img8(1,indmask(indmask4(indmask41)))=img8(1,indmask(indmask4(indmask41)))-4;
        img8(2,indmask(indmask4(indmask41)))=img8(2,indmask(indmask4(indmask41)))-4;
    end
    mask42=img8(1,indmask(indmask4))>255 & abs(img8(1,indmask(indmask4))-img8(2,indmask(indmask4)))>34 ;;
    if sum(mask42)>0%adjust FOB start
        indmask42=find(mask42);
        img8(1,indmask(indmask4(indmask42)))=img8(1,indmask(indmask4(indmask42)))-4;
    end
 end
 if sum(mask3)>0
     globalStream = RandStream.getGlobalStream;
     myState = globalStream.State;
    indmask3=find(mask3);
    r=randi([0 1], 1, size(indmask3,2));
    rmask=r==0;
    r(rmask)=-1;
    img8(2,indmask(indmask3))=img8(2,indmask(indmask3))+r;
    globalStream.State = myState;
    mask31=img8(2,indmask(indmask3))>255 & abs(img8(1,indmask(indmask3))-img8(2,indmask(indmask3)))<=32;
    if sum(mask31)>0%adjust FOB start
        indmask31=find(mask31);
        img8(2,indmask(indmask3(indmask31)))=img8(2,indmask(indmask3(indmask31)))-2;
        img8(1,indmask(indmask3(indmask31)))=img8(1,indmask(indmask3(indmask31)))-4;
    end
    mask32=img8(2,indmask(indmask3))>255 & abs(img8(1,indmask(indmask3))-img8(2,indmask(indmask3)))>32;
    if sum(mask32)>0%adjust FOB start
        indmask32=find(mask32);
        img8(2,indmask(indmask3(indmask32)))=img8(2,indmask(indmask3(indmask32)))-2;
    end
    mask33=img8(2,indmask(indmask3))<0 & abs(img8(1,indmask(indmask3))-img8(2,indmask(indmask3)))>32;
    if sum(mask33)>0%adjust FOB start
        indmask33=find(mask33);
        img8(2,indmask(indmask3(indmask33)))=img8(2,indmask(indmask3(indmask33)))+2;
    end
    mask34=img8(2,indmask(indmask3))<0 & abs(img8(1,indmask(indmask3))-img8(2,indmask(indmask3)))<=32;
    if sum(mask34)>0%adjust FOB start
        indmask34=find(mask34);
        img8(2,indmask(indmask3(indmask34)))=img8(2,indmask(indmask3(indmask34)))+2;
        img8(1,indmask(indmask3(indmask34)))=img8(1,indmask(indmask3(indmask34)))+4;
    end
 end%adjusting of falling off boundary finished
 maskFOB=img8>255 | img8<0;
 if sum(sum(maskFOB))>0
     disp('Falling off boundary');
     return;
 end
 % adjusting of intervals falling below T start
 dif_after_emb=abs(img8(1,:)-img8(2,:));
 %mask_after_embLT=dif_after_emb<T & mask;
 mask_after_embLT=dif_after_emb(indmask)<T & mask;
 if sum(mask_after_embLT)
     indmaskLT=find(mask_after_embLT);
     maskLT1=img8(1,indmaskLT)>=img8(2,indmaskLT) & img8(2,indmaskLT)>=2;%find pairs with d<T s.t. 1st>=2nd & 1st<255
     if sum(maskLT1)>0%increment 1st
       indmaskLT1=find(maskLT1);
       img8(2,indmaskLT(indmaskLT1))=img8(2,indmaskLT(indmaskLT1))-2;
     end
     maskLT1=img8(1,indmaskLT)>=img8(2,indmaskLT) & img8(2,indmaskLT)<2;%find pairs with d<T s.t. 1st>=2nd & 1st<255
     if sum(maskLT1)>0%increment 1st
       indmaskLT1=find(maskLT1);
       img8(1,indmaskLT(indmaskLT1))=img8(1,indmaskLT(indmaskLT1))+4;
     end
     maskLT2=img8(1,indmaskLT)<img8(2,indmaskLT) & img8(2,indmaskLT)<254;%find pairs with d<T s.t. 1st<2nd & 2nd<255
     if sum(maskLT2)>0%increment 2nd
       indmaskLT2=find(maskLT2);
       img8(2,indmaskLT(indmaskLT2))=img8(2,indmaskLT(indmaskLT2))+2;
     end
     maskLT2=img8(1,indmaskLT)<img8(2,indmaskLT) & img8(2,indmaskLT)>=254;%find pairs with d<T s.t. 1st<2nd & 2nd<255
     if sum(maskLT2)>0%increment 2nd
       indmaskLT2=find(maskLT2);
       img8(1,indmaskLT(indmaskLT2))=img8(1,indmaskLT(indmaskLT2))-4;
     end
 end%adjusting intrrvals below T fisnished
 difcheck=abs(img8(1,:)-img8(2,:));
 mask_dif_check=difcheck>=T;
 %if sum(mask~=mask_dif_check)>0
if sum(mask~=mask_dif_check(indmask))>0
     disp('Different differences above T');
     return;
 end
 img71=reshape(img8, n1, n2,[]);%return original shape
 img61=img71(:,:,2:2:end);%get even blocks transposed
 img62=permute(img61,[2 1 3]);%transpose transposed blocks
 img72=img71;
 img72(:,:,2:2:end)=img62;%return retransposed back
 img41=permute(img72, [2 1 3]);%transpose blocks to n2 x n1
 img31=reshape(img41, n2, M, []);%get back horizontal strips
 img21=permute(img31, [2 1 3]);%get back vertical strips
 img11=reshape(img21, M, []);%get back original image
 stego_image(1:M,1:N)=uint8(img11);
 MSE=sum(sum((single(stego_image)-single(cover_image)).^2))/size(stego_image,1)/size(stego_image,2);
 img13=reshape(stego_image(1:M,1:N), M,n2,[]);%make vertical strips M x n2 x ( N/n2)
 img23=permute(img13,[2 1 3]);%make strips horizontal n2 x M x (N/n2)
 img33=reshape(img23, n2,n1,[]);%split into n2 x n1 x M*N/n1/n2
 img43=permute(img33, [2 1 3]);% transpose blocks to n1 x n2 x M*N/n1/n2
 img53=img43(:,:,2:2:end);% take even numbered blocks
 img63=permute(img53,[2 1 3]);% transpose odd numbered blocks to consider verticall pairs
 img73=img43;
 img73(:,:,2:2:end)=img63;%blocks are prepared for embedding
 img83=single(reshape(img73, 2, []));%reshape image to pairs
 difext=abs(img83(1,:)-img83(2,:));
 maskext=difext>=T;
 %if sum(maskext~=mask)
 if sum(maskext(indmask)~=mask)     
     disp('Diffetent pairs for threshold T');
     return;
 end
 mm=mod(img83(1:2,indmask),2);
 mm(2,:)=f(img83(1,indmask),img83(2,indmask));
 if sum(sum(mm(1:2,1:szm1)~=m1(1:2, 1:szm1)))
     disp('Incorrect extraction');
     return;
 end
 
 %imshow(img);
 %x = randi([0, 1], code.n, 1); %double(rand(code.n,1)<0.5);
%  x=double(reshape(mod(img,2),[],1));
%  [y min_cost] = dual_viterbi(code, x, w, m);
%  x = x'; %x
%  y = y';% y
%  m1 = [m' ; calc_syndrome(code,y)'];% m
%  if  sum(m1(1,:)~=m1(2,:))
%      disp('incorrect stc embedding');
%      return;
%  end
%  signs=randi([0 1], 1, size(m,2));
%  mask=signs==0;
%  signs(mask)=-1;
%  stego_image=reshape(stego_image,1,[]);
%  lsb=mod(stego_image(1:size(m,1)), 2);
%  y2=xor(m',lsb);
%  y1=y2.*signs;
%  mask1=(stego_image(1:size(m,1))==255) & (y1==1);
%  y1(mask1)=-1;
%  maskm1=(stego_image(1:size(m,1))==0) & (y1==-1);
%  y1(maskm1)=1;
%  stego_image1=stego_image;
%  stego_image1(1:size(m,1))=uint8(double(stego_image(1:size(m,1)))+y1);
%  if sum(mod(stego_image1(1:size(m,1)),2)~=m')
%      disp('incorect mlsb embedding');
%      return;
%  end
%  stego_image=reshape(stego_image1, M,N);
 
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
function x=f(x1,x2)
 x=mod(floor(x1/2)+x2,2);
end
