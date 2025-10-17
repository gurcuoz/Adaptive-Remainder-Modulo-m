function [ stego_image r, MSE] = adaptive01052021(cover_image, secret_message, mdl, n1, n2, Threshold);
% clc;
% clear; 
% %uf=figure;
% % Rtab=uitable(uf);
% % Rtab.Data={0, 15, 3, 0; 16, 31, 4, 0; 32, 255, 5, 0};
% % Rtab.ColumnName={'Lower', 'Upper', 'Bit no', 'used'};
% %Thresholdedit = uicontrol(uf,'Style','edit',...
% %         'Position',[0 0 60 15],'Value', 192,...
% %         'String','Treshold', 'CallBack',@getThreshold);
% global Threshold;
%     
% %R=[[0 15 3 0]; [16 31 4 0]; [32 255 5 0]];%  Ri2-Ri1>=2^Ri3 Modified
% R=[[0 255 1 0]];%  Ri2-Ri1>=2^Ri3 Modified
% %T=192; K=3;
% T=256; K=3;%K=1;
% %T=Threshold; K=1;
% Freq = [0 0 0 0 0];
% %mask=[1 0;0 1];
% mask=[0 1;1 0];
% %mask = [0 0; 1 1];
% %mask = [0 0; 0 1];
% % cover_image=imread('C:\Users\nor\Desktop\thesis\GRAY SCALE\LSBCT\zeldaBMP.bmp');
% %start_length = 2*524288;
% %start_length = 5*2^18;
% %for t=1:1 % 6
% result = [];
% resi=[];
% rng(123);
% %        pict=1;
% folder1=2; folder2=2;
% % training=input('Training? Yes - 1; No - 0: ');
% folder1=input('Folder?/2..10 = ');
% % method=input('Method?/Khodaei - 1; PM(M) - 2: ATD - 3: STC - 4; MLSB - 5; OPAP - 6; LSB - 7; OptimalOPAP - 8');
% % if method==2
% %     mdl=input('Modulo value? ');
% %     R=[[0 255 mdl]];
% % elseif method==1 %Khodaei
% %     mdl=2;
% % elseif method==3 %method=3 ATD
% %     mdl=3;
% % elseif method==4
% %     %alpha=input('Alpha value? ');
% %     H_hat = [141 179 203 255]; %[253 199 251 167];
% %     w=size(H_hat,2);
% %     alfa0=1/w;
% %     mdl=2;
% % elseif method==5
% %     mdl=2;
% %     %alpha=input('Alpha value? ');
% % elseif method==6 | method==7 | method==8
%      mdl=input('Modulo value? ');
% % end
% % %chunk= mdl^floor(13/log2(mdl));
% chunk= 3*2^13;%floor(2^14/log2(mdl));%floor(2^13/log2(mdl));
% folder2=folder1;
%  %folder2=10;
%      for i=folder1:folder2 %4:4 %3:3%1:2%9
%        %start_length=(i-1)*2^19;%19;%13;%4;%5;%6; %7%; 8;%17;%18;%17;%15;%17; %6;
%         start_length=(i-2)*chunk;%12;%19;%13;%4;%5;%6; %7%; 8;%17;%18;%17;%15;%17; %6;
%        %secret_message= randi([0 2],1,start_length); 
%         secret_message= randi([0 mdl-1],1,start_length);      
%        %secret_message= randi([0 1],1,start_length);
%        %path='D:\Ucheba\Masters\Hajer\Hajer Thesis\GRAY SCALE\LSBCT\pictures\pictures';
%        %[FileName,path] = uigetfile('*.*','Select the images folder');
% %        if training==1
% %            names=fullfile(path,{'*.bmp','*.jpg','*.tiff'});%for training
% %        else
% %            names=fullfile(path,{'*.bmp','*.jpg','*.tiff', '*.tif','*.png'});%for testing
% %        end
%           path1='C:\Temp';
%           path1='C:\Temp\ucid.v2';
%           %path='G:\Pictures';
%           %path='C:\Temp\2';
%           path='C:\Temp\ucid.v2';
% %         names=fullfile(path,'\1\','*.tif');%for training 
%           names=fullfile(path,'*.tif');%for training 
% %        names_n=numel(names);
%         filenames=dir(char(names));
%         names_n=numel(filenames);
%         files={};
%         for in=1:names_n
% %         filenames=dir(char(names(in)));
%          files=cat(2,files,{filenames(in).name});
%         end
% 
%        %diri=fullfile(path,char(i+48)); 
%        diri=fullfile(path1,int2str(i)); 
%        [status, msg, msgID]= mkdir (diri);
% %        if training==1
% %            filestart=1; fileend=names_n/2;
% %        else
% %            filestart=n/2+1; fileend=names_n;
% %        end
%        filestart=1; fileend=names_n;
%        for i1=filestart:fileend 
%         %filename = char(fullfile(path,'\1\', files(i1))); 
%         filename = char(fullfile(path, files(i1))); 
%         %filename = char(fullfile(path,'\2\', files(i1)));  
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
%       stride(1:2)=1;
%       if M1>512
%          stride(1)=floor(M1/512);
%       end
%       if N1>512
%           stride(2)=floor(N1/512);
%       end
%       if (M1>512) & (N1>512)
%        a=cover_image(1:stride(1):stride(1)*512,1:stride(2):stride(2)*512);
%       elseif M1>512
%        a=cover_image(1:stride(1):stride(1)*512,:);
%       elseif N1>512
%        a=cover_image(:,1:stride(2):stride(2)*512);
%       end
%       cover_image=a;
%       [M N L]=size(cover_image);
%       stego_image=cover_image;
% %       n1=2; n2=8;
% steg1=reshape(stego_image,n1,floor(M/n1),[]);
%n1=8; n2=8;
stego_image=cover_image;
M=size(stego_image,1); N=size(stego_image,2);
steg1=reshape(stego_image,n1,floor(M/n1),[]);
steg2=permute(steg1, [1 3 2]);
steg3=reshape(steg2,n1,n2,[]);
steg4=reshape(steg3,n1*n2,[]);
% steg31=reshape(steg4,n1,n2,[]);
% steg21=reshape(steg31,n1, N, []);
% steg11=permute(steg21, [1 3 2]);
%steg11=reshape(steg21,n1, floor(M/n1),[]);
%stego_image1=reshape(steg11, M,[]);
steg5=floor(single(steg4)/mdl);
%steg5=floor(steg4/mdl);
mx=max(steg5); mn=min(steg5);
dif=mx-mn;
[dif1, ind1]=sort(dif,'descend');
mask=mx~=mn;
%stdev1=std(single(steg4));
%hg=histogram(stdev1);
%mask=stdev1>=mdl/2;
% blockno=floor(M*N/n1/n2);
% mapbno=ceil(blockno/n1/n2);
% if mapbno==blockno/n1/n2 mapbno=mapbno+1; end
% mask=stdev1(mapbno+1:end)>=mdl/2;
%[stego_im r not_used2] = mlsb_embed_30032021(steg4(:,2:mapbno), mask');
%steg4(:,2:mapbno)=stego_im;
% steg41=steg4(:,mapbno+1:end);
szsteg=sum(mask);
sec=reshape(secret_message, n1*n2,[]);
szsec=size(sec,2);
r=szsec;
if szsec==0
    MSE=0;
     return;
end
% mdl1=mdl;
% if szsteg>szsec
%  while 1
%     mdl1=mdl1+1;
%     mask1=stdev1(mapbno+1:end)>=mdl1/2;
%     szsteg1=sum(mask1);
%     if szsteg1>szsec
%         mask=mask1;
%         szsteg=sum(mask);
%     else
%         break;
%     end
%  end
% else
% while (szsteg<szsec)& (mdl1>0)
%       mdl1=mdl1-1;
%       mask=stdev1(mapbno+1:end)>=mdl1/2;
%       szsteg=sum(mask);
% end
% if mdl1<1
%     disp('Not enough space');
%     MSE=-1; r=-1;
%     return;
% end
% end
%steg5=single(steg4(:,mask));
%[stego_im rr not_used2] = mlsb_embed_30032021(steg4(:,2:mapbno), mask');
% steg4(:,2:mapbno)=stego_im;
% if szsteg<szsec
%     disp('Not enough space for embedding');
%     MSE=-1; r=-1;
%     return;
% end
%steg51=single(steg4(:,mask));
steg51=single(steg4(:,ind1));
%sz1=szsec;
%Threshold=5;
sz_thr=sum(dif1>=Threshold);
sz1=min([szsec, size(steg51,2),sz_thr]);
r=sz1;
steg6=steg51(:,1:sz1) - mod(steg51(:,1:sz1),mdl)+sec(:,1:sz1);
maskgt=steg6>255;
steg6(maskgt)=steg6(maskgt)-mdl;
% mask1=(steg6-steg51(:,1:sz1)>mdl/2) & (steg6>=mdl);
% steg6(mask1)=steg6(mask1)-mdl;
% mask2=(steg6-steg51(:,1:szsec)<-mdl/2) & (steg6<256-mdl);
% steg6(mask2)=steg6(mask2)+mdl;
%ind1=find(mask);
%steg4(:,ind1(1:sz1))=steg6;
steg4(:,ind1(1:sz1))=steg6;
%steg4(:,mapbno+1:end)=steg41;
steg31=reshape(steg4,n1,n2,[]);
steg21=reshape(steg31,n1, N, []);
steg11=permute(steg21, [1 3 2]);
%steg11=reshape(steg21,n1, floor(M/n1),[]);
 stego_image1=reshape(steg11, M,[]);
 stego_image=stego_image1;
 MSE=sum(sum((single(stego_image)-single(cover_image)).^2))/M/N;
 M=size(stego_image,1); N=size(stego_image,2);
steg11=reshape(stego_image,n1,floor(M/n1),[]);
steg21=permute(steg11, [1 3 2]);
steg31=reshape(steg21,n1,n2,[]);
steg42=reshape(steg31,n1*n2,[]);
steg52=floor(single(steg42)/mdl);
mx1=max(steg52); mn1=min(steg52);
dif2=mx-mn;
[dif21, ind11]=sort(dif,'descend');
mask1=mx1~=mn1;
% blockno1=floor(M*N/n1/n2);
% mapbno1=ceil(blockno1/n1/n2);
% if mapbno1==blockno1/n1/n2 mapbno1=mapbno1+1; end
% stego_im1=steg42(:,2:mapbno);
% mask11=mod(stego_im1,2);
% if sum(mask11(1:size(stego_im1,1)*size(stego_im1,2)-mapbno1)~=mask)
%     disp('Incorrect mask extracted')';
% end
%if sum(mod(stego42(1:size(mask,2)),2)~=mask)
% steg43=steg42(:,logical(mask11(1:size(stego_im1,1)*size(stego_im1,2)-mapbno1)));
% sec1=mod(steg43, mdl);
%ind11=find(mask1);
%sec2=reshape(mod(steg42(:,mapbno1+ind11(1:r)),mdl),1,[]);
sec2=reshape(mod(steg42(:,ind11(1:r)),mdl),1,[]);
%sm1=sum(sec2~=secret_message(1:r));
%sm1=sum(sec2~=secret_message(1:r*size(steg42,1))');
sm1=sum(sec2~=secret_message(1:r*size(steg42,1)));
if sm1~=0
    disp('Incorrect extraction');
    return;
end
end
     
     