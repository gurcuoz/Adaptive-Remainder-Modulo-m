clc;
clear; 
%uf=figure;
% Rtab=uitable(uf);
% Rtab.Data={0, 15, 3, 0; 16, 31, 4, 0; 32, 255, 5, 0};
% Rtab.ColumnName={'Lower', 'Upper', 'Bit no', 'used'};
%Thresholdedit = uicontrol(uf,'Style','edit',...
%         'Position',[0 0 60 15],'Value', 192,...
%         'String','Treshold', 'CallBack',@getThreshold);
global Threshold method;
    
%R=[[0 15 3 0]; [16 31 4 0]; [32 255 5 0]];%  Ri2-Ri1>=2^Ri3 Modified
%R=[[0 15 1 0]; [16 255 2 0]];%  Ri2-Ri1>=2^Ri3 Modified
%R=[[0 255 1 0]];%  Ri2-Ri1>=2^Ri3 Modified
%R=[[16 255 1 0]];% R34 Ri2-Ri1>=2^Ri3 Modified
%R=[[8 255 1 0]];%  Ri2-Ri1>=2^Ri3 Modified
%R=[[4 255 1 0]];%  Ri2-Ri1>=2^Ri3 Modified
%R=[[16 255 1 0];[8 15 1 0];[4 7 1 0]];%  R37 Ri2-Ri1>=2^Ri3 Modified
R=[[16 255 1 0];[8 15 1 0];[6 7 1 0]];%  R38 Ri2-Ri1>=2^Ri3 Modified
%R=[[16 255 1 0];[12 15 1 0];[10 11 1 0]];%  R39 Ri2-Ri1>=2^Ri3 Modified
%R=[[64 255 6 0];[32 63 5 0];[16 31 4 0]];%  R40 Ri2-Ri1>=2^Ri3 Modified
%R=[[16 255 1 0];[8 15 1 0];[5 7 1 0]];%  R41 Ri2-Ri1>=2^Ri3 Modified
%R=[[64 255 3 0];[16 63 2 0];[6 15 1 0]];%  R42 Ri2-Ri1>=2^Ri3 Modified
%R=[[64 255 2 0];[16 63 2 0];[12 15 2 0];[10 11 1 0]];%  R43 Ri2-Ri1>=2^Ri3 Modified
%R=[[0 63 1 0]; [64 255 2 0] ];%  Ri2-Ri1>=2^Ri3 Modified
%R=[[0 15 2 0]; [16 255 3 0]];
%T=192; K=2;
%T=192; K=1;
T=0; K=0;
%T=0; K=2;
%T=192; K=3;
%T=160; K=3;
%T=256; K=3;%K=1;
%T=Threshold; K=1;
Freq = [0 0 0 0 0];
%mask=[1 0;0 1];
mask=[0 1;1 0];
%mask = [0 0; 1 1];
%mask = [0 0; 0 1];
% cover_image=imread('C:\Users\nor\Desktop\thesis\GRAY SCALE\LSBCT\zeldaBMP.bmp');
%start_length = 2*524288;
%start_length = 5*2^18;
%for t=1:1 % 6
result = [];
resi=[];
%rng(123);
rng('default');
%        pict=1;
folder1=2; folder2=2;
training=input('Training? Yes - 1; No - 0: ');
folder1=input('Folder?/2..10 = ');
% Create pop-up menu
 % Create a figure and axes
    fg = figure(1);%('Visible','on');
    ax = axes('Units','pixels');
    method=0;
popup = uicontrol('Style', 'popup',...
           'String', {'Method','Khodaei','PM(M)','ATD','STC','MLSB','OPAP', 'LSB','OOPAP','Adaptive', 'LSB Adaptive', 'Nguyen', 'LSBM Revisited', ... 
           'LSBOPAP Adaptive', 'LSBRDH', 'Weng', 'Weng IDH'},...
           'Position',  [20 340 100 50],...
           'Callback', @setmethod);
       
%method=input('Method?/Khodaei - 1; PM(M) - 2: ATD - 3: STC - 4; \n MLSB - 5; OPAP - 6; LSB - 7; OptimalOPAP - 8; Adaptive - 9; LSB Adaptive - 10: Nguyen: 11: LSBM revisited 12 ');
if method==2
    mdl=input('Modulo value? ');
    R=[[0 255 mdl]];
elseif method==1 %Khodaei
    mdl=2;
elseif method==3 %method=3 ATD
    mdl=3;
elseif method==4
    %alpha=input('Alpha value? ');
    H_hat = [253 199 251 167];%[141 179 203 255]; %[253 199 251 167];
    %H_hat = 512+[253 199 251 167];%[141 179 203 255]; %[253 199 251 167];
    l=max(ceil(log2(H_hat)));
    w=size(H_hat,2);
    alfa0=1/w;
    mdl=2;
elseif method==5
    mdl=2;
    %alpha=input('Alpha value? ');
elseif method==6 | method==7 | method==8
    mdl=input('Modulo value? ');
elseif method==9 | method==10 | method==11 | method==13
    mdl=input('Modulo value? ');
    n1=input('Block size vertical (rows)? ');
    n2=input('Block size horizontal (cols)? ');
    Threshold1=input('Threshold='); % 0 instead of 5 used earlier
elseif method==12
    mdl=2;
    n1=input('Block size vertical (rows)? ');
    n2=input('Block size horizontal (cols)? ');
elseif method==14 %LSBRDH
    k=input('Bit plane number k=? ');
    arch=input('Archiver?/Zip:Z;Gzip:G;Tar:T','s');
    mdl=2;
elseif method==15 | method == 16 %Weng
    n1=input('n1=? ');
    n2=input('n2=? ');
    %arch=input('Archiver?/Zip:Z;Gzip:G;Tar:T','s');
    base=input('base=?');
    Th=input('Th=?');
    mdl=2;
    if method==16
        mdl=input('mdl=?');
    end
    result=[];
    result1=[];
    fail=[];
else
    disp('Incorrect method');
    return;
end
%chunk= mdl^floor(13/log2(mdl));
% chunk= 2^14; %3*2^13;%floor(2^14/log2(mdl));%floor(2^13/log2(mdl));
%chunk=floor(2^14/log2(mdl));
chunk=floor(2^16*0.05/log2(mdl)/n1/n2)*n1*n2;% it is for 5% step payload for 256x256 images
sec=input('Secret random? [R]/from image: [I]','s');
folder2=folder1;
 %folder2=10;
     for i=folder1:folder2 %4:4 %3:3%1:2%9
       %start_length=(i-1)*2^19;%19;%13;%4;%5;%6; %7%; 8;%17;%18;%17;%15;%17; %6;
        start_length=(i-2)*chunk;%12;%19;%13;%4;%5;%6; %7%; 8;%17;%18;%17;%15;%17; %6;
       %secret_message= randi([0 2],1,start_length);
       if sec~='R'          
[file,path] = uigetfile('*.*');
if isequal(file,0)
   disp('User selected Cancel');
else
   disp(['User selected ', fullfile(path,file)]);
end
filename1 = fullfile(path,file); 
secrm=imread(filename1);
secrmm=single(secrm(1:start_length));%copy of secrm
%[Ms Ns Ls]=size(secrm);
end
if sec=='R'
    
            %secret_message= randi([0 md-1],512);%8start_length);
            secret_message= randi([0 mdl-1],1,start_length);
else
          k=log2(mdl);
          parts=ceil(8/k);
          %secret=cover_image;
          for ii=1:parts
            secr(:,ii)=mod(secrmm,mdl);
            secrmm=floor(single(secrmm)/mdl);
          end
          secret_message=reshape(secr',1,[]);
end

        %secret_message= randi([0 mdl-1],1,start_length);      
       %secret_message= randi([0 1],1,start_length);
       %path='D:\Ucheba\Masters\Hajer\Hajer Thesis\GRAY SCALE\LSBCT\pictures\pictures';
       %[FileName,path] = uigetfile('*.*','Select the images folder');
%        if training==1
%            names=fullfile(path,{'*.bmp','*.jpg','*.tiff'});%for training
%        else
%            names=fullfile(path,{'*.bmp','*.jpg','*.tiff', '*.tif','*.png'});%for testing
%        end
         % path1='C:\Temp';
          %path1='C:\Temp\ucid.v2';
          path1='C:\Temp\Boss1.01';
          %path='G:\Pictures';
          %path='C:\Temp\2';
          %path='C:\Temp\ucid.v2';
          %path='C:\Temp\ucid.v2\2';
          path='C:\Temp\Boss1.01\BOSSbase_1.01';
%         names=fullfile(path,'\1\','*.tif');%for training 
          %names=fullfile(path,'*.tif');%for training 
          names=fullfile(path,'*.pgm');%for training 
%        names_n=numel(names);
        filenames=dir(char(names));
        names_n=numel(filenames);
        files={};
        for in=1:names_n
%         filenames=dir(char(names(in)));
         files=cat(2,files,{filenames(in).name});
        end

       %diri=fullfile(path,char(i+48)); 
       diri=fullfile(path1,int2str(i)); 
       [status, msg, msgID]= mkdir (diri);
%        if training==1
%            filestart=1; fileend=names_n/2;
%        else
%            filestart=n/2+1; fileend=names_n;
%        end
failed=[];
       filestart=1; %269; %1; 
       fileend=names_n;
       res=[];
       for i1=filestart:fileend 
        %filename = char(fullfile(path,'\1\', files(i1))); 
        filename = char(fullfile(path, files(i1))); 
        %filename = char(fullfile(path,'\2\', files(i1)));  
       %filename = cat(2, 'D:\Ucheba\Masters\Hajer\Hajer Thesis\GRAY SCALE\LSBCT\pictures\pictures\',  char(pictures(pict)));  
       cover_image=imread(filename);
       %if start_length>0
       %cover_image = randi([0 255],512,512);
       %cover_image = cover_image(1:10,1:10);
       %secret_message = secret_message(1:200);
       %%%%[M N L]=size(cover_image);
       %%%%b=cover_image;
       %%%%Cover_image=imresize(b,[256,256]); %resze to 256 for SRNet
       Cover_image=imresize(cover_image,[256,256]); %resze to 256 for SRNet
        cover_image=Cover_image;
       %for ll=1:L
        %if L>=1
         %a=cover_image(:,:,1);
        % a=cover_image(:,:,ll);
        % a=Cover_image(:,:,ll);
        %%%% cover_image=a;
      %%%   cover_image=Cover_image;
       %end
     %%%  if mod(N,2)==0
     %%%     N1=N;
    %  else
    %     N1=N-1;
   %   end
   %   if mod(M,2)==0
    %      M1=M;
   %   else
   %       M1=M-1;
    %  end
   %   stride(1:2)=1;
   %   if M1>512
    %     stride(1)=floor(M1/512);
   %   end
    %  if N1>512
    %      stride(2)=floor(N1/512);
    % cover_image=Cover_image;  end
    %%%  if (M1>512) & (N1>512)
    %%%   a=cover_image(1:stride(1):stride(1)*512,1:stride(2):stride(2)*512);
    %%%  elseif M1>512
    %%%   a=cover_image(1:stride(1):stride(1)*512,:);
   %%%   elseif N1>512
    %%%   a=cover_image(:,1:stride(2):stride(2)*512);
    % end
    %  cover_image=a;
   %   [M N L]=size(cover_image);
      stego_image=cover_image;
      if start_length>0
       %secret_message = mod(cover_image,2);
       %secret_message = reshape([secret_message secret_message],1,[]);

    % [e r]=size(secret_message);
       %[stego_image r not_used] = Embedding23112017( R,cover_image,secret_message);
       %[stego_image r not_used Freqcase1 R] = EmbeddingKhodaei13052018( R,T, K, cover_image,secret_message);
       if method==1% Khodaeii
       %[stego_image r not_used Freqcase1 R] = EmbeddingKhodaei08042018( R,T, K, cover_image,secret_message);
       %[stego_image r not_used Freqcase1 R] = EmbeddingKhodaei13052018( R,T, K, cover_image,secret_message);
       Khostart=tic;
       [stego_image r not_used Freqcase1 R] = EmbeddingKhodaei23032022( R,T, K, cover_image,secret_message);
       %[stego_image r not_used Freqcase1 R] = EmbeddingKhodaei18042018( R,T, K, cover_image,secret_message);
       Khoend=toc(Khostart);
       res=[res r-1];
       elseif method==2 %method PM(M)
           [stego_image r1 not_used] = Embedding18032020Multipleanyrange( R,cover_image,secret_message);
           r=r1;
       elseif method==3 %ATD
           [stego_image r1 not_used]=EmbbeddingATD_18032020(cover_image,secret_message);%embedding function
           r=r1;
       elseif method==4 %STC
           M=floor(M/l)*l;N=floor(N/w)*w;
           cover_image=cover_image(1:M,1:N);
           alfa=start_length/M/N;
           mm=floor(M*N*alfa0);%3;%33000;
           start1=tic;
           code = create_code_from_submatrix(H_hat, mm);% it is in D:\Ucheba\CMSE492 Spring2021\STC matlab
           if alfa~=alfa0
             delta=alfa-alfa0;
             shf=code.shift;
             shf1=reshape(shf, w,[]);
             shf2=reshape(shf1(1:w-1,:),1,[]);
             sz_shf=size(shf2,2);
             m_sz_delta=abs(start_length-sum(code.shift));%floor(code.n*abs(delta));
             if delta>0
              stride=floor(sz_shf/m_sz_delta);
              shf2(1:stride:(m_sz_delta-1)*stride+1)=1;
              %shf2(end)=1;
              shf3=reshape(shf2,w-1,[]);
              shf1(1:w-1,:)=shf3;
             else
              shf1(w,1:m_sz_delta)=0;
             end
           code.shift=reshape(shf1,1,[]);
           shf1=[]; shf2=[]; shf3=[];shf=[];
        end
         w1 = ones(code.n,1);%ones(code.n,1);
%          rho = S_UNIWARD_04042022(cover_image);
%          w1=reshape(rho,[],1);
         [stego_image embedded1 not_used1] = stc_embed_29032021(cover_image,secret_message, code, w1);
         end1=toc(start1);
         r=embedded1;
         res=[res r];
        %end
       elseif method==5
         [stego_image r not_used2] = mlsb_embed_30032021(cover_image, secret_message');
       elseif method==6 
           msgs=min(M*N,size(secret_message,2));
           %strd=floor((M*N-1)/(msgs-1));
           strd=1;
           %[stego r MSE] = OPAP14042021(cover_image(1:msgs), secret_message, mdl );
           [stego r MSE] = OPAP21052021(cover_image(1:strd:1+(msgs-1)*strd), secret_message(1:msgs), mdl );
           %[stego r MSE] = OPAP14042021(cover_image(1:strd:1+(msgs-1)*strd), secret_message(1:msgs), mdl );
           stego_image(1:strd:1+(msgs-1)*strd)=stego;
       elseif method==7
           msgs=min(M*N,size(secret_message,2));
           %strd=floor((M*N-1)/(msgs-1));
           strd=1;
           [ stego r MSE] = LSB14042021(cover_image(1:strd:1+(msgs-1)*strd), secret_message(1:msgs), mdl );
           stego_image(1:strd:1+(msgs-1)*strd)=stego;
       elseif method==8
           %[ stego_image r MSE] = ansari14042021(cover_image, secret_message, mdl );
           [ stego r MSE] = ansari27042021(cover_image, secret_message, mdl );
           MSE1=MSE*size(stego,2)/M/N;
           result=[result; MSE1, 10*log10(255*255/MSE1)];
           stego_image=cover_image;
           stego_image(1:size(stego,2))=stego;
       elseif method==9 %adaptive MLSB with mask
           %[ stego_image r MSE] = ansari14042021(cover_image, secret_message, mdl );
           %[ stego_image r MSE] = adaptive01052021_1(cover_image, secret_message, mdl,n1,n2 );
           %[ stego_image r MSE] = adaptive02052021(cover_image, secret_message, mdl,n1,n2 );
           [ stego_image r MSE] = adaptive06052021(cover_image, secret_message, mdl,n1,n2 );
           %MSE1=MSE*size(stego,2)/M/N;
           if MSE<0
               failed=[failed, i1];
               continue;
           else
               result=[result; MSE, 10*log10(255*255/MSE)];
           end
           elseif method==10 %adaptive LSB without mask
           %[ stego_image r MSE] = ansari14042021(cover_image, secret_message, mdl );
           %[ stego_image r MSE] = adaptive01052021_1(cover_image, secret_message, mdl,n1,n2 );
           %[ stego_image r MSE] = adaptive02052021(cover_image, secret_message, mdl,n1,n2 );
           %[ stego_image r MSE] = adaptiveLSB05062021(cover_image, secret_message, mdl,n1,n2 );
           %tic;
           %for c=1:1000
                [stego_image r, MSE] = adaptiveLSB29092023(cover_image, secret_message, mdl, n1, n2, Threshold1);
           %end
           %embedextractTime=toc;
%            stego_image=cover_image;
%            stego_image(1:size(stego,2))=stego;
           if MSE<0
               failed=[failed, i1];
               continue;
               else
               result=[result; MSE, 10*log10(255*255/MSE)];
           end
           elseif method==11 %adaptive LSB without mask
           %[ stego_image r MSE] = ansari14042021(cover_image, secret_message, mdl );
           %[ stego_image r MSE] = adaptive01052021_1(cover_image, secret_message, mdl,n1,n2 );
           %[ stego_image r MSE] = adaptive02052021(cover_image, secret_message, mdl,n1,n2 );
           CGC=1;%0;
           [ stego_image r MSE] = adaptiveNguyen09062021(cover_image, secret_message, mdl,n1,n2, CGC );
%            stego_image=cover_image;
%            stego_image(1:size(stego,2))=stego;
           if MSE<0
               failed=[failed, i1];
               continue;
           else
               result=[result; MSE, 10*log10(255*255/MSE), r];
           end
       elseif method==12
         [stego_image r MSE] = mlsbrevpair_embed_16072021(cover_image, secret_message', n1, n2); %it is in D:\Ucheba\CMSE492 Spring2021\STC matlab
         result=[result; MSE, 10*log10(255*255/MSE), r];
       elseif method==13 %adaptive LSB without mask
           %[stego_image r MSE] = adaptiveLSBOPAP30092021(cover_image, secret_message, mdl,n1,n2 );
           [stego_image r MSE] = adaptiveLSBOPAP25082021(cover_image, secret_message, mdl,n1,n2 );
           %[stego_image r MSE] = adaptiveLSBOPAP24082021(cover_image, secret_message, mdl,n1,n2 );
            %[stego_image r MSE] = adaptiveLSBOPAP19082021(cover_image, secret_message, mdl,n1,n2 );
            %[stego_image r MSE] = adaptiveLSBOPAP14082021(cover_image, secret_message, mdl,n1,n2 );
            %[stego_image r MSE] = adaptiveLSBOPAP12082021(cover_image, secret_message, mdl,n1,n2 );
            if MSE<0
               failed=[failed, i1];
               continue;
           else
               result=[result; MSE, 10*log10(255*255/MSE), r];
            end
       elseif method==14 %LSBRDH
         [stego_image szcompr MSE PSNR]= LSBRDH06112021(cover_image, secret_message, arch, k);
         r=0;
       elseif method==15 %Weng
         [stego_image result]=Weng_Embedding_11042022(cover_image,  n1, n2, Th, secret_message, files(i1),base);
         result1=[result1; result];
         if result{11}<0
             fail=[fail i1];
         end
         r=0;
        elseif method==16 %Weng IDH
         %[stego_image result]=Weng_Embedding_IDH_28042022(cover_image,  n1, n2, Th, secret_message, files(i1),base);
         [stego_image result]=Weng_Embedding_IDH_30042022(cover_image,  n1, n2, Th, secret_message, files(i1),base,mdl);
         result1=[result1; result];
         if result{11}<0
             fail=[fail i1];
         end
         r=0;
       else
           disp('Incorrect method #');
           return;
           %end
       end
        %output = extraction28122017Multipleanyrange(R, stego_image,r1-1);
%      Freq(1) = Freq(1) + Freqcase1;
%        for i1=1:size(R,1)
%          Freq(i1+1) = Freq(i1+1) + R(i1,4);
%        end
       if r>1
           if method==1% Khodaei
             output = extractionKhodaei08042018(R, T, K, stego_image,r-1, secret_message);
             ii = output~=char(secret_message(1:size(output,2))+48);
           elseif method==2% method PM(M)
             output = extraction28122017Multipleanyrange(R, stego_image,r1);
             ii=(transpose(output)~=secret_message(1,:));
           elseif method==3 %ATD
             output = extractionATD_18032020( stego_image,r1 );%extraction function
              m1=min(size(output,2),size(secret_message,2));
              ii=(output(1:m1)~=secret_message(1:m1));
           end
%          if sum(sum(output(1:m1)~=secret_message(1:m1)))
%           disp('Incorrect TDA extraction');
%          else
%           disp('Correct TDA extraction');
%          end   
        %end
         %  end
       %ii = output~=char(secret_message(1:size(output,2))+48);
       if method<4
       if sum(ii)>0
           break;
       end
       end
       r = r-1;
       end
       end
       outname=char(fullfile(diri, files(i1)));%last was i1=1939
       %%%outname=char(fullfile(diri,strcat(char('0')+ll, char(files(i1)))));
        imwrite(uint8(stego_image),outname);
       %end %ll=1:L
       end%i1last=268 start from 269
     end%i
     %end
     
     function setmethod(source,event)
        global method;
        method = source.Value-1;
        %maps = source.String;
        % For R2014a and earlier: 
        % val = get(source,'Value');
        % maps = get(source,'String'); 

%         newmap = maps{val};
%         colormap(newmap);
    end
 
     function getThreshold(source,~)
        global Threshold;
        val = source.Value;
        Threshold=val;
        %maps = source.String;
        % For R2014a and earlier: 
        % val = get(source,'Value');
        % maps = get(source,'String'); 

        %newmap = maps{val};
        %colormap(newmap);
    end