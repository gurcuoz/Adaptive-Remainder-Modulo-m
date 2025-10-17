function [ stego_image embedded MSE] = ansari_14042021(cover_image, secret_message, md )
%clear;
% n=8;%size of the oost matrix
% m=[[10, 4, 2, 4, 10, 20, 18, 20]; ...
%    [0, 1, 4, 9, 16, 9, 4, 1]; ...
%    [0, 1, 4, 9, 16, 9, 4, 1]; ...
%    [16, 11, 12, 19, 32, 19, 12, 11]; ...
%    [16, 9, 4, 1, 0, 1, 4, 9]; ...
%    [0, 0, 0, 0, 0, 0, 0, 0]; ...
%    [16, 9, 4, 1, 0, 1, 4, 9]; ...
%    [2, 5, 14, 29, 34, 29, 14, 5]];
% m1=[[10, 4, 2, 4, 10, 20, 34, 52]; ...
%    [0, 1, 4, 9, 16, 25, 36, 49]; ...
%    [0, 1, 4, 9, 16, 25, 36, 49]; ...
%    [16, 11, 12, 19, 32, 51, 76, 107]; ...
%    [16, 9, 4, 1, 0, 1, 4, 9]; ...
%    [0, 0, 0, 0, 0, 0, 0, 0]; ...
%    [16, 9, 4, 1, 0, 1, 4, 9]; ...
%    [50, 37, 30, 29, 34, 45, 62, 85]];
% md=8;
% [assgn k]=ansari01102020(md,m);
% cost=0;
% for i=1:md
%     cost=cost+m(assgn(i,1),assgn(i,2));
% end
% pictures = [
%     %"46794713-toys-.jpg"; ...
% %     "airplane.tif"; ...
%      "baboon.bmp"; ...
%  %    "BABOON24012018.bmp"; ...3D image Khodaei works correctly on couple
%  %   "Baboon24012018.gif"; ...
%  %    "baby.png"; ...
%     "barbara.bmp"; ...
% %     "lena.bmp"; ... Khodaei works correctly on Lena.bmp
%  %    "boat.gif"; ...
%   %   "boy.tif"; ...
%  %    "cameraman.tif"; ...
%  %    "couple.tif"; ... Khodaei works co
%      "goldhill.bmp"; ...rrectly on couple
%      "elaine.bmp"; ... Khodaei works correctly on elain.bmp
%      "girlface.dib.bmp"; ...
% %     "home.gif"; ...
% %     "Jerusalem.gif"; ... Khodaei works correctly on jerusalem.gif
% %   "jet.bmp"; ...
% %     "lady.tif"; ...  Khodaei works correctly on Lady.tif
% %     "lake.gif"; ...
%     "lena.bmp"; ... Khodaei works correctly on Lena.bmp
%      "peppers.bmp"; ...
%     "tank.tif"; ... Khodaei works correctly on tank.tif
% %     "tiffany.jpg"; ...3D image Khodaei works correctly on tank.tif
%      "truck.tif"; ...
% %     "46794713-toys-.jpg"; ...
%      "zelda.bmp"; ...%     ];    
% %              "zeldaBMP.bmp"; ...
% %             "peppersBMP.bmp"; ...
% %             "lenaBMP.bmp"; ...
% %             "lady.tif"; ...
% %            "house.tif"; ...
% %             "elainBMP.bmp"; ...
% %            "camerman.tif"; ...
% %             "BOY.tif"; ...
% %             "boat.gif"; ...
% %             "barbaraBMP.bmp"; ...
% %             "BABY.png"; ...
% %             "BaboonBMP.bmp"; ...
% %             "airplaneBMP.bmp"
%            ];
%start_length = 2*524288;
%start_length = 5*2^18;
%for t=1:1 % 6
% for pict=1:size(pictures,1)
% %         rng(123);
% %         for i=1:9
%         start_length=2^9;%17;%15;%17; %6;
% %        %secret_message= randi([0 2],1,start_length); 
%         
%        %secret_message= randi([0 1],1,start_length); 
%        %filename = cat(2, 'D:\Ucheba\Masters\Hajer\Hajer Thesis\GRAY SCALE\LSBCT\',  char(pictures(pict)));
% %        [file,path] = uigetfile('*.*');
% % if isequal(file,0)
% %    disp('User selected Cancel');
% % else
% %    disp(['User selected ', fullfile(path,file)]);
% % end
% % filename = fullfile(path,file);
%        filename = cat(2, 'D:\Ucheba\Masters\Hajer\Hajer Thesis\GRAY SCALE\LSBCT\pictures\pictures\',  char(pictures(pict)));  
%        cover_image=imread(filename);
%        %cover_image = randi([0 255],512,512);
%        %cover_image = cover_image(1:10,1:10);
%        %secret_message = secret_message(1:200);
%        [M N L]=size(cover_image);
% end
%result = [];
  %S=sum(cover_image(:,:));
  %avg=sum(S)/(M*N);
%disp('cover_image       Size_secret_data          PSNR             Actual_PSNR             MSNR             MSE             Ml        Mu         T ')
%disp('                                             dB                  dB                   dB')
%p='zeldaBMP.bmp';
%disp('=================================================================================================================================================')
   
  
  %r=0;MSE=0;PSNR=0;Bpp=0;
  MSE=-1;
  m=secret_message;
  embedded=size(m,1)*size(m,2);
%not_used=M*N-embedded;
if embedded==0
    stego_image=cover_image;
    return;
end
  M=size(secret_message,1); N=size(secret_message,2);
stego_image=single(cover_image(1:M, 1:N));
%M=size(cover_image,1); N=size(cover_image,2);


  resi=[];
  %rng(123);%rng('default');
  %avhdif(1:256)=0;
  %md=input('md='); 8;k=floor(log2(md)); rng(123);
  store=[];
  %costs=[];
  costs1=[];
%   lb=zeros(md);
%   ub=ones(md);
%   %f=ones(md);
%   beq=ones(md,2);
%   A=[]; b=[];
%   Aeq=zeros(md^2,2*md);
%   for j=1:md
%    Aeq((j-1)*md+1:j*md,j)=1;
%   end
%   for j=md+1:2*md
%    Aeq(j-md:md:md^2,j)=1;
%   end
%   x=linprog(m1,A,b,Aeq',beq,lb,ub);
%   sol=reshape(x,md,[]);
% ewsum(1:510,1:511)=0;
% wesum(1:510,1:511)=0;
% snsum(1:509,1:512)=0;
% nssum(1:509,1:512)=0;
% neswsum(1:509,1:511)=0;
% swnesum(1:509,1:511)=0;
% senwsum(1:509,1:511)=0;
% nwsesum(1:509,1:511)=0;
% sec=input('Secret random? [R]/partitioned image: hor? [H]/ ver? [V]','s');
% method=input('Method? OPAP [P]/Optimal [O]/LSB [L]/','s');
% for pict1=1:size(pictures,1)%try all pictures as secret for all pictures
% if sec~='R'          
% [file,path] = uigetfile('*.*');
% if isequal(file,0)
%    disp('User selected Cancel');
% else
%    disp(['User selected ', fullfile(path,file)]);
% end
% filename1 = fullfile(path,file); 
% secrm=imread(filename1);
% secrmm=secrm;%copy of secrm
%[Ms Ns Ls]=size(secrm);
% end
% if sec=='R'
%     
%             secret_message= randi([0 md-1],512);%8start_length);
%       else
%           parts=ceil(8/k);
%           %secret=cover_image;
%           for i=1:parts
%             secr(1:Ms,1:Ns,i)=mod(secrm,md);
%             secrm=floor(single(secrm)/md);
%           end
%           if sec=='H'
%             Ms1=floor(Ms/parts);
%             for i=1:parts
%               secret_message((i-1)*Ms1+1:i*Ms1,1:Ns)=secr(1:Ms1,1:Ns,i);
%             end
%             Ms=parts*Ms1;
%           else
%             Ns1=floor(Ns/parts);
%             for i=1:parts
%               secret_message(1:Ms,(i-1)*Ns1+1:i*Ns1)=secr(1:Ms,1:Ns1,i);
%             end
%             Ns=parts*Ns1;
%           end
%      end
% for pict=1:size(pictures,1)
    %for ii=1:10
%         rng(123);
%         for i=1:9
       % start_length=2^9;%17;%15;%17; %6;
%        %secret_message= randi([0 2],1,start_length); 
        
       %secret_message= randi([0 1],1,start_length); 
       %filename = cat(2, 'D:\Ucheba\Masters\Hajer\Hajer Thesis\GRAY SCALE\LSBCT\',  char(pictures(pict)));
%        [file,path] = uigetfile('*.*');
% if isequal(file,0)
%    disp('User selected Cancel');
% else
%    disp(['User selected ', fullfile(path,file)]);
% end
% filename = fullfile(path,file);
%        filename = cat(2, 'D:\Ucheba\Masters\Hajer\Hajer Thesis\GRAY SCALE\LSBCT\pictures\pictures\',  char(pictures(pict)));  
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
%       if sec~='R'
%           M=min(M,Ms); N=min(N,Ns);
%       end
%if method=='O' 
sec=single(secret_message(1:M, 1:N));
stego_image=stego_image - mod(stego_image,md)+sec;
mask=(stego_image-single(cover_image(1:M, 1:N)))>md/2 & (stego_image>=md);
stego_image(mask)=stego_image(mask)-md;
mask=(stego_image-single(cover_image(1:M, 1:N)))<=-md/2 & (stego_image<=255-md);
stego_image(mask)=stego_image(mask)+md;
MSE=sum(sum((stego_image-single(cover_image(1:M, 1:N))).^2))/M/N;
if sum(sum(mod(stego_image, md)~=sec))~=0
  disp('Incorrect embedding');
  return;
end
return;
stego_image(mask)=stego_image(mask)-md;
     a=[];
       for i=0:md-1
          mask=secret_message(1:M,1:N)==i;
        for j=0:md-1
          stego=cover_image(1:M,1:N);
          bb=cover_image(1:M,1:N);
          %sumi=sum(sum(mask));
          a(1:M,1:N)=0;
          a(mask)=mod(stego(mask),md);
          stego(mask)=double(stego(mask))-a(mask)+j;
          masklt=(single(stego)-single(bb)<-md/2) & (single(stego)<256-md);
          stego(masklt)=stego(masklt)+md;
          maskgt=single(stego)-single(bb)>md/2 & stego>=md;
          stego(maskgt)=stego(maskgt)-md;
          m2(i+1,j+1)=sum(sum((double(stego(mask))-double(bb(mask))).^2));
        end
      end
  %store=[store, m2];
      

%assgn=ansari28092020(n,m);
%[assgn k]=ansari28092020(md,m2);
%[assgn k]=ansari30092020(md,m2);
%[assgn k]=ansari01102020(md,m2);%works for md=2,4,8,16, does not work for md=32
%[assgn k]=ansari03102020(md,m2);%works for md=2,4,8,16, does not work for
%md=32; md=16 calculates 3 yimes faster than 01102020
%   x=linprog(m2,A,b,Aeq',beq,lb,ub);
%   sol=reshape(x,md,[]);
%   [row,col]=find(sol');
%assgn=ansari04102020(md,m2);
%disp(assgn);
% cost=0;
% for i=1:md
%     %cost=cost+m2(assgn(i,1),assgn(i,2));
%     cost=cost+m2(assgn(i,1),assgn(i,2));
% end
%cost=sum(m2(logical(sol)));
% costp=sum(diag(m2));
% disp(['cost=' int2str(cost) 'costp=' int2str(costp)]);
% costs=[costs; [cost/M/N, costp/M/N, M, N]];
%end
% eastswest=cover_image(1:M,2:end)-cover_image(1:M,1:end-1);%east-west
% westseast=cover_image(1:M,1:end-1)-cover_image(1:M,2:end);%west-east
% southsnorth=cover_image(2:M,:)-cover_image(1:M-1,:);%south-north
% northssouth=cover_image(1:M-1,:)-cover_image(2:M,:);%north-south
% nessw=cover_image(1:M-1,2:end)-cover_image(2:M,1:end-1);%northeast-southwest
% swsne=cover_image(2:M,1:end-1)-cover_image(1:M-1,2:end);%southeast-northwest
% nwsse=cover_image(1:M-1,1:end-1)-cover_image(2:M,2:end);%northwest-southeast
% sesnw=cover_image(2:M,1:end-1)-cover_image(1:M-1,2:end);%southwest-northeast
% ewsum=ewsum+double(eastswest);
% wesum=wesum+double(westseast);
% snsum=snsum+double(southsnorth);
% nssum=nssum+double(northssouth);
% neswsum=neswsum+double(nessw);
% swnesum=swnesum+double(swsne);
% senwsum=senwsum+double(sesnw);
% nwsesum=nwsesum+double(nwsse);
%secretcoded=secret_message;
stego=single(cover_image(1:M,1:N));
bb=cover_image(1:M,1:N);
mm2=0;
for i=0:md-1
    mask=secret_message(1:M,1:N)==i;
   % if method=='O'
     % val=row(i+1)-1;
%     else
       val=i;
%     end
    %secretcoded(mask)=val;
    stego(mask)=single(stego(mask))-single(mod(stego(mask),md))+val;
    %stego(mask)=stego(mask)-mod(stego(mask),md)+j;
   % if method~='L'
     masklt=(single(stego)-single(bb)<-md/2) & (single(stego)<256-md);
     stego(masklt)=stego(masklt)+md;
     maskgt=single(stego)-single(bb)>=md/2 & stego>=md;
     stego(maskgt)=stego(maskgt)-md;
    %end
    mm2=mm2+sum(sum((single(stego(mask))-single(bb(mask))).^2));
end
% if sec=='R'
%  costs1=[costs1; [mm2/M/N, M,N, pictures(pict)]];
% else
%  costs1=[costs1; [mm2/M/N, M,N, pictures(pict), pictures(pict1)]];
% end
secret1=mod(stego,md);%encoded
secret2=secret1;
%if method=='O'
%  for i=0:md-1%decoding
%       val=row(i+1)-1;    
%       mask1=secret1(1:M,1:N)==val;
%       secret2(mask1)=i;%decoded
%  end
%end
% if sec=='R'
  if sum(sum(secret_message(1:M,1:N)~=secret2(1:M,1:N)))~=0
      disp('Extracted is not the same as embedded for R');
      return;%break;
  end
% else
%     if sec=='H'
%        M1=M/parts;
%        for i=1:parts
%            secr(1:M1,1:N,i)=secret2((i-1)*M1+1:i*M1,1:N);
%        end
%        secrm1(1:M1,1:N)=secr(1:M1,1:N,1);
%     for i=2:parts
%       secrm1=secrm1+secr(1:M1,1:N,i)*md^(i-1);
%     end
%     if sum(sum(secrmm(1:M1,1:N)~=secrm1(1:M1,1:N)))~=0
%        disp('Extracted is not the same as embedded for H');
%        break;
%     end
%     elseif sec=='V'
%          N1=N/parts;
%        for i=1:parts
%            secr(1:M,1:N1,i)=secret2(1:M,(i-1)*N1+1:i*N1);
%        end
%        secrm1(1:M,1:N1)=secr(1:M,1:N1,1);
%     for i=2:parts
%       secrm1=secrm1+secr(1:M,1:N1,i)*md^(i-1);
%     end
%     if sum(sum(secrmm(1:M,1:N1)~=secrm1(1:M,1:N1)))~=0
%        disp('Extracted is not the same as embedded for V');
%        break;
%     end
%     end
%     
%         
%end
%end
%end
MSE=mm2/M/N;
stego_image=stego;
end