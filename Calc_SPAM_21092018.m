clc;
clear;
threshold=3;folder1=2; 
folder1=input('Folder#?/2..10: ');
folder2=folder1;
training=input('Training? Yes - 1; No - 0');
for i=folder1:folder2 %2:2 %5:5 %4:4 %3:3 % 1:2%9
       %path='D:\Ucheba\Masters\Hajer\Hajer Thesis\GRAY SCALE\LSBCT\pictures\pictures';
       %path='C:\Temp';
      % path='C:\Temp\ucid.v2';
       path='C:\Temp\BOSS1.01';
       
       %diri=fullfile(path,char(i+48));
       if i>0
        diri=fullfile(path,int2str(i));
       else
         diri=fullfile(path,'BOSSbase_1.01');;
       end
       %names=dir(fullfile(diri,{'*.bmp','*.jpg','*.tiff'}));
%        if training==1
%          names=fullfile(diri,{'*.bmp','*.jpg','*.tiff'});%for training
%        else
%          names=fullfile(path,{'*.bmp','*.jpg','*.tiff', '*.tif','*.png'});%for testing
%        end
       %names=fullfile(diri,{'*.tif'});
       names=fullfile(diri,{'*.pgm'});
       filenames=dir(char(names));
       names_n=numel(filenames);
       files={};
       for in=1:names_n
        %filenames=dir(char(names(in)));
        files=cat(2,files,filenames(in).name);
       end
      %diri=fullfile(path,char(i+48)); 
     % diri=fullfile(path,int2str(i)); 
%       [status, msg, msgID]= mkdir (diri);
       %for i1=3:names_n
       %names_n=names_n-2;
%        if training==1
%          filestart=1; fileend=names_n/2;
%        else
%          filestart=names_n/2+1; fileend=names_n;
%        end
       filestart=1; fileend=names_n;
       if training==1
         fln=char(fullfile(diri,'features1TR.dat'));%for training
       else
         fln=char(fullfile(diri,'features1TT.dat'));%for testing
       end
       flnh=fopen(fln,'w');
       %flnh=fopen(fln,'a');
       for i1=filestart:fileend
        %filename = char(fullfile(diri, names(i1).name));  
        filename = char(fullfile(diri, files(i1)));  
       %filename = cat(2, 'D:\Ucheba\Masters\Hajer\Hajer Thesis\GRAY SCALE\LSBCT\pictures\pictures\',  char(pictures(pict)));  
       %features2=spam686_1(filename);
       features2=spam686(filename);
       %features2=spam686(filename);
%        image=imread(filename);
%        [MD MDD]=spam06052018(image, threshold);%2nd parameter, T, threshold for SPAM
%         %MD=spam06052018(image, threshold);%2nd parameter, T, threshold for SPAM
%         [F11 F12]=firstorder(MD,1);% 1st ordet statistics
%         [F21 F22]=firstorder(MDD,2);% 2nd order statistics
%         %features(i1,1:size(F11,1)*size(F11,2))=reshape(F11,1,[]);
%         %features(i1,1+size(F11,1)*size(F11,2):size(F11,1)*size(F11,2)+size(F12,1)*size(F12,2))=reshape(F12,1,[]);   
%         features1=[F11(:); F12(:)];
%         features2=[F21(:); F22(:)];
        fwrite(flnh,features2,'double');
       end
       fclose(flnh);
     end
