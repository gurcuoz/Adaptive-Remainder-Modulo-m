%clc;
%clear;
features=[];
training=input('Training? Yes - 1; No - 0 normally, 1 ');
threshold=input('Threshold? normally 3 = ');
 %path='C:\Temp';
 %path='C:\Temp\ucid.v2';
 path='C:\Temp\BOSS1.01';
 for i=1:2 %1:2
      % path='D:\Ucheba\Masters\Hajer\Hajer Thesis\GRAY SCALE\LSBCT\pictures\pictures';
       %if i==2
           j=input('i= normally at first 2, then used directory 3..');
       %end
       %diri=fullfile(path,char(i+1+48));
       if j>0
        diri=fullfile(path,int2str(j));
       else
        diri=fullfile(path,'BOSSbase_1.01');
       end
       %diri=fullfile(path,char(i+49));
       %names=dir(fullfile(diri,{'*.bmp','*.jpg','*.tiff'}));
       
%        if training==1
%          names=fullfile(diri,{'*.bmp','*.jpg','*.tiff'});%for training
%        else
%         names=fullfile(path,{'*.bmp','*.jpg','*.tiff', '*.tif','*.png'});%for testing
%        end
       %names=fullfile(path,'\1\','*.tif');%for testing
       %names=fullfile(path,'\2\','*.tif');%for testing
       %names=fullfile(diri,'*.tif');%for testing
       names=fullfile(diri,'*.pgm');%for testing
       %names_n=numel(names);
%        files={};
%        for in=1:names_n
%         filenames=dir(char(names(in)));
%         files=cat(2,files,{filenames.name});
%        end
        filenames=dir(char(names));
%       diri=fullfile(path,char(i+48)); 
%       [status, msg, msgID]= mkdir (diri);
        %file_n=numel(files);
        %file_n=numel(filenames)/2;
        file_n=numel(filenames);
       %for i1=3:names_n 
%        for i1=1:numel(files)
%         %filename = char(fullfile(diri, names(i1).name));  
%         filename = char(fullfile(diri, files(i1)));  
%        %filename = cat(2, 'D:\Ucheba\Masters\Hajer\Hajer Thesis\GRAY SCALE\LSBCT\pictures\pictures\',  char(pictures(pict)));  
%         image=imread(filename);
%         [MD MDD]=spam06052018(image, 4);%2nd parameter, T, threshold for SPAM
%         [F11 F12]=firstorder(MD,1);% 1st ordet statistics
%         [F21 F22]=firstorder(MDD,2);% 2nd order statistics
%         features(i1,1:size(F11,1)*size(F11,2))=reshape(F11,1,[]);
%         features(i1,1+size(F11,1)*size(F11,2):size(F11,1)*size(F11,2)+size(F12,1)*size(F12,2))=reshape(F12,1,[]);   
%        end
       if training==1
         fln=char(fullfile(diri,'features1TR.dat'));
       else
        fln=char(fullfile(diri,'features1TT.dat'));
       end
       flnh=fopen(fln,'r');
       features=[features; fread(flnh,[file_n 2*(2*threshold+1)^3],'double')];
       %features(301:600,:)=fread(flnh,[file_n 2*(2*threshold+1)^3],'double');
       fclose(flnh);
%        if i~=1
%            i=2;
%        end
end
features(1:size(features,1),size(features,2)+1)=0;%natural images
features(size(features,1)/2+1:size(features,1),size(features,2))=1;%embedded images
TT=array2table(features);

