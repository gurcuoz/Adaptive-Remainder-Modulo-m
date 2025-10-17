% TT1(1:1000,:)=TT(1:1000,:);
% TT1(1001:2000,:)=TT(1501:2500,:);
% TT2(1:500,:)=TT(1001:1500,:);
% TT2(501:1000,:)=TT(2501:3000,:);
clear TT1 TT2;
sz=size(TT,1);
sz2=sz/2;
chunk=input('chunk originally it was 800 = ');
TT1(1:chunk,:)=TT(1:chunk,:);
TT1(chunk+1:2*chunk,:)=TT(sz2+1:sz2+chunk,:);
%TT1(801:1600,:)=TT(887:1686,:);
TT2(1:sz2-chunk,:)=TT(chunk+1:sz2,:);
%TT2(1:86,:)=TT(801:886,:);
TT2(sz2-chunk+1:2*(sz2-chunk),:)=TT(sz2+chunk+1:end,:);
%TT2(87:172,:)=TT(1687:1772,:);
%
%rng(123);
% index_train=randperm(1500,1000);
% index_train=randperm(1500,1000);
% index_train=randperm(1500,1000);
% index_train=randperm(1500,1000);
% index_test=setdiff(1:1500, index_train);
% clear TT1 TT2;
% TT1(1:1000,:)=TT(index_train,:);
% TT1(1001:2000,:)=TT(1500+index_train,:);
% TT2(1:500,:)=TT(index_test,:);
% TT2(501:1000,:)=TT(1500+index_test,:);

