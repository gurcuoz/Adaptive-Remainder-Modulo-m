function [RM SM UM, RMM SMM UMM] = RSsteganalysis(stego_image, mask)
 [M N]=size(stego_image);
 [m1 m2] = size(mask);
 if mod(M,m1)~=0
   M1=M-mod(M,m1);
 else
     M1=M;
 end
 if mod(N,m2)~=0
   N1=N-mod(N,m2);
 else
     N1=N;
 end
 groups=stego_image(1:M1,1:N1)';
 %groups=reshape(groups,512,2,[]);
 %groups=reshape(groups,M1,2,[]);
 %groups=groups';
 %groups=permute(groups, [2 1 3]);
 %groups=reshape(groups, 2,2,[]);
 groups=reshape(groups, N1,m1,[]);
 groups=permute(groups, [2 1 3]);
 groups=reshape(groups, m1,m2,[]);
 fg=discr_f(groups);
 FMg=disturb(groups, mask);
 FMMg=disturb(groups,-mask);
 fFMg=discr_f(FMg);
 fFMMg=discr_f(FMMg);
 [RM SM UM]=calcRSU(fFMg, fg);
 [RMM SMM UMM]=calcRSU(fFMMg, fg);
end

function difgr=discr_f(groups)
 %difgr=abs(groups(1,1,:)-groups(1,2,:))+abs(groups(2,1,:)-groups(2,2,:));
 [m1 m2 m3]=size(groups);
 absdif=abs(groups(:,2:m2,:)-groups(:,1:m2-1,:));
 difgr=sum(absdif);
 if size(difgr,2)>1
     difgr=sum(difgr);
 end
end

%function gr=disturb(groups, one)
function gr=disturb(groups, mask)
[M1 M2, M3]=size(groups);
maskrep=repmat(mask,1,1,M3);
range1=(maskrep==1);
range2=(mod(groups,2)==0) & (groups~=0);
range= range1 & range2;
range3=(mod(groups,2)==1);
groups(range)=groups(range)+1;
range=range1 & range3;
groups(range)=groups(range)-1;
range1=(maskrep==-1);
range2=(mod(groups,2)==0) & (groups~=0);
range= range1 & range2;
range3=(mod(groups,2)==1);
groups(range)=groups(range)-1;
range=range1 & range3;
groups(range)=groups(range)+1;
gr=groups;
end

function [RM SM UM]=calcRSU(fFMg, fg)
 RM=sum(fFMg>fg);
 SM=sum(fFMg<fg);
 UM=sum(fFMg==fg);
end