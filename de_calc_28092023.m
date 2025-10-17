de=(sum(yfit(1:size(yfit,1)/2)==1)+ sum(yfit(size(yfit,1)/2+1:end)==0))/size(yfit,1)
fp=sum(yfit(1:size(yfit,1)/2)==1)
fn=sum(yfit(size(yfit,1)/2+1:end)==0)