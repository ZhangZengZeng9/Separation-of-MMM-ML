function [mae,mse,rmse,mape,error,errorPercent]=calc_error(x1,x2)

if nargin==2
    if size(x1,2)==1
        x1=x1'; 
    end
    
    if size(x2,2)==1
        x2=x2';  
    end
    
    num=size(x1,2);
    error=x2-x1; 
    errorPercent=abs(error)./x1; 
    
    mae=sum(abs(error))/num; 
    mse=sum(error.*error)/num;  
    rmse=sqrt(mse);     
    mape=mean(errorPercent);  
    
    disp(['mae: ',num2str(mae)])
    disp(['mse£º',num2str(mse)])
    disp(['rmse: ',num2str(rmse)])
    disp(['mape£º',num2str(mape*100),' %'])
    
else
    disp('Function call method is incorrect, please check the input parameter')

end

end

