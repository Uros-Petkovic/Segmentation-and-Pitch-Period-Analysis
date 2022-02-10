
function y_ck=Centralno_klipovanje(y,cl)
    
N=length(y);
y_ck=zeros(1,N);
for i=1:N
    if(y(i)<=cl && y(i)>=-cl)
        y_ck(i)=0;
    else
        if(y(i)>cl)
            y_ck(i)=y(i)-cl;
        else
            y_ck(i)=y(i)+cl;
        end
    end
end
end
    