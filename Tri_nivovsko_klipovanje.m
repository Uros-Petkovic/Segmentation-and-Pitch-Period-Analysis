
function y_3k=Tri_nivovsko_klipovanje(y,cl)

N=length(y);
y_3k=zeros(1,N);
for i=1:N
    if (y(i)<cl && y(i)>-cl)
        y_3k(i)=0;
    else
        if(y(i)>=cl)
            y_3k(i)=1;
        else
            y_3k(i)=-1;
        end
    end
end
end