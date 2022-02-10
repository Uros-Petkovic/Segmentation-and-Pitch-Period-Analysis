function  rxx = auto_correlation(x)

N=length(x);
for k=0:N-1
    rxx(k+1)=0;
    for n=0:N-1-k
        rxx(k+1)=rxx(k+1)+x(n+1)*x(n+1+k);
    end
    rxx(k+1)=rxx(k+1)/N;
end