function [x,Po] = myViterbi(v,p,a,b)

% inputs:
% v - opservation sequence
% p - initial probabilities
% a - transition probabilities
% b - emission probabilities

% Po - probability
% x - state sequence

N=length(p);
T=length(v);


    disp('DHMM')
    % initialization
    delta(1,:)=p.*b(:,v(1))'; %pi pocetne ver,bi ono sto odgovara prvoj opservaciji i stanjima
    psi(1,:)=zeros(1,N);  %Psi postavljamo na nulu

    % recursion
    for t=2:T
        for j=1:N
            [delta(t,j),psi(t,j)]=...
                max(delta(t-1,:).*a(:,j)');
            delta(t,j)=delta(t,j)...
                *b(j,v(t));
%             [delta(t,j),psi(t,j)]=...
%                 max(delta(t-1,:).*a(:,j)'...
%                 *b(j,v(t)));
        end
    end
%
    % termination
    [Po(T),x(T)]=max(delta(T,:));

    for t=T-1:-1:1
        x(t)=psi(t+1,x(t+1));
        Po(t)=delta(t,x(t));
    end
    
    

