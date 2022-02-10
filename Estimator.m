function e = Estimator(y,lambda, tau, win)

%Sacekamo blanking time od maksimuma,onda pocinjemo da opadamo
%Kada eksp. funkcija udari u naredni maksimum i uzimamo ukupno vreme
%izmedju ta dva maksimuma

crtaj = 0; % sa ukljucenom opcijom za crtanje, moze se pogledati kako 
% signal izgleda i uociti u okviru njega periodicnost koju i ovaj 
% estimator treba da uoci. moze pomoci oko boljeg izbora parametara
if crtaj
    figure;
    hold on
    plot(y);
end
    
j=1;
while y(j)==0 && j<(win - 1)  %Za slucaj da nismo bas na prvom maksimumu
    j=j+1;   %bice pomereniji,pa zato povecavamo indeks dok ne naidjemo na max
end
mt=y(j);t=j;        %cekamo tau
j=min(win,j+tau-1); %dodavanje blanking perioda 3ms
if j==win   %Za slucaj da smo udarili u kraj prozora,ne uzimaj tu procenu
    e=NaN;  %jer to nije dovoljno merodavno,a desice se kad je npr. pik na
    return  %polovini ili pri kraju prozora,pa nam blanking time pretekne preko
end         %cele duzine prozora,pa je to NaN
if crtaj
    plot([t j],[mt mt],'g')
end
ind=0;
e1=[];
while j<win % eksponencijalno opadajuci deo
    while y(j)<mt*exp(-lambda*(ind)) && j<win  %pustamo nasu fcn da opada
        j=j+1;   %trazimo kada je vrednost sledeceg max veca od eksp dela
        ind=ind+1; %i pazimo da ne izadjemo iz prozora
        if crtaj
            plot(j,mt*exp(-lambda*(ind)),'g*');
        end
    end    %t je u stvari prvi trenutak kada smo nasli prvi max,zato oduzimamo sad
    e=j-t;  %dobijamo info o tome koliko iznosi nasa perioda
    if (j<win)  %ako posmatramo jedan prozor i onje obuhvatio vise od 2 maksimuma,da potrazi
        e1=[e1 e]; %i tu drugu procenu kao onu koja je veca od te dve
    else
        e1=[e1 NaN];
    end
    mt=y(j); t=j; j=min(win,j+tau-1); ind=0;
end
if crtaj
    pause
end
e=nanmax(e1);  %taj maks se nalazi ovde,max koji nije NaN
end
