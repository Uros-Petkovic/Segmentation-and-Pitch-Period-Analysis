
%% Prvi zadatak 1 i 2 - Vizuelizacija i segmentacija reci 

%Vizuelni i audio prikaz snimljene sekvence

clc; clear all;

[y,fs]=audioread('SekvencaSegm.wav');
T=1/fs;
y=y(4001:68000);   % y vektor duzine 64000
figure(1);
plot(T:T:length(y)*T,y);
hold on;
title('Ulazni signal');
%sound(y,fs);

% Zdravo, ja sam Uros Petkovic, zivim i studiram u Beogradu.

% Filtriranje signala

Wn=[100 3900]/(fs/2); % definisemo normalizovani opseg ucestanosti
[B,A]=butter(6,Wn,'bandpass');
y_f=filter(B,A,y);
%sound(y_f,fs); % ukoliko je mikrofon iole kvalitetan, ne bi trebalo da se cuje razlika, 
figure(1);
plot(T:T:length(y)*T,y_f,'r'); % eventualno se vidi razlika na grafiku
title('Snimljeni govorni signal pre i posle filtriranja'); % talasni oblik
xlabel('vreme');
ylabel('govorni signal');
legend('Pocetni signal','Filtrirani signal');
y=y_f;
%%
% Kratkovremenska energija i ZCR

%Signal ima vecu energiju tamo gde ima govornog segmenta
%ZeroCross tamo gde je zvucni govor ZCR je nesto nizi,vece su amplitude,pa
%je manje prolazaka kroz nulu, tamo gde je sum cinculira oko nule

wl = fs*20e-3; % prozor od 20 milisekundi (u odbircima)
E = zeros(size(y)); % energija
Z = zeros(size(y)); % brzina prolaska kroz nulu

for i = wl : length(y)
    rng = (i-wl+1) : i-1; % definisanje prozora od kog do kog indeksa
    %prethodnih wl odbiraka gledamo,mogli smo i do i,ali stoji -1 da bismo
    %sveli na kauzalni oblik ovaj deo rng+1
    E(i) = sum(y(rng).^2);% suma kvadrata prethotnih wl odbiraka,posmatramo
    %u tom prozoru
    Z(i) = sum(abs(sign(y(rng+1))-sign(y(rng))));% broj promena znaka
end
Z=Z/2/wl;  %Delimo sa 2 jer svaki ovaj Z ulazi sa 2 i normalizujemo sa duzinom
%prozora da to ne bi uticalo na Z,ali kasnije cemo postavljati prag neki za
%Z tako da ovo /2/wl nece ni morati,samo cemo postaviti drugaciju konstantu

% Prikaz talasnih oblika i kratkorocne energije
time=T:T:length(y)*T;
figure(2)
plotyy(time,y,time,E);
title('Talasni oblik i kratkorocna energija')
%Energija je velika tamo gde ima signala

% Prikaz talasnih oblika i ucestanosti presecanja nule
figure(3)
plotyy(time,y,time,Z);
title('Talasni oblik i ucestanost presecanja nule')
% Manje vrednosti ZCR na mestima gde imamo govorni signal
%%
% Segmentacija govornog signala

ITU=0.09*max(E);
varE=var(E(32000:35000));
ITL = 7* varE^0.5;
%ITL1=0.003*max(E);

pocetak_reci=[];
kraj_reci=[];

for i=2:length(y)
    if (E(i-1)<ITU)&&(E(i)>=ITU)
        pocetak_reci=[pocetak_reci i]; %prethodni manji od praga,a sledeci veci
    end                         %pocetak reci              
    if(E(i-1)>ITU)&&(E(i)<=ITU) %prethodni manji,a sledeci beci,kraj reci
        kraj_reci=[kraj_reci i];
    end
end

pocetak=pocetak_reci;
kraj=kraj_reci;

% rec je signal koji je 0 kada nema reci, max(E) kada ima reci
rec=zeros(length(y),1);
for i=1:length(pocetak)
    rec(pocetak(i):kraj(i),1)=max(E)*ones(kraj(i)-pocetak(i)+1,1); 
end

% pomeranje levo/desno do donjeg praga
for i=1:length(pocetak)
    pomeranje=pocetak(i);
    while(E(pomeranje)>ITL)
        pomeranje=pomeranje-1; % pomeranje u levu stranu
    end
    pocetak(i)=pomeranje; %Kad udarimo u donji prag,prekinemo pricu
end

for i=1:length(kraj)
    pomeranje=kraj(i);
    while(E(pomeranje)>ITL)
        pomeranje=pomeranje+1; % pomeranje u levu stranu
    end
    kraj(i)=pomeranje; 
end
%Ono sto bi moglo da se desi je da neki pocetak udari u kraj,da se mimoidju
%ili da udare jedan u drugi
%Npr.drugi pocetak pomeramo levo i on udari u prvi pocetak zbog praga,zato
%moramo da brisemo duple pocetke i krajeve,da ne bismo imali duple i laznu
%informaciju u broju reci,zato ostavljamo samo razlicite pocetke i krajeve
% brisanje duplih pocetaka za slucaj da se pojave

pocetak1(1)=pocetak(1);
k=1;
for i=2:length(pocetak)
    if pocetak(i)~= pocetak1(k)
        k=k+1;
        pocetak1(k)=pocetak(i);
    end
end

kraj1(1)=kraj(1);
k=1;
for i=2:length(kraj)
    if kraj(i)~= kraj1(k)
        k=k+1;
        kraj1(k)=kraj(i);
    end
end

clear rec pocetak kraj

pocetak=[pocetak1(1:5) pocetak1(7:11)];
kraj=[kraj1(1:4) kraj1(6:11)];

%Krajnje dobijeni signal je sa tacno koliko reci imamo,gde su poceci i
%krajevi
rec=zeros(length(y),1);
for i=1:length(pocetak)
    rec(pocetak(i):kraj(i),1)= max(E)*ones(kraj(i)-pocetak(i)+1,1);
end
figure(4); hold on; 
plot(time,E,'b',time,rec/2,'g');
hold off
xlabel('Vreme[s]');
ylabel('Energija i reci');
title('Segmentisane reci: ZDRAVO, JA SAM UROS PETKOVIC, ZIVIM I STUDIRAM U BEOGRADU');

%% Preslusavanje reci

sound(y(pocetak(1):kraj(1)),fs);
pause;
sound(y(pocetak(2):kraj(2)),fs);
pause;
sound(y(pocetak(3):kraj(3)),fs);
pause;
sound(y(pocetak(4):kraj(4)),fs);
pause;
sound(y(pocetak(5):kraj(5)),fs);
pause;
sound(y(pocetak(6):kraj(6)),fs);
pause;
sound(y(pocetak(7):kraj(7)),fs);
pause;
sound(y(pocetak(8):kraj(8)),fs);
pause;
sound(y(pocetak(9):kraj(9)),fs);
pause;
sound(y(pocetak(10):kraj(10)),fs);
%%
figure(5);
histogram(Z(32000:35000),20)
hold on
histogram(Z(57600:60000),20)
title('Histogramska procena praga ZCR');
xlabel('Vrednost praga');
hold on
stem(0.14875,400,'r','filled');
hold off
%Prag za ZCR jednak 0.14875

%% ZCR

%Posmatramo 25 frejmova levo od pocetaka svih reci i 25 frejmova
%desno od krajeva svih reci i gledamo koliko puta je ZCR u tih 25 frejmova
%pre pocetka odnosno posle kraja bio iznad nekog praga-TZCR)
%Ako je u 25 frejmova levo od pocetka ZCR bio 3 ili vise puta iznad praga,
%novi pocetak posmatrane reci postaje tacka koja je u okviru tih 25
% frejmova najdalja od starog pocetka reci-najlevlja i pritom je u njoj ZCR
% veci od TZCR; na slican nacin se formira i novi kraj samo sto ce novi
% kraj biti desno u odnosu na stari kraj

ZPrag=0.14875;
novi_pocetak=[];
for i=1:length(pocetak)
j=pocetak(i);   %trenutni pocetak
PredjenPrag1=0;
    while(j>1 && j>pocetak(i)-25*wl)  %Gledamo prethodnih 25 odbiraka
        if (Z(j)>ZPrag)   %Ako je veci od praga, povecaj brojac za 1
           PredjenPrag1=PredjenPrag1+1;
           if (PredjenPrag1==1)
               k=j;
           end
        end
        j=j-1; %Idi unazad do 25 koraka pre
    end
    if (PredjenPrag1>=3)  %Ako je predjen prag 3 i vise puta, onda je to nas novi pocetak
       novi_pocetak(i)=k-1;
    else
       novi_pocetak(i)=pocetak(i); %Ako nije vise od 3 puta, onda staje prethodni pocetak
    end
end

novi_kraj=[];
for i=1:length(kraj)
j=kraj(i);
PredjenPrag2=0;
    while(j<length(E) && j<kraj(i)+25*wl) %Gledamo 25 odbiraka nakon kraja dok ne dodjemo do kraja sekvence
        if (Z(j)>ZPrag)  %Ako je Z veci od praga, povecavamo brojac
           PredjenPrag2=PredjenPrag2+1;
           if (PredjenPrag2==1)
               k=j;
           end
        end
        j=j+1;  % Pomeramo se za jedno mesto gore
    end
    if (PredjenPrag2>=3)  %Ako smo presli prag vise od 3 puta
       novi_kraj(i)=k+1;    %ovo je nas novi kraj
    else
       novi_kraj(i)=kraj(i);   %Ako nismo, onda je nas stari kraj ostao
    end
end

rec=zeros(length(y),1);
for i=1:length(novi_pocetak)
   rec(novi_pocetak(i):novi_kraj(i),1)=max(E)*ones(novi_kraj(i)-novi_pocetak(i)+1,1); 
end
figure(6)
plot(T:T:length(E)*T,E,T:T:length(rec)*T,rec) 
title('Izdvojene reci nakon pomeranja usled posmatranja ZCR')
xlabel('vreme[s]');

figure(7)
plot(T:T:length(y)*T,y)
hold all
for i=1:length(novi_pocetak)
    plot([novi_pocetak(i)/length(y)*8 novi_pocetak(i)/length(y)*8],[min(y) max(y)],'r')
    plot([novi_kraj(i)/length(y)*8 novi_kraj(i)/length(y)*8],[min(y) max(y)],'r')
    line([novi_pocetak(i)/length(y)*8 novi_kraj(i)/length(y)*8], [max(y) max(y)], 'Color', 'red'); 
    line([novi_pocetak(i)/length(y)*8 novi_kraj(i)/length(y)*8], [min(y) min(y)], 'Color', 'red');
end
hold off
title('Filtrirani govorni signal i izdvojene reci')
xlabel('vreme[s]');

%% Prvi zadatak 3a - Izdvajanje Pitch periode


clc; clear all;

[x,fs]=audioread('PitchSekvenca.wav');
T=1/fs;
n=length(x);
figure(1);
t=T:T:length(x)*T;
plot(t,x);
title('Ulazni govorni signal');
axis([0 n*T min(x) max(x)]);
hold on

xc=x;
[B,A]=butter(6,300/(fs/2)); % jer je Fpitch manje od 300Hz
xc=filter(B,A,xc);
plot(t,xc);
title('Ulazni i filtrirani signal');
legend('Ulazni signal','Filtrirani signal');
xlabel('vreme[s]');
ylabel('Amplituda');
hold off

y1=xc(5200:13500); N1=length(y1);
y2=xc(19800:29600);N2=length(y2);
y3=xc(33500:41500);N3=length(y3);
y4=xc(46000:54300);N4=length(y4);
y5=xc(56500:65100);N5=length(y5);
y6=xc(67000:73800);N6=length(y6);

%Preslusavanje izdvojenih reci
sound(y1,fs);
pause;
sound(y2,fs);
pause;
sound(y3,fs);
pause;
sound(y4,fs);
pause;
sound(y5,fs);
pause;
sound(y6,fs);
pause;

% Formiranje povorki impulsa za svaku rec
%%
m=1;
for xcf={y1, y2, y3, y4, y5, y6}

    m1=zeros(1,length(xcf{1}));
    m2=zeros(1,length(xcf{1}));
    m3=zeros(1,length(xcf{1}));
    m4=zeros(1,length(xcf{1}));
    m5=zeros(1,length(xcf{1}));
    m6=zeros(1,length(xcf{1}));
    m4pom=zeros(1,length(xcf{1}));
    maksp=[];
    minp=[];

    %Formiranje m1 i m4     nalazimo maksimume i minimume
    for i=2:length(xcf{1})-1
        if xcf{1}(i)>xcf{1}(i-1) && xcf{1}(i)>xcf{1}(i+1)
            m1(i)=max(0,xcf{1}(i));
            maksp=[maksp i];
        end
        if xcf{1}(i)<xcf{1}(i-1) && xcf{1}(i)<xcf{1}(i+1)
            m4pom(i)=xcf{1}(i);
            m4(i)=max(0,-xcf{1}(i));   %minus potice od toga sto trazimo aps. vrednost
            minp=[minp i];
        end
    end


    %Formiranje m2 i m3  oduzimamo od m1 tmp1,a to je vrednost minimuma
    tmp1=0;tmp2=0;  %u prvom koraku oduzimamo nulu jer nemamo prethodni min
    for i=maksp      %za m3 od maksimuma skidamo prethodni maks
        m2(i)=max(0,m1(i)-tmp1);
        m3(i)=max(0,m1(i)-tmp2);
        tmp2=m1(i);   
        ind=minp(find(minp>i,1,'first'));  %citamo koji je to indeks
        tmp1=m4pom(ind);
    end

    % formiranje m5 i m6  nalazimo razlike
    tmp1=0;tmp2=0;
    for i=minp
        m5(i)=max(0,-(m4pom(i)-tmp1));
        m6(i)=max(0,-(m4pom(i)-tmp2));
        ind=maksp(find(maksp>i,1,'first'));
        tmp1=m1(ind);
        tmp2=m4pom(i);
    end

    figure();
    subplot(7,1,1);plot(xcf{1});
    title('Prikaz 6 izlaza geeneratora impulsa');
    axis([0 length(xcf{1}) -0.1 0.1 ]);
    subplot(7,1,2);plot(m1);
    axis([0 length(xcf{1}) 0 0.1 ]);
    subplot(7,1,3);plot(m2);
    axis([0 length(xcf{1}) 0 0.2 ]);
    subplot(7,1,4);plot(m3);
    axis([0 length(xcf{1}) 0 0.1 ]);
    subplot(7,1,5);plot(m4);
    axis([0 length(xcf{1}) 0 0.1 ]);
    subplot(7,1,6);plot(m5);
    axis([0 length(xcf{1}) 0 0.2 ]);
    subplot(7,1,7);plot(m6);
    axis([0 length(xcf{1}) 0 0.1 ]);
    xlabel('Vreme[s]');

    % Procena pitch periode

    %Postavimo se na prvi maksimum

    Npoc=max(find(m1~=0,2,'first'));
    m1=m1(Npoc:end);
    m2=m2(Npoc:end);
    m3=m3(Npoc:end);
    m4=m4(Npoc:end);
    m5=m5(Npoc:end);
    m6=m6(Npoc:end);
    N=length(m1);

    win=round(fs*15e-3); %Prozor od 15ms
    NN=floor(N/(win/2)); %koliko cu prozora imati u okviru cele sekvence
    lambda = 120/fs;   %da ne odemo preko najvece
    tau=round(fs*3e-3);
    % tau=30; sacekamo dovoljno dugo da bude veca od najmanje periode

    E1=zeros(1,NN);
    E2=zeros(1,NN);  %sve procene na nula na pocetku
    E3=zeros(1,NN);
    E4=zeros(1,NN);
    E5=zeros(1,NN);
    E6=zeros(1,NN);
    E=zeros(1,NN);

    i=2;
    for i_win=1:win/2:N-win+1  %Radimo procenu za svaki prozor
        % prvi estimator 
        y=m1(i_win:i_win+win-1);
        E1(i)=Estimator(y,lambda,tau,win);
    
        % drugi estimator
        y=m2(i_win:i_win+win-1);
        E2(i)=Estimator(y,lambda,tau,win);

        % treci estimator
        y=m3(i_win:i_win+win-1);
        E3(i)=Estimator(y,lambda,tau,win);

        % cetvrti estimator
        y=m4(i_win:i_win+win-1);
        E4(i)=Estimator(y,lambda,tau,win);

        % peti estimator
        y=m5(i_win:i_win+win-1);
        E5(i)=Estimator(y,lambda,tau,win);

        % sesti estimator
        y=m6(i_win:i_win+win-1);
        E6(i)=Estimator(y,lambda,tau,win);
    
        E(i)=nanmedian([E1(i) E2(i) E3(i) E4(i) E5(i) E6(i) E(i-1)]); % da bi se zanemarile nan vrednosti
        i=i+1;   %Na kraju radimo medijan svih procena,ovih 6 i prethodne procene
    end

    figure(); 
    subplot(7,1,1); plot(1./(E1/fs));
    title('Procena pitch periode za svaki izlaz generatora');
    axis([0 length(1./(E1/fs)) 0 400 ]);
    subplot(7,1,2); plot(1./(E2/fs));
    axis([0 length(1./(E2/fs)) 0 400 ]);
    subplot(7,1,3); plot(1./(E3/fs));
    axis([0 length(1./(E3/fs)) 0 400 ]);
    subplot(7,1,4); plot(1./(E4/fs));
    axis([0 length(1./(E4/fs)) 0 400 ]);
    subplot(7,1,5); plot(1./(E5/fs));
    axis([0 length(1./(E5/fs)) 0 400 ]);
    subplot(7,1,6); plot(1./(E6/fs)); %podelimo sa fs jer je u odbircima
    axis([0 length(1./(E6/fs)) 0 400 ]);
    subplot(7,1,7); plot(1./([0 E(1:end-1)]/fs));
    axis([0 length(1./([0 E(1:end-1)]/fs)) 0 400 ]);
    xlabel('Vreme[s]');
    
    %Filtriranje
    for i=4:length(E)-3 
        Ek(i)=median(E(i-3:i+3));
    end
    Estimacija(m)=nanmedian(1./(Ek/fs));
    figure(); plot(1./(Ek/fs));
    axis([0 length(1./([0 E(1:end-1)]/fs)) 80 220]);
    title('Estimacija pitch periode date reci');
    xlabel('Vreme[s]');
    ylabel('Vrednost');
    disp(['Procenjena pitch perioda je: ', num2str(Estimacija(m))]);
    m=m+1;
end
disp(['Ukupna procenjena pitch perioda je: ', num2str(nanmean(Estimacija))]);

% Dobijena je 127.1249 Hz

%% Prvi zadatak 3 - Procena Pitch periode na osnovu autokorelacije prva rec

%sound(y1(6000:8100),fs);
figure(5);
plot(1:length(y1),y1);
%
Rxx=autocorr(y1); % autokorelaciona funkcija bez klipovanja

figure(6)
t=T:T:length(y1)*T;
plot(t*fs, Rxx);
axis([0 length(Rxx) min(Rxx) max(Rxx) ])
title('Autokorelaciona funkcija pre klipovanja') 
xlabel('n (broj odbiraka)')
ylabel('R_{xx}[n]')

% Centralno klipovanje

cl = 0.3*max(y1);
xcfck=Centralno_klipovanje(y1,cl);
Rxx_ck=autocorr(xcfck); % autokorelaciona funkcija nakon centralnog klipovanja

figure(7)
t=T:T:length(xcfck)*T;
plot(t*fs, Rxx_ck);
axis([0 length(Rxx_ck) min(Rxx_ck) max(Rxx_ck) ])
title('Autokorelaciona funkcija sa centralnim klipovanjem (Rxx_ck)') 
xlabel('n (broj odbiraka)')
ylabel('R_{xx}[n]')


% Tri nivovsko klipovanje  

cl = 0.3*max(y1);
xcf3k=Tri_nivovsko_klipovanje(y1, cl);
Rxx_3k=autocorr(xcf3k);

figure(8)
t=T:T:length(xcf3k)*T;
plot(t*fs, Rxx_3k);
axis([0 length(Rxx_3k) min(Rxx_3k) max(Rxx_3k) ])
title('Autokorelaciona funkcija sa tri nivovskim klipovanjem (Rxx_3k)') 
xlabel('n (broj odbiraka)')
ylabel('R_{xx}[n]')

Pitch(1)=mean([126.9841,123.07692]);

%% Procena Pitch periode na osnovu autokorelacije druga rec

% sound(y2(5000:6500),fs);
figure(5);
plot(1:length(y2),y2);
%
Rxx=autocorr(y2); % autokorelaciona funkcija bez klipovanja

figure(6)
t=T:T:length(y2)*T;
plot(t*fs, Rxx);
axis([0 length(Rxx) min(Rxx) max(Rxx) ])
title('Autokorelaciona funkcija pre klipovanja') 
xlabel('n (broj odbiraka)')
ylabel('R_{xx}[n]')

% Centralno klipovanje

cl = 0.3*max(y2);
xcfck=Centralno_klipovanje(y2,cl);
Rxx_ck=autocorr(xcfck); % autokorelaciona funkcija nakon centralnog klipovanja

figure(7)
t=T:T:length(xcfck)*T;
plot(t*fs, Rxx_ck);
axis([0 length(Rxx_ck) min(Rxx_ck) max(Rxx_ck) ])
title('Autokorelaciona funkcija sa centralnim klipovanjem (Rxx_ck)') 
xlabel('n (broj odbiraka)')
ylabel('R_{xx}[n]')


% Tri nivovsko klipovanje  

cl = 0.3*max(y2);
xcf3k=Tri_nivovsko_klipovanje(y2, cl);
Rxx_3k=autocorr(xcf3k);

figure(8)
t=T:T:length(xcf3k)*T;
plot(t*fs, Rxx_3k);
axis([0 length(Rxx_3k) min(Rxx_3k) max(Rxx_3k) ])
title('Autokorelaciona funkcija sa tri nivovskim klipovanjem (Rxx_3k)') 
xlabel('n (broj odbiraka)')
ylabel('R_{xx}[n]')

Pitch(2)=mean([121.542,123.945]);

%% Procena Pitch periode na osnovu autokorelacije treca rec

%sound(y3(3000:6000),fs);
figure(5);
plot(1:length(y3),y3);
%
Rxx=autocorr(y3); % autokorelaciona funkcija bez klipovanja

figure(6)
t=T:T:length(y3)*T;
plot(t*fs, Rxx);
axis([0 length(Rxx) min(Rxx) max(Rxx) ])
title('Autokorelaciona funkcija pre klipovanja') 
xlabel('n (broj odbiraka)')
ylabel('R_{xx}[n]')

% Centralno klipovanje

cl = 0.3*max(y3);
xcfck=Centralno_klipovanje(y3,cl);
Rxx_ck=autocorr(xcfck); % autokorelaciona funkcija nakon centralnog klipovanja

figure(7)
t=T:T:length(xcfck)*T;
plot(t*fs, Rxx_ck);
axis([0 length(Rxx_ck) min(Rxx_ck) max(Rxx_ck) ])
title('Autokorelaciona funkcija sa centralnim klipovanjem (Rxx_ck)') 
xlabel('n (broj odbiraka)')
ylabel('R_{xx}[n]')


% Tri nivovsko klipovanje  

cl = 0.3*max(y3);
xcf3k=Tri_nivovsko_klipovanje(y3, cl);
Rxx_3k=autocorr(xcf3k);

figure(8)
t=T:T:length(xcf3k)*T;
plot(t*fs, Rxx_3k);
axis([0 length(Rxx_3k) min(Rxx_3k) max(Rxx_3k) ])
title('Autokorelaciona funkcija sa tri nivovskim klipovanjem (Rxx_3k)') 
xlabel('n (broj odbiraka)')
ylabel('R_{xx}[n]')

Pitch(3)=mean([125.000,119.536]);

%% Procena Pitch periode na osnovu autokorelacije cetvrta rec

%sound(y4(3000:5000),fs);
figure(5);
plot(1:length(y4),y4);
%
Rxx=autocorr(y4); % autokorelaciona funkcija bez klipovanja

figure(6)
t=T:T:length(y4)*T;
plot(t*fs, Rxx);
axis([0 length(Rxx) min(Rxx) max(Rxx) ])
title('Autokorelaciona funkcija pre klipovanja') 
xlabel('n (broj odbiraka)')
ylabel('R_{xx}[n]')

% Centralno klipovanje

cl = 0.3*max(y4);
xcfck=Centralno_klipovanje(y4,cl);
Rxx_ck=autocorr(xcfck); % autokorelaciona funkcija nakon centralnog klipovanja

figure(7)
t=T:T:length(xcfck)*T;
plot(t*fs, Rxx_ck);
axis([0 length(Rxx_ck) min(Rxx_ck) max(Rxx_ck) ])
title('Autokorelaciona funkcija sa centralnim klipovanjem (Rxx_ck)') 
xlabel('n (broj odbiraka)')
ylabel('R_{xx}[n]')


% Tri nivovsko klipovanje  

cl = 0.3*max(y4);
xcf3k=Tri_nivovsko_klipovanje(y4, cl);
Rxx_3k=autocorr(xcf3k);

figure(8)
t=T:T:length(xcf3k)*T;
plot(t*fs, Rxx_3k);
axis([0 length(Rxx_3k) min(Rxx_3k) max(Rxx_3k) ])
title('Autokorelaciona funkcija sa tri nivovskim klipovanjem (Rxx_3k)') 
xlabel('n (broj odbiraka)')
ylabel('R_{xx}[n]')

Pitch(4)=mean([123.077,121.212]);

%% Procena Pitch periode na osnovu autokorelacije peta rec

%sound(y5,fs);
figure(5);
plot(1:length(y5),y5);
%
Rxx=autocorr(y5); % autokorelaciona funkcija bez klipovanja

figure(6)
t=T:T:length(y5)*T;
plot(t*fs, Rxx);
axis([0 length(Rxx) min(Rxx) max(Rxx) ])
title('Autokorelaciona funkcija pre klipovanja') 
xlabel('n (broj odbiraka)')
ylabel('R_{xx}[n]')

% Centralno klipovanje

cl = 0.3*max(y5);
xcfck=Centralno_klipovanje(y5,cl);
Rxx_ck=autocorr(xcfck); % autokorelaciona funkcija nakon centralnog klipovanja

figure(7)
t=T:T:length(xcfck)*T;
plot(t*fs, Rxx_ck);
axis([0 length(Rxx_ck) min(Rxx_ck) max(Rxx_ck) ])
title('Autokorelaciona funkcija sa centralnim klipovanjem (Rxx_ck)') 
xlabel('n (broj odbiraka)')
ylabel('R_{xx}[n]')


% Tri nivovsko klipovanje  

cl = 0.3*max(y5);
xcf3k=Tri_nivovsko_klipovanje(y5, cl);
Rxx_3k=autocorr(xcf3k);

figure(8)
t=T:T:length(xcf3k)*T;
plot(t*fs, Rxx_3k);
axis([0 length(Rxx_3k) min(Rxx_3k) max(Rxx_3k) ])
title('Autokorelaciona funkcija sa tri nivovskim klipovanjem (Rxx_3k)') 
xlabel('n (broj odbiraka)')
ylabel('R_{xx}[n]')

Pitch(5)=mean([114.286,111.111]);

%% Procena Pitch periode na osnovu autokorelacije sesta rec

%sound(y6(2000:3000),fs);
figure(5);
plot(1:length(y6),y6);
%
Rxx=autocorr(y6); % autokorelaciona funkcija bez klipovanja

figure(6)
t=T:T:length(y6)*T;
plot(t*fs, Rxx);
axis([0 length(Rxx) min(Rxx) max(Rxx) ])
title('Autokorelaciona funkcija pre klipovanja') 
xlabel('n (broj odbiraka)')
ylabel('R_{xx}[n]')

% Centralno klipovanje

cl = 0.3*max(y6);
xcfck=Centralno_klipovanje(y6,cl);
Rxx_ck=autocorr(xcfck); % autokorelaciona funkcija nakon centralnog klipovanja

figure(7)
t=T:T:length(xcfck)*T;
plot(t*fs, Rxx_ck);
axis([0 length(Rxx_ck) min(Rxx_ck) max(Rxx_ck) ])
title('Autokorelaciona funkcija sa centralnim klipovanjem (Rxx_ck)') 
xlabel('n (broj odbiraka)')
ylabel('R_{xx}[n]')


% Tri nivovsko klipovanje  

cl = 0.3*max(y6);
xcf3k=Tri_nivovsko_klipovanje(y6, cl);
Rxx_3k=autocorr(xcf3k);

figure(8)
t=T:T:length(xcf3k)*T;
plot(t*fs, Rxx_3k);
axis([0 length(Rxx_3k) min(Rxx_3k) max(Rxx_3k) ])
title('Autokorelaciona funkcija sa tri nivovskim klipovanjem (Rxx_3k)') 
xlabel('n (broj odbiraka)')
ylabel('R_{xx}[n]')

Pitch(6)=mean([125.000,126.9841]);
PitchPeriod=median(Pitch);
PitchPeriod

%% Procena pitch periode na osnovu autokorelacije sa prozorima
%Za svaku rec radim posebno

wl = fs*20e-3; % prozor od 20 milisekundi (u odbircima)
AKF=0;
k=1;
for i = wl : length(y1)
    rng = (i-wl+1) : i-1; 
    AKF=AKF+autocorr(y1(rng));
    k=k+1;
end
AKF=AKF/k;
figure();
plot(1:length(AKF),AKF);
title('Autokorelaciona funkcija na prozorima bez klipovanja');
xlabel('Odbirci[n]');
ylabel('Autokorelaciona funkcija');

cl = 0.3*max(y1);
y1=Centralno_klipovanje(y1,cl);
%y1=Tri_nivovsko_klipovanje(y1, cl);
wl = fs*20e-3; % prozor od 20 milisekundi (u odbircima)
AKF=0;
k=1;
for i = wl : length(y1)
    rng = (i-wl+1) : i-1; 
    AKF=AKF+autocorr(y1(rng));
    k=k+1;
end
AKF=AKF/k;

figure();
plot(1:length(AKF),AKF);
title('Autokorelaciona funkcija na prozorima sa centralnim klipovanjem');
xlabel('Odbirci');
ylabel('Autokorelaciona funkcija');

cl = 0.3*max(y1);
y1=Tri_nivovsko_klipovanje(y1, cl);
wl = fs*20e-3; % prozor od 20 milisekundi (u odbircima)
AKF=0;
k=1;
for i = wl : length(y1)
    rng = (i-wl+1) : i-1; 
    AKF=AKF+autocorr(y1(rng));
    k=k+1;
end
AKF=AKF/k;

figure();
plot(1:length(AKF),AKF);
title('Autokorelaciona funkcija na prozorima sa 3-nivovskim klipovanjem');
xlabel('Odbirci');
ylabel('Autokorelaciona funkcija');

%y1 65  123.0769
%y2 68  117.6471
%y3 69  115.9420
%y4 65  123.0769
%y5 60  133.3333
%y6 65  123.0769

%Srednja je 122.6922 Hz



%% Drugi zadatak

clc; clear all;

[y,fs]=audioread('SekvencaSegm.wav');
T=1/fs;
y=y(4001:68000);   % y vektor duzine 64000
figure(1);
plot(T:T:length(y)*T,y);
hold on;
title('Ulazni signal');
xlabel('Vreme[s]');
ylabel('Vrednost signala');
%sound(y,fs);
%%
% Zdravo, ja sam Uros Petkovic, zivim i studiram u Beogradu.

% Projektovanje mi kompanding kvantizatora

B=4;
for mi=[100 500]
    M=2^B;
    ymax=max(abs(y)); % amplituda govornog signala
    delta=2*ymax/M; % korak kvantizacije
    yq_1=ymax*(log10(1+mi*abs(y)/ymax))/(log10(1+mi)).*sign(y);
    yq_mi=round(yq_1/delta)*delta; % ovaj korak je neophodan jer pravimo mid
    % tread kvantizator, rezultat ovog kvantizatora mora biti celobrojni
    % umnozak koraka kvantizacije-zato prvo radimo zaokruzivanje na veci ceo
    % broj izraza yq_1/delta, a zatim dobijeni rezultat mnozimo sa korakom kvantizacije
    yq_mi(yq_mi>(M-1)*delta/2)=(M/2-1)*delta; % ako je M npr 8-> imamo 8
    % kvantizacionih nivoa, moze da se desi da je prethodni korak doveo do toga
    % da je yq_1 npr 4*delta (yq_1 je bilo npr 3. nesto), a kod mid tread-a 
    % sa pozitivne strane x ose idemo do 3*delta, a sa negativne do 4*delta, 
    % plus nivo na nuli-> u tom slucaju bismo imali ukupno 9 kvantizacionih 
    % nivoa umesto 8-ovaj korak omogucuje da nikada nemamo vise koraka nego sto
    % bi trebalo

    figure(2)
    plot(y,yq_mi,'*')
    hold on
    title('Mi kompanding kvantizator')
    xlabel('originalni signal (y)')
    ylabel('kvantizovani signal (y_{q})')
end
hold off
legend('B=4 mi=100','B=4 mi=500','Location','SouthEast');

B=8;
for mi=[100 500]
    M=2^B;
    ymax=max(abs(y)); % amplituda govornog signala
    delta=2*ymax/M; % korak kvantizacije
    yq_1=ymax*(log10(1+mi*abs(y)/ymax))/(log10(1+mi)).*sign(y);
    yq_mi=round(yq_1/delta)*delta; % ovaj korak je neophodan jer pravimo mid
    % tread kvantizator, rezultat ovog kvantizatora mora biti celobrojni
    % umnozak koraka kvantizacije-zato prvo radimo zaokruzivanje na veci ceo
    % broj izraza yq_1/delta, a zatim dobijeni rezultat mnozimo sa korakom kvantizacije
    yq_mi(yq_mi>(M-1)*delta/2)=(M/2-1)*delta; % ako je M npr 8-> imamo 8
    % kvantizacionih nivoa, moze da se desi da je prethodni korak doveo do toga
    % da je yq_1 npr 4*delta (yq_1 je bilo npr 3. nesto), a kod mid tread-a 
    % sa pozitivne strane x ose idemo do 3*delta, a sa negativne do 4*delta, 
    % plus nivo na nuli-> u tom slucaju bismo imali ukupno 9 kvantizacionih 
    % nivoa umesto 8-ovaj korak omogucuje da nikada nemamo vise koraka nego sto
    % bi trebalo

    figure(3)
    plot(y,yq_mi,'*')
    hold on
    title('Mi kompanding kvantizator')
    xlabel('originalni signal (y)')
    ylabel('kvantizovani signal (y_{q})')
end
hold off
legend('B=8 mi=100','B=8 mi=500','Location','SouthEast');

B=12;
for mi=[100 500]
    M=2^B;
    ymax=max(abs(y)); % amplituda govornog signala
    delta=2*ymax/M; % korak kvantizacije
    yq_1=ymax*(log10(1+mi*abs(y)/ymax))/(log10(1+mi)).*sign(y);
    yq_mi=round(yq_1/delta)*delta; % ovaj korak je neophodan jer pravimo mid
    % tread kvantizator, rezultat ovog kvantizatora mora biti celobrojni
    % umnozak koraka kvantizacije-zato prvo radimo zaokruzivanje na veci ceo
    % broj izraza yq_1/delta, a zatim dobijeni rezultat mnozimo sa korakom kvantizacije
    yq_mi(yq_mi>(M-1)*delta/2)=(M/2-1)*delta; % ako je M npr 8-> imamo 8
    % kvantizacionih nivoa, moze da se desi da je prethodni korak doveo do toga
    % da je yq_1 npr 4*delta (yq_1 je bilo npr 3. nesto), a kod mid tread-a 
    % sa pozitivne strane x ose idemo do 3*delta, a sa negativne do 4*delta, 
    % plus nivo na nuli-> u tom slucaju bismo imali ukupno 9 kvantizacionih 
    % nivoa umesto 8-ovaj korak omogucuje da nikada nemamo vise koraka nego sto
    % bi trebalo

    figure(4)
    plot(y,yq_mi,'*')
    hold on
    title('Mi kompanding kvantizator')
    xlabel('originalni signal (y)')
    ylabel('kvantizovani signal (y_{q})')
end
hold off
legend('B=12 mi=100','B=12 mi=500','Location','SouthEast');
%% Odnos signal-sum

% SNR=var(y)/var(y-yq_mi); 
% SNR_db=10*log10(SNR);-SNR prakticno
% SNR_db=4.77+6*B-20*log10(log(1+mi1))-10*log10(1+(odnos/mi1).^2+sqrt(2)*(odnos/mi1)));-SNR teorijski

for B=[4 8 12]
    for mi=[100 500]
        M=2^B;
        ymax=max(abs(y));
        delta=2*ymax/M;
        P=zeros(1,length(y)); % xmax/sigma_x
        Q=zeros(1,length(y)); % SNR

        k=1;
        for i=0.01:0.01:1
           y_utisano=y*i;
           P(k)=ymax/sqrt(var(y_utisano));
           yq_1=ymax*(log10(1+mi*abs(y_utisano)/ymax))/(log10(1+mi)).*sign(y_utisano);
           yq_mi_utisano=round(yq_1/delta)*delta; 
           yq_mi_utisano(yq_1>(M-1)*delta/2)=(M/2-1)*delta;
           yq_mi_utisano2=((10.^(abs(yq_mi_utisano)*log10(1+mi)/ymax)-1)*ymax/mi).*sign(yq_mi_utisano);
           Q(k)=var(y_utisano)/var(y_utisano-yq_mi_utisano2);
           k=k+1;
        end

        figure(3)
        semilogx(P,10*log10(Q))
        hold on
        semilogx(P,4.77+6*B-20*log10(log(1+mi))-10*log10(1+sqrt(2)*(P/mi)+(P/mi).^2))
        title('Odnos signal-sum prakticni i teorijski')
        xlabel('y_{max}/\sigma_{x}')
        ylabel('SNR[dB]')
        legend('SNR prakticno','SNR teorijski')
        axis([0 1633 0 64])
    end
end
hold off
legend('SNR prakticni B=4 mi=100','SNR teorijski B=4 mi=100','SNR prakticni B=4 mi=500','SNR teorijski B=4 mi=500','SNR prakticni B=8 mi=100','SNR teorijski B=8 mi=100','SNR prakticni B=8 mi=500','SNR teorijski B=8 mi=500','SNR prakticni B=12 mi=100','SNR teorijski B=12 mi=100','SNR prakticni B=12 mi=500','SNR teorijski B=12 mi=500','Location','SouthWest');
grid on


%% Treci zadatak a)

clear all
close all
clc

N=3; % broj stanja (urni)
M=3; % velicina alfabeta (broj boja)
T=100; % broj opservacija

for MC=1:100
    a=0.05;
    b=0.1;

    % Matrica tranzicije 
    A=[1-3*a a 2*a; b 1-2*b b; 0.1 0.1 0.8];
    % Matrica opservacija
    B=[5/8 2/8 1/8; 2/13 7/13 4/13; 1/10 3/10 6/10];
    % Matrica inicijalnih verovatnoca
    Pi=[1/3 1/3 1/3];  %svaka ima jednaku verovatnocu

    Q=zeros(1,T); % skup stanja u kojima se nalazimo u svakom  od t(100) trenutaka
    O=zeros(1,T); % skup opservacija
    Q(1)=Generator(Pi); %generisem random prvu urnu

    for t=1:T-1
        O(t)=Generator(B(Q(t),:)); %za datu urnu generisem opservacije na osnovu matrice B
        Q(t+1)=Generator(A(Q(t),:)); %za datu urnu generisem stanja na osnovu matrice A
    end

    % Problem 1

    alfa=zeros(T,N); % forward koeficijenti-alfa(t, i) je varovatnoca da je u trenutku t aktivno stanje Si

    % Inicijalizacija

    for i=1:N
        alfa(1,i)=Pi(i)*B(i, O(1));
    end

    % Indukcija

    for t=1:T-1
        for j=1:N
            for i=1:N
            alfa(t+1, i)=alfa(t+1, i)+alfa(t,j)*A(j,i)*B(i, O(t+1));
            end
        end
    end

    % Terminacija

    P1(1)=sum(alfa(T,:));

    % Problem 2

    Q1=zeros(2,T);
    P2=zeros(2,T);
    [Q1(1,:), P2(1,:)]=myViterbi(O,Pi,A,B);
    Procena1(MC)=sum(Q==Q1(1,:));

    % Treci zadatak b)

    a=0.2;
    b=0.33;

    % Matrica tranzicije 
    A=[1-3*a a 2*a; b 1-2*b b; 0.1 0.1 0.8];
    % Matrica opservacija
    B=[5/8 2/8 1/8; 2/13 7/13 4/13; 1/10 3/10 6/10];
    % Matrica inicijalnih verovatnoca
    Pi=[1/3 1/3 1/3];  %svaka ima jednaku verovatnocu


    Q=zeros(1,T); % skup stanja u kojima se nalazimo u svakom  od t(100) trenutaka
    O=zeros(1,T); % skup opservacija
    Q(1)=Generator(Pi); %generisem random prvu urnu
    
    for t=1:T-1
        O(t)=Generator(B(Q(t),:)); %za datu urnu generisem opservacije na osnovu matrice B
        Q(t+1)=Generator(A(Q(t),:)); %za datu urnu generisem stanja na osnovu matrice A
    end

    % Problem 1

    alfa=zeros(T,N); % forward koeficijenti-alfa(t, i) je varovatnoca da je u trenutku t aktivno stanje Si

    % Inicijalizacija

    for i=1:N
        alfa(1,i)=Pi(i)*B(i, O(1));
    end

    % Indukcija

    for t=1:T-1
        for j=1:N
            for i=1:N
           alfa(t+1, i)=alfa(t+1, i)+alfa(t,j)*A(j,i)*B(i, O(t+1));
            end
        end
    end

    % Terminacija

    P1(2)=sum(alfa(T,:));

    % Problem 2

    [Q1(2,:), P2(2,:)]=myViterbi(O,Pi,A,B);
    Procena2(MC)=sum(Q==Q1(2,:));
end
Procena1=median(Procena1);
Procena2=median(Procena2);

figure()
subplot(2,1,1);


figure(4)
plot(1:100,P2(1,:),'r');
hold on
plot(1:100,P2(2,:),'b');
hold off
max1=max(P2(1,:));
max2=max(P2(2,:));
axis([1 8 0 max([max1 max2])]);
title('Prikaz verovatnoca za datu sekvencu');
legend('a=0.05 b=0.1','a=0.2 b=0.33');
ylabel('Verovatnoca')
xlabel('t')
P1
% 0.3694 i 0.0383 e na -44  P1
% Procena prvog je 64%
% Procena drugog je 69%

%%
clear all
close all
clc

N=3; % broj stanja (urni)
M=3; % velicina alfabeta (broj boja)
T=100; % broj opservacija


a=0.05;
b=0.1;

% Matrica tranzicije 
A=[1-3*a a 2*a; b 1-2*b b; 0.1 0.1 0.8];
% Matrica opservacija
B=[5/8 2/8 1/8; 2/13 7/13 4/13; 1/10 3/10 6/10];
% Matrica inicijalnih verovatnoca
Pi=[1/3 1/3 1/3];  %svaka ima jednaku verovatnocu

Q=zeros(1,T); % skup stanja u kojima se nalazimo u svakom  od t(100) trenutaka
O=zeros(1,T); % skup opservacija
Q(1)=Generator(Pi); %generisem random prvu urnu

for t=1:T-1
    O(t)=Generator(B(Q(t),:)); %za datu urnu generisem opservacije na osnovu matrice B
    Q(t+1)=Generator(A(Q(t),:)); %za datu urnu generisem stanja na osnovu matrice A
end
O(T)=Generator(B(Q(T),:));
figure();
subplot(2,1,1);
stem(1:length(Q),Q);
title('Generisana sekvenca stanja i opservacija za prvi slucaj');
xlabel('Vreme');
ylabel('Stanja urni');
subplot(2,1,2);
stem(1:length(O),O);
xlabel('Vreme');
ylabel('Vrednosti opservacije');
% Problem 1

alfa=zeros(T,N); % forward koeficijenti-alfa(t, i) je varovatnoca da je u trenutku t aktivno stanje Si

% Inicijalizacija

for i=1:N
    alfa(1,i)=Pi(i)*B(i, O(1));
end

% Indukcija

for t=1:T-1
    for j=1:N
        for i=1:N
        alfa(t+1, i)=alfa(t+1, i)+alfa(t,j)*A(j,i)*B(i, O(t+1));
        end
    end
end

% Terminacija

P1(1)=sum(alfa(T,:));

% Problem 2

Q1=zeros(2,T);
P2=zeros(2,T);
[Q1(1,:), P2(1,:)]=myViterbi(O,Pi,A,B);


% Treci zadatak b)

a=0.2;
b=0.33;

% Matrica tranzicije 
A=[1-3*a a 2*a; b 1-2*b b; 0.1 0.1 0.8];
% Matrica opservacija
B=[5/8 2/8 1/8; 2/13 7/13 4/13; 1/10 3/10 6/10];
% Matrica inicijalnih verovatnoca
Pi=[1/3 1/3 1/3];  %svaka ima jednaku verovatnocu


Q=zeros(1,T); % skup stanja u kojima se nalazimo u svakom  od t(100) trenutaka
O=zeros(1,T); % skup opservacija
Q(1)=Generator(Pi); %generisem random prvu urnu
    
for t=1:T-1
    O(t)=Generator(B(Q(t),:)); %za datu urnu generisem opservacije na osnovu matrice B
    Q(t+1)=Generator(A(Q(t),:)); %za datu urnu generisem stanja na osnovu matrice A
end
O(T)=Generator(B(Q(T),:));
figure();
subplot(2,1,1);
stem(1:length(Q),Q);
title('Generisana sekvenca stanja i opservacija za drugi slucaj');
xlabel('Vreme');
ylabel('Stanja urni');
subplot(2,1,2);
stem(1:length(O),O);
xlabel('Vreme');
ylabel('Vrednosti opservacije');
% Problem 1

alfa=zeros(T,N); % forward koeficijenti-alfa(t, i) je varovatnoca da je u trenutku t aktivno stanje Si

% Inicijalizacija

for i=1:N
    alfa(1,i)=Pi(i)*B(i, O(1));
end

% Indukcija

for t=1:T-1
    for j=1:N
        for i=1:N
       alfa(t+1, i)=alfa(t+1, i)+alfa(t,j)*A(j,i)*B(i, O(t+1));
        end
    end
end

% Terminacija

P1(2)=sum(alfa(T,:));

% Problem 2

[Q1(2,:), P2(2,:)]=myViterbi(O,Pi,A,B);

figure();
subplot(2,1,1);
stem(1:length(Q1(1,:)),Q1(1,:));
title('Procenjena sekvenca stanja za prvi i drugi slucaj');
xlabel('Vreme');
ylabel('Stanja urni');
subplot(2,1,2);
stem(1:length(Q1(2,:)),Q1(2,:));
xlabel('Vreme');
ylabel('Vrednosti opservacije');
