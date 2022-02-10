

%% Prvi zadatak 3 - Izdvajanje Pitch periode

clc; clear all;

[y,fs]=audioread('SekvencaPitch1.wav');
T=1/fs;
y=y(5501:71000);
n=length(y);
figure(1);
t=T:T:length(y)*T;
plot(t,y);
title('Ulazni govorni signal');
axis([0 n*T min(y) max(y)]);
hold on
N=length(y);
xc=y;

[B,A]=butter(6,300/(fs/2)); % jer je Fpitch manje od 300Hz
xcf=filter(B,A,xc);
plot(t,xcf);
title('Ulazni i filtrirani signal');
legend('Ulazni signal','Filtrirani signal');

%Formiranje povorki impulsa

m1=zeros(1,N);
m2=zeros(1,N);
m3=zeros(1,N);
m4=zeros(1,N);
m5=zeros(1,N);
m6=zeros(1,N);
m4pom=zeros(1,N);
maksp=[];
minp=[];

%Formiranje m1 i m4     nalazimo maksimume i minimume
for i=2:N-1
    if xcf(i)>xcf(i-1) && xcf(i)>xcf(i+1)
        m1(i)=max(0,xcf(i));
        maksp=[maksp i];
    end
    if xcf(i)<xcf(i-1) && xcf(i)<xcf(i+1)
        m4pom(i)=xcf(i);
        m4(i)=max(0,-xcf(i));   %minus potice od toga sto trazimo aps. vrednost
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


figure(2);
subplot(7,1,1);plot(xcf(1200:4800));
title('Procena Pitch periode za jedan segment');
axis([0 length(xcf(1200:4800)) -0.1 0.1 ]);
subplot(7,1,2);plot(m1(1200:4800));
axis([0 length(xcf(1200:4800)) 0 0.1 ]);
subplot(7,1,3);plot(m2(1200:4800));
axis([0 length(xcf(1200:4800)) 0 0.2 ]);
subplot(7,1,4);plot(m3(1200:4800));
axis([0 length(xcf(1200:4800)) 0 0.1 ]);
subplot(7,1,5);plot(m4(1200:4800));
axis([0 length(xcf(1200:4800)) 0 0.1 ]);
subplot(7,1,6);plot(m5(1200:4800));
axis([0 length(xcf(1200:4800)) 0 0.2 ]);
subplot(7,1,7);plot(m6(1200:4800));
axis([0 length(xcf(1200:4800)) 0 0.1 ]);


figure(3);
subplot(7,1,1);plot(xcf);
title('Procena Pitch periode za celu sekvencu');
axis([0 length(xcf) -0.1 0.1 ]);
subplot(7,1,2);plot(m1);
axis([0 length(xcf) 0 0.1 ]);
subplot(7,1,3);plot(m2);
axis([0 length(xcf) 0 0.2 ]);
subplot(7,1,4);plot(m3);
axis([0 length(xcf) 0 0.1 ]);
subplot(7,1,5);plot(m4);
axis([0 length(xcf) 0 0.1 ]);
subplot(7,1,6);plot(m5);
axis([0 length(xcf) 0 0.2 ]);
subplot(7,1,7);plot(m6);
axis([0 length(xcf) 0 0.1 ]);

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

figure(4); 
subplot(7,1,1); plot(1./(E1/fs));
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

%Filtriranje
for i=4:length(E)-3 
    Ek(i)=median(E(i-3:i+3));
end
Estimacija=1./(Ek/fs);
figure(5); plot(Estimacija);
axis([0 length(1./([0 E(1:end-1)]/fs)) 80 220]);
disp(['Procenjena pitch perioda je: ', num2str(median(Estimacija(4:end)))]);

% Dobijena pitch perioda glasa je 141.1294 Hz
