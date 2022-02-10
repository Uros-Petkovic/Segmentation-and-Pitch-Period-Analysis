
function stanje=Generator(Pi)

pom=rand(1,1);
if (pom<=Pi(1))
    stanje=1;
end
if (Pi(1)<pom && pom<=Pi(1)+Pi(2))
    stanje=2;
end
if (Pi(1)+Pi(2)<pom && pom<=1)
    stanje=3;
end
end