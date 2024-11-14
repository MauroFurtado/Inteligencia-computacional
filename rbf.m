
dados = load('two_classes.dat');
x = dados(:,1:2);
y = dados(:,3);
N = length(x);
p = 2;
q = 10;
X = x';D = y';
C = randn(p, q);
sigma = 1;
Z = zeros(q, N);
X=zscore(X)

for i = 1:N
    for j = 1:q
        u = norm(X(:,i) - C(:,j));
        fu = exp((-u^2) / (2 * sigma^2));
        Z(j, i) = fu;
    end
end

Z = [(-1) * ones(1, N); Z];
M = (D * Z')/(Z * Z');

acerto=0;
for i=1:N
  [~, previsao]=max(M*Z(i));

  if previsao==D(i)
    acerto=acerto+1;
  endif
end
tx=acerto/N;
disp(['Taxa de acerto:',num2str(tx*100),'%'])



