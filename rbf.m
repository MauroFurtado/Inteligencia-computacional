clear;
clc;
close
dados = load('two_classes.dat');
x = dados(:, 1:2);
y = dados(:, 3);
N = size(x, 1);  % Número de amostras
p = 2;  % Dimensionalidade dos dados (correspondente ao número de características em x)
q = 10;  % Número de centros

X = x';  % Transpor para que as colunas representem amostras
D = y';  % Transpor para alinhar as dimensões com X
C = randn(p, q);  % Inicializar aleatoriamente os centros com a dimensionalidade correta
sigma = 1;  % Parâmetro sigma da RBF
Z = zeros(q, N);  % Inicializar a matriz Z

% Calcular os valores da RBF
for i = 1:N
    for j = 1:q
        u = norm(X(:, i) - C(:, j));
        fu = exp((-u^2) / (2 * sigma^2));
        Z(j, i) = fu;
    end
end

% Adicionar o termo do Bias em Z
Z = [(-1) * ones(1, N); Z];

% Calcular a matriz de pesos M
M = (D * Z') / (Z * Z');

% Realizar as previsões
acerto = 0;
for i = 1:N
    % Previsão baseada no sinal de M * Z(:, i)
    if M * Z(:, i) > 0
        previsao = 1;
    else
        previsao = -1;
    end

    desejado = D(i);  % O valor desejado da classe

    % Verifica se a previsão está correta
    if previsao == desejado
        acerto = acerto + 1;
    end
end

% Calcular a taxa de acerto
tx = acerto / N;
disp(['Taxa de acerto: ', num2str(tx * 100), '%']);

% Criar uma grade de pontos no espaço das características
[x1Grid, x2Grid] = meshgrid(linspace(min(x(:,1)), max(x(:,1)), 100), linspace(min(x(:,2)), max(x(:,2)), 100));

% Inicializar a matriz para as previsões da superfície de decisão
ZGrid = zeros(size(x1Grid));

% Fazer previsões para cada ponto na grade
for i = 1:numel(x1Grid)
    ponto = [x1Grid(i); x2Grid(i)];
    Ztemp = zeros(q, 1);

    for j = 1:q
        u = norm(ponto - C(:, j));
        fu = exp((-u^2) / (2 * sigma^2));
        Ztemp(j) = fu;
    end

    Ztemp = [-1; Ztemp];  % Adicionar o termo de viés
    if M * Ztemp > 0
        ZGrid(i) = 1;
    else
        ZGrid(i) = -1;
    end
end

% Plotar os dados e a superfície de decisão
figure;
contourf(x1Grid, x2Grid, ZGrid, 'LineColor', 'none');  % Plotar a superfície de decisão
hold on;
scatter(x(y == 1, 1), x(y == 1, 2), 'r', 'filled');  % Plotar os dados da classe 1
scatter(x(y == -1, 1), x(y == -1, 2), 'b', 'filled');  % Plotar os dados da classe -1
title('Superfície de Decisão');
xlabel('x1');
ylabel('x2');
legend('Superfície de Decisão', 'Classe 1', 'Classe -1');
hold off;

