clc;
clear;

% Solicitar ao usuário a quantidade de vetores de entrada
n = input('Digite a quantidade de vetores de entrada: ');

% Inicializar os vetores de entrada e rótulos
inputs = zeros(n, 2);
outputs = zeros(n, 1);

% Solicitar ao usuário os vetores de entrada e rótulos
for i = 1:n
    fprintf('Digite o vetor de entrada %d (formato: [x1 x2]): ', i);
    inputs(i, :) = input('');
    fprintf('Digite o rótulo para o vetor de entrada %d (0 ou 1): ', i);
    outputs(i) = input('');
end

% Inicializar os parâmetros
input_size = 2; % Número de entradas
W = rand(1, input_size + 1); % Pesos aleatórios (incluindo bias)
bias = -1;

% Treinamento do Perceptron
learning_rate = 0.1;
epocas = 100;

for epoch = 1:epocas
    for i = 1:size(inputs, 1)
        x = [inputs(i, :) 1]; % Adiciona o bias como entrada adicional
        u = x * W'; % Produto escalar
        y = u > 0; % Função de ativação
        erro = outputs(i) - y;
        W = W + learning_rate * erro * x; % Atualização dos pesos
    end
end

% Exibir os W e bias finais
disp('Pesos finais:');
disp(W);
disp('Bias final:');
disp(bias);

% Plotar os pontos e a reta de decisão
figure;
hold on;

% Plotar os pontos com cores diferentes para cada classe
for i = 1:size(inputs, 1)
    if outputs(i) == 0
        plot(inputs(i, 1), inputs(i, 2), 'ro'); % Classe 0 em vermelho
    else
        plot(inputs(i, 1), inputs(i, 2), 'bo'); % Classe 1 em azul
    end
end

% Plotar a reta de decisão
x_values = linspace(min(inputs(:, 1)), max(inputs(:, 1)), 100);
y_values = -(W(1) * x_values + W(3)) / W(2);
plot(x_values, y_values, 'k-'); % Reta de decisão em preto

xlabel('x1');
ylabel('x2');
title('Perceptron - Classificação');
legend('Classe 0', 'Classe 1', 'Reta de Decisão');
hold off;

