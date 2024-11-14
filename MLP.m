clc;
clear;
pkg load nnet;

% Carregar os dados, incluindo a coluna de texto
fid = fopen('column_3C.dat', 'r');
dados_brutos = textscan(fid, '%f %f %f %f %f %f %s', 'Delimiter', ' ');
fclose(fid);

% Extrair as colunas numéricas e a coluna categórica
dados_numericos = cell2mat(dados_brutos(1:6));  % Colunas 1 a 6
rotulos = dados_brutos{7};  % Recebe a coluna 7 de categorias

% Converter as classes para uma matriz de classificação
n = length(rotulos);
matriz_classificacao = zeros(n, 3);

for i = 1:n
    if strcmp(rotulos{i}, 'DH')
        matriz_classificacao(i, :) = [1 0 0];
    elseif strcmp(rotulos{i}, 'SL')
        matriz_classificacao(i, :) = [0 1 0];
    elseif strcmp(rotulos{i}, 'NO')
        matriz_classificacao(i, :) = [0 0 1];
    end
end

% Definir os dados de entrada e saída
entradas = dados_numericos;
saidas = matriz_classificacao;

% Definir o número de execuções
num_execucoes = 10;
acuracias = zeros(1, num_execucoes);



% Executar o processo 10 vezes
for exec = 1:num_execucoes
    % Embaralhar os dados
    indices_aleatorios = randperm(n);
    entradas_embaralhadas = entradas(indices_aleatorios, :);
    saidas_embaralhadas = saidas(indices_aleatorios, :);

    % Separar 70% para treino e 30% para teste
    tamanho_treino = round(0.7 * n);
    entradas_treino = entradas_embaralhadas(1:tamanho_treino, :);
    saidas_treino = saidas_embaralhadas(1:tamanho_treino, :);
    entradas_teste = entradas_embaralhadas(tamanho_treino+1:end, :);
    saidas_teste = saidas_embaralhadas(tamanho_treino+1:end, :);

    % Definir os intervalos de entrada (mínimo e máximo para cada dimensão)
    intervalos_entrada = [min(entradas); max(entradas)]';

    % Definir a arquitetura da rede neural
    num_neuronios = [6, 3];  % 6 neurônios na camada oculta e 3 na camada de saída
    funcoes_ativacao = {"tansig", "purelin"};  % Funções de ativação para as camadas oculta e de saída
    algoritmo_treino = "trainlm";  % Algoritmo de treino
    funcao_aprendizado = "learngdm";  % Função de aprendizado
    funcao_erro = "mse";  % Função de erro

    % Criar a rede MLP
    rede_MLP = newff(intervalos_entrada, num_neuronios, funcoes_ativacao, algoritmo_treino, funcao_aprendizado, funcao_erro);
    rede_MLP.trainParam.epochs = 100;  % Número de épocas
    rede_MLP.trainParam.goal = 0.01;  % Meta para o erro
    rede_MLP.trainParam.lr = 0.1;  % Taxa de aprendizado

    % Treinar a rede neural
    rede_MLP = train(rede_MLP, entradas_treino', saidas_treino');

    % Fazer previsões usando o conjunto de teste
    saidas_previstas = sim(rede_MLP, entradas_teste');

    % Converter as saídas para rótulos
    [~, rotulos_previstos] = max(saidas_previstas, [], 1);
    [~, rotulos_reais] = max(saidas_teste, [], 2);

    % Calcular a acurácia
    acuracia_atual = sum(rotulos_previstos' == rotulos_reais) / length(rotulos_reais);
    acuracias(exec) = acuracia_atual;
end

% Calcular a acurácia média
acuracia_media = mean(acuracias);
fprintf('Acuracia media apos 10 execucoes: %.2f%%\n', acuracia_media * 100);
