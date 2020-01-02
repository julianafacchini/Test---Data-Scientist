# Test---Data-Scientist
Predição de Sexo

Para criar um ambiente isolado para trabalhar com um projeto com as dependências definidas, basta seguir os passos:

1) Criar o novo ambiente:

virtualenv ENV

2) Acessar o diretório do ambiente:

cd ENV

3) Copiar o projeto para o diretório do ambiente, incluindo:
- requirements.txt, 
- test_data_CANDIDATE.csv,
- o arquivo newsample.csv;

4) Ativar o ambiente:

bin/activate

5) Instalar as dependências do projeto:

pip install -r requirements.txt

6) Rodar o arquivo py:

python3 sex_predictor.py 

O resultado estará gravado no arquivo: newsample_PREDICTIONS_{Juliana Facchini de Souza}.csv
