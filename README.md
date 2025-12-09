## Setup

Para utilizar o código deste repositório, siga as instruções a seguir:

Crie um ambiente virtual do Python:

``` shell
python3 -m venv env
```

Ative o ambiente virtual (**você deve fazer isso sempre que for executar algum script deste repositório**):

``` shell
source ./env/bin/activate
```

Instale as dependências com:

``` shell
python3 -m pip install -r requirements.txt --upgrade
```

## Deployment

O material utiliza o [mkdocs](https://www.mkdocs.org/) para gerar a documentação. Para visualizar a documentação, execute o comando:

``` shell
mkdocs serve -o
```

Para subir ao GitHub Pages, execute o comando:

``` shell
mkdocs gh-deploy
```


## Notebooks

Para subir notebooks no mkdocs, podemos utilizar a biblioteca do [mkdocs-jupyter](https://github.com/danielfrg/mkdocs-jupyter).

Instalação, utilização e exemplos podem ser vistos na [documentação oficial](https://github.com/danielfrg/mkdocs-jupyter).

O arquivo `mkdocs.yml` tem anotações nos nós modificados de exemplo.


```bash

# 1️⃣ Cria a rede (tudo certo aqui)
docker network create martech

# 2️⃣ Sobe o PostgreSQL
docker run -d \
  --name raiz \
  --network martech \
  -p 5432:5432 \
  -e POSTGRES_USER=root \
  -e POSTGRES_PASSWORD=root \
  -e POSTGRES_DB=raiz \
  -v pgdata:/var/lib/postgresql/data \
  -v /home/andre/Documentos/projects/machine_learning_espm/data:/data \
  postgres:15

# 3️⃣ Sobe o Metabase
docker run -d \
  --name metabase \
  --network martech \
  -p 3000:3000 \
  -e MB_DB_TYPE=postgres \
  -e MB_DB_DBNAME=raiz \
  -e MB_DB_PORT=5432 \
  -e MB_DB_USER=root \
  -e MB_DB_PASS=root \
  -e MB_DB_HOST=raiz \
  -v metabase-data:/metabase-data \
  metabase/metabase

```

https://www.kaggle.com/competitions/home-credit-default-risk/data

extraia os arquivos para data/home-credit-deafult-risk/

docker exec -it raiz psql -U root -d raiz
(senha: root)


 charge_postgres("data/home-credit-default-risk/application_train.csv", "application_train")