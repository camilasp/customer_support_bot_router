# Customer Support Bot Router
## Objetivo
<p>O objetivo desse projeto é construir uma aplicação que consiga encaminhar um cliente que entra em contato com o serviço de suporte de uma empresa para o atendimento especializado.</p>

## Dados
A base de dados que utilizada para treinar o modelo está disponível no [Kaggle](https://www.kaggle.com/datasets/536db59649ec509a2808c8d2c85d560c64e1dce44778a22ab79ce3408813e8fb). A base está em inglês e dessa forma só poderá classificar textos nessa língua.
A base contém 20 mil registros/sentenças devidamente classificadas de acordo com o assunto de que tratam. Os assuntos presentes na base são:

* ACCOUNT

* CANCELLATION_FEE

* CONTACT

* DELIVERY

* FEEDBACK

* INVOICES

* NEWSLETTER

* ORDER

* PAYMENT

* REFUNDS

* SHIPPING

## Tratamento e Modelagem
O texto passou por uma etapa de limpeza e adequação de 6 passos e posteriormente foi transformado usando TF-IDF. Em seguida, utilizamos dois classificadores para treinar o modelo.
Como forma de validação dos reusltados, testamos os dois modelos com outra base de dados, que contém frases/questões relacionadas aos temas presentes na base de treinamento mas que foi gerada artificialmente pela ferramenta ChatGPT.

Todos os passos descritos podem ser visualizados na íntegra neste [notebook](https://github.com/camilasp/customer_support_bot_router/blob/main/customer_support_bot.ipynb)
  
  
## Preparação do código e deploy da aplicação
 
A ferramenta Streamlit foi usada para disponibiizar o modelo online.
 
Para isso, tanto o modelo de melhor performance quanto o vetorizador foram salvos em arquivos do tipo .sav. Em seguida, o programa da aplicação, contendo os passos de limpeza e a composição dessas funções com o vetorizador e o modelo, criando um pipeline, foi escrito no arquivo prediction.py. O arquivo app.py por sua vez é o responsável por chamar a função, quando acionado via interface do app.
  
O app pode ser acessado neste [link](https://rodriguesrbruno-customer-support-bot-router-app-prod-bqkgzl.streamlit.app/).

 
  
