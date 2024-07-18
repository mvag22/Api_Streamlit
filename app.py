import pandas as pd
import streamlit as st
#from pycaret.regression import *
import joblib 

# loading the trained model.
model = joblib.load('model/modelo-final-LR.pkl')

# carregando uma amostra dos dados.
dataset = pd.read_csv('data/imoveis.csv') 
#classifier = pickle.load(pickle_in)


# título
st.title("Data App - Predição de Valores de Alugueis")

# subtítulo
st.markdown("Este é um Data App utilizado para exibir a solução de Machine Learning para o problema de predição de valores alugueis de imóveis.")



st.sidebar.subheader("Defina os atributos do imóvel para predição do aluguel")


# mapeando dados do usuário para cada atributo
area = st.sidebar.number_input("Área total", value=dataset["area"].mean())
num_quartos = st.sidebar.number_input("Número de Quartos", value=dataset["num_banheiros"].mean())
num_banheiros = st.sidebar.number_input("Número de Banheiros", value=dataset["num_banheiros"].mean())
vagas_garagem = st.sidebar.number_input("Vagas de Garagem", value=dataset["garagem"].mean())
num_andares = st.sidebar.number_input("Número de Andares", value=dataset["num_andares"].mean())
aceita_animais = st.sidebar.selectbox("Aceita Animais?",("Sim","Não"))
mobilia = st.sidebar.selectbox("Mobiliado?",("Sim","Não"))

# transformando o dado de entrada em valor binário
aceita_animais = 1 if aceita_animais == "Sim" else 0
mobilia = 1 if mobilia == "Sim" else 0


estado = st.sidebar.selectbox("Estado",(
                            "SP"
                            ,"RS"
                            ,"RJ"
                            ,"MG"
                            )
                        )


if estado == "SP":
    estado = 1
if estado == "RS":
    estado = 2
if estado == "RJ":
    estado = 3
if estado == "MG":
    estado = 4
    

cidade = st.sidebar.selectbox("Cidade",(
                            "São Paulo"
                            ,"Porto Alegre"
                            ,"Rio De Janeiro"
                            ,"Campinas"
                            ,"Belo Horizonte"
                            )
                        )


if cidade == "São Paulo": 
    cidade = 1
if cidade == "Porto Alegre": 
    cidade = 2
if cidade == "Rio De Janeiro": 
    cidade = 3
if cidade == "Campinas": 
    cidade = 4
if cidade == "Belo Horizonte": 
    cidade = 5
   

valor_condominio = st.sidebar.number_input("Valor do Condomínio", value=dataset["valor_condominio"].mean())
valor_iptu = st.sidebar.number_input("Valor do IPTU", value=dataset["valor_iptu"].mean())
valor_seguro_incendio = st.sidebar.number_input("Valor do Seguro Incêndio", value=dataset["valor_seguro_incendio"].mean())

# inserindo um botão na tela
btn_predict = st.sidebar.button("Realizar Predição")

# verifica se o botão foi acionado
if btn_predict:
    data_teste = pd.DataFrame()

    data_teste["cidade"] =	[cidade]
    data_teste["estado"] =	[estado]    
    data_teste["area"] = [area]
    data_teste["num_quartos"] = [num_quartos]	
    data_teste["num_banheiros"] = [num_banheiros]
    data_teste["garagem"] = [vagas_garagem]
    data_teste["num_andares"] = [num_andares]
    data_teste["aceita_animais"] =	[aceita_animais]
    data_teste["mobilia"] =	[mobilia]
    data_teste["valor_condominio"] = [valor_condominio]
    data_teste["valor_iptu"] = [valor_iptu]
    data_teste["valor_seguro_incendio"] = [valor_seguro_incendio]
    
    #imprime os dados de teste    
    print(data_teste)

    #realiza a predição
    result = model.predict(data_teste)
    
    st.subheader("O valor de aluguel previsto para o imóvel é:")
    result = "R$ "+str(round(result[0],2))
    
    st.write(result)

#no prompt do anaconda, ir no diretorio da api_streamlit : C:\Users\gusta\Meu Drive\Colab Notebooks\DML\api_streamlit>

#rodar o comando: streamlit run app.py

