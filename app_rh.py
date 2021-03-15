import streamlit as st 
import pandas as pd
import pickle
from xgboost import Booster
from xgboost import XGBClassifier
from PIL import Image
import numpy as np
import base64


st.set_page_config(page_title='Projeto Machine Learning - Recursos Humanos', layout='wide', )

image = Image.open('logo3.jpg')

st.image(image, use_column_width=True)
st.write("""
# Aplicação de Machine Learning



Implementação do modelo de Machine Learning **XGBoost Classifier** para identicar se o colaborador sairá da empresa.




""")
        
st.sidebar.header("""**Parâmentros**""")
st.sidebar.markdown("""
[Download do arquivo modelo](https://github.com/renatopaschoalim/rh_ml_heroku/raw/main/Planilha_Modelo_RH.xls)
""")
uploader_file = st.sidebar.file_uploader("Carregar base", type=['xls'], accept_multiple_files=False)
if uploader_file is not None:
    input_df = pd.read_excel(uploader_file)

else:
    def user_input_features():
        Idade = st.sidebar.slider('Idade', 18, 70, 30)
        Viagem_Negocio = st.sidebar.selectbox('Viagem a Negócio',('Não Viaja','Raramente', 'Frequentemente'))
        Valor_Diario = st.sidebar.slider('Valor Diário', 0, 1500, 750)
        Departamento = st.sidebar.selectbox('Departamento', ('Vendas', 'Pesquisa & Desenvolvimento', 'Recursos Humanos'))
        Distancia_Casa = st.sidebar.slider('Distância de Casa', 1, 60, 30)
        Educacao = st.sidebar.selectbox('Grau Escolaridade' ,('Ensino Médio', 'Graduado', 'Pós Graduado', 'Mestrado', 'Doutorado'))
        Area_Formacao = st.sidebar.selectbox('Área de Formação', ('Ciências da Vida', 'Outros', 'Medicina', 'Marketing','Grau Técnico', 'Recursos Humanos'))
        Satisfeito_Ambiente = st.sidebar.selectbox('Satisfação com o Ambiente', ('Baixo', 'Médio', 'Alto', 'Muito Alto'))
        Genero = st.sidebar.selectbox('Genero', ('Feminino', 'Masculino'))
        Horas_Trabalhadas = st.sidebar.slider('Horas Trabalhadas', 10, 150, 83)
        Envolvimento_Trabalho = st.sidebar.selectbox('Envolvimeno no Trabalho', ('Baixo', 'Médio', 'Alto', 'Muito Alto'))
        Nivel_Emprego = st.sidebar.slider('Nível de Emprego', 1, 5, 2)
        Cargo = st.sidebar.selectbox('Cargo', ('Executivo de vendas', 'Cientista de pesquisa', 'Técnico de laboratório', 'Diretor de Fabricação', 'Representante de Saúde', 'Gerente', 'Representante de Vendas', 'Diretor de Pesquisa', 'Recursos Humanos'))
        Satisfeito_Trabalho = st.sidebar.selectbox('Grau Satisfação com Trabalho', ('Baixo', 'Médio', 'Alto', 'Muito Alto'))
        Estado_Civil = st.sidebar.selectbox('Estado Civil', ('Solteiro', 'Casado', 'Divorciado'))
        Renda_Mensal = st.sidebar.slider('Renda Mensal', 100.00, 50000.00, 4500.00)
        Taxa_Mensal = st.sidebar.slider('Taxa Mensal', 2.0, 27.0, 15.0)
        Num_Empresa_Trabalhou = st.sidebar.slider('Nº de Empresa que já trabalhou', 1, 20, 6)
        Hora_Extra = st.sidebar.selectbox('Faz Hora Extra?', ('Não', 'Sim'))
        Aumento_Percentual_Salar = st.sidebar.slider('Percentual de aumento de Salário ', 0.0, 100.0, 15.0)
        Avaliacao_Desempenho = st.sidebar.selectbox('Avaliação de Desempenho', ('Baixo', 'Bom', 'Excelente', 'Excepcional'))
        Satisfacao_Relacionamento = st.sidebar.selectbox('Satisfação do relacionamento no trabalho', ('Baixo', 'Médio', 'Alto', 'Muito Alto'))
        Nivel_Acoes_Empresa = st.sidebar.slider('Nível de ações da empresa', 0, 5, 0)
        Tempo_De_Registro = st.sidebar.slider('Tempo de registro em carteira', 0, 35, 3)
        Tempo_Treinamento_Ano_Passado = st.sidebar.slider('Tempo de treinamento no ano anterior', 0, 20, 0)
        Equilibrio_Trab_Vida_Pess = st.sidebar.selectbox('Equilíbrio entre o trabalho e vida pessoal', ('Ruim', 'Bom', 'Ótimo', 'Excelente'))
        Tempo_Na_Empresa = st.sidebar.slider('Tempo de registro na empresa atual', 0, 35, 3)
        Anos_Funcao_Atual = st.sidebar.slider('Anos na função atual', 0, 35, 3)
        Anos_Desde_Ultim_Promo = st.sidebar.slider('Anos desde da última promoção', 0, 35, 3)
        Anos_Com_Mesmo_Gerente = st.sidebar.slider('Anos com o mesmo gerente', 0, 35, 3)

        data = {
        'Idade': Idade,
        'Viagem_Negocio': Viagem_Negocio,
        'Valor_Diario':Valor_Diario,
        'Departamento':Departamento,
        'Distancia_Casa':Distancia_Casa,
        'Educacao':Educacao,
        'Area_Formacao':Area_Formacao,
        'Satisfeito_Ambiente':Satisfeito_Ambiente,
        'Genero':Genero,
        'Horas_Trabalhadas':Horas_Trabalhadas,
        'Envolvimento_Trabalho':Envolvimento_Trabalho,
        'Nivel_Emprego':Nivel_Emprego,
        'Cargo':Cargo,
        'Satisfeito_Trabalho':Satisfeito_Trabalho,
        'Estado_Civil':Estado_Civil,
        'Renda_Mensal':Renda_Mensal,
        'Taxa_Mensal':Taxa_Mensal,
        'Num_Empresa_Trabalhou': Num_Empresa_Trabalhou,
        'Hora_Extra': Hora_Extra,
        'Aumento_Percentual_Salar':Aumento_Percentual_Salar,
        'Avaliacao_Desempenho': Avaliacao_Desempenho,
        'Satisfacao_Relacionamento': Satisfacao_Relacionamento,
        'Nivel_Acoes_Empresa': Nivel_Acoes_Empresa,
        'Tempo_De_Registro': Tempo_De_Registro,
        'Tempo_Treinamento_Ano_Passado':Tempo_Treinamento_Ano_Passado,
        'Equilibrio_Trab_Vida_Pess': Equilibrio_Trab_Vida_Pess,
        'Tempo_Na_Empresa': Tempo_Na_Empresa,
        'Anos_Funcao_Atual': Anos_Funcao_Atual,
        'Anos_Desde_Ultim_Promo': Anos_Desde_Ultim_Promo,
        'Anos_Com_Mesmo_Gerente': Anos_Com_Mesmo_Gerente
         }

        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()
    
st.header('**Visualização dos dados.**')
st.write(input_df)

input_df['Hora_Extra'] = input_df['Hora_Extra'].apply(lambda x: 1 if x == 'Yes' else 0)

input_df = input_df.replace(['Ensino Médio', 'Graduado', 'Pós Graduado', 'Mestrado', 'Doutorado'],[1, 2, 3, 4, 5])
input_df = input_df.replace(['Baixo', 'Médio', 'Alto', 'Muito Alto'],[1, 2, 3, 4])
input_df = input_df.replace(['Ruim', 'Bom', 'Ótimo', 'Excelente'],[1, 2, 3, 4])
input_df = input_df.replace(['Ruim', 'Bom', 'Excelente', 'Excepcional'],[1, 2, 3, 4])
input_df = input_df.replace(['Não Viaja','Raramente', 'Frequentemente'], ['Non-Travel','Travel_Rarely', 'Travel_Frequently'])
input_df = input_df.replace(['Vendas', 'Pesquisa & Desenvolvimento', 'Recursos Humanos'], ['Sales', 'Research & Development', 'Human Resources'])
input_df = input_df.replace(['Ciências da Vida', 'Outros', 'Medicina', 'Marketing','Grau Técnico', 'Recursos Humanos'], ['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources'])
input_df = input_df.replace(['Feminino', 'Masculino'], ['Female', 'Male'])
input_df = input_df.replace(['Executivo de vendas', 'Cientista de pesquisa', 'Técnico de laboratório', 'Diretor de Fabricação', 'Representante de Saúde', 'Gerente', 'Representante de Vendas', 'Diretor de Pesquisa', 'Recursos Humanos'], 
                ['Sales Executive', 'Research Scientist', 'Laboratory Technician',
       'Manufacturing Director', 'Healthcare Representative', 'Manager',
       'Sales Representative', 'Research Director', 'Human Resources'])
input_df = input_df.replace(['Solteiro', 'Casado', 'Divorciado'], ['Single', 'Married', 'Divorced'])


input_df_cat = input_df[['Viagem_Negocio', 'Departamento', 'Area_Formacao', 'Genero', 'Cargo', 'Estado_Civil']]
input_df_num = input_df[['Idade', 'Valor_Diario', 'Distancia_Casa', 'Educacao', 'Satisfeito_Ambiente', 'Horas_Trabalhadas', 'Envolvimento_Trabalho', 
                        'Nivel_Emprego', 'Satisfeito_Trabalho', 'Renda_Mensal', 'Taxa_Mensal', 'Num_Empresa_Trabalhou', 'Hora_Extra', 'Aumento_Percentual_Salar',
                        'Avaliacao_Desempenho', 'Satisfacao_Relacionamento', 'Nivel_Acoes_Empresa', 'Tempo_De_Registro', 'Tempo_Treinamento_Ano_Passado',
                        'Equilibrio_Trab_Vida_Pess', 'Tempo_Na_Empresa', 'Anos_Funcao_Atual', 'Anos_Desde_Ultim_Promo', 'Anos_Com_Mesmo_Gerente']]

with open('./precessing_data.pkl', 'rb') as f:
    scaler, onehotencoder = pickle.load(f)

input_df_cat = onehotencoder.transform(input_df_cat).toarray()
input_df_cat = pd.DataFrame(input_df_cat)

input_df_all = pd.concat([input_df_cat, input_df_num], axis=1)

input_df_all = scaler.transform(input_df_all)

    
xgb = XGBClassifier()
booster = Booster()
booster.load_model('./model.dat')
xgb._Booster = booster


pred = xgb.predict(input_df_all)
pred_proba = xgb.predict_proba(input_df_all)
pred_proba = pd.DataFrame(pred_proba, columns=['Não', 'Sim'])
pred = pd.DataFrame(pred, columns=['Previsão'])
      
st.header('***Abaixo mostra a previsão de acordo com os dados informado acima.***')
st.write('''
               
         
         -> Previsão igual a ***0*** indica que o colaborador tem a probalidade baixa de sair da empresa.
         
         
         -> Previsão igual a ***1*** indica que o colaborador tem a probalidade alta de sair da empresa.
         
         
         ''')    
st.write('***Previsão***', pred)
st.write('***Probabilidade em porcentagem %:***', round(pred_proba*100, 2))

df_finally = pd.concat([pred, pred_proba, input_df], axis=1)

df_finally['Viagem_Negocio'] = df_finally['Viagem_Negocio'].replace(['Non-Travel','Travel_Rarely', 'Travel_Frequently'], ['Não Viaja','Raramente', 'Frequentemente'])
df_finally['Departamento'] = df_finally['Departamento'].replace(['Sales', 'Research & Development', 'Human Resources'], ['Vendas', 'Pesquisa & Desenvolvimento', 'Recursos Humanos'])
df_finally['Educacao'] = df_finally['Educacao'].replace([1, 2, 3, 4, 5], ['Ensino Médio', 'Graduado', 'Pós Graduado', 'Mestrado', 'Doutorado'])
df_finally['Area_Formacao'] = df_finally['Area_Formacao'].replace(['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources'], ['Ciências da Vida', 'Outros', 'Medicina', 'Marketing','Grau Técnico', 'Recursos Humanos'])
df_finally['Satisfeito_Ambiente'] = df_finally['Satisfeito_Ambiente'].replace([1, 2, 3, 4],['Baixo', 'Médio', 'Alto', 'Muito Alto'])
df_finally['Equilibrio_Trab_Vida_Pess'] = df_finally['Equilibrio_Trab_Vida_Pess'].replace([1, 2, 3, 4], ['Ruim', 'Bom', 'Ótimo', 'Excelente'])
df_finally['Avaliacao_Desempenho'] = df_finally['Avaliacao_Desempenho'].replace([1, 2, 3, 4], ['Ruim', 'Bom', 'Excelente', 'Excepcional'])
df_finally['Genero'] = df_finally['Genero'].replace(['Female', 'Male'], ['Feminino', 'Masculino'])
df_finally['Cargo'] = df_finally['Cargo'].replace(['Sales Executive', 'Research Scientist', 'Laboratory Technician','Manufacturing Director', 'Healthcare Representative', 'Manager',
       'Sales Representative', 'Research Director', 'Human Resources'], ['Executivo de vendas', 'Cientista de pesquisa', 'Técnico de laboratório', 'Diretor de Fabricação', 'Representante de Saúde', 'Gerente', 'Representante de Vendas', 'Diretor de Pesquisa', 'Recursos Humanos'])
df_finally['Estado_Civil'] = df_finally['Estado_Civil'].replace(['Single', 'Married', 'Divorced'], ['Solteiro', 'Casado', 'Divorciado'])
df_finally['Hora_Extra'] = df_finally['Hora_Extra'].replace([0, 1], ['Não', 'Sim'])
df_finally['Satisfacao_Relacionamento'] = df_finally['Satisfacao_Relacionamento'].replace([1, 2, 3, 4],['Baixo', 'Médio', 'Alto', 'Muito Alto'])
df_finally['Envolvimento_Trabalho'] = df_finally['Envolvimento_Trabalho'].replace([1, 2, 3, 4],['Baixo', 'Médio', 'Alto', 'Muito Alto'])
df_finally['Satisfeito_Trabalho'] = df_finally['Satisfeito_Trabalho'].replace([1, 2, 3, 4],['Baixo', 'Médio', 'Alto', 'Muito Alto'])


st.header('***Visão dos dados com a previsão e probabilidade***')
st.write(df_finally)


def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="previsoes_RH.csv">Download dos dados previstos</a>'
    return href

st.markdown(filedownload(df_finally), unsafe_allow_html=True)



