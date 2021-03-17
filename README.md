# Aplicação de Machine Learning para setor de Recursos Humanos.

## Desgaste e desempenho de funcionário do IBM HR Analytics
## Previsão do desgaste de seus valiosos funcionários

Descubra os fatores que levam ao desgaste de funcionários e como consequentemente a sair da empresa que trabalha. Este é um conjunto de dados fictício criado por cientistas de dados da IBM com 1470 registros.


Informações:
* Idade
* Desgaste
* Viagem a negócio
* Salário diário
* Departamento
* Distância de casa do trabalho
* Educação
  * 1 Ensino Médio
  * 2 Graduado
  * 3 Pós Graduado
  * 4 Mestre
  * 5 Doutor
* Área de formação
* Contagem Funcionário
* Matricula do funcionário
* Satisfeito com ambiente
  * 1 Baixo
  * 2 Médio
  * 3 Alto
  * 4 Muito Alto
* Genero
* Horas trabalhadas
* Envolvimento com trabalho
  * 1 Baixo
  * 2 Médio
  * 3 Alto
  * 4 Muito Alto
* Nível do emprego
* Cargo
* Satisfeito com o trabalho
  * 1 Baixo
  * 2 Médio
  * 3 Alto
  * 4 Muito Alto
* Estado Civil
* Renda mensal
* Taxa de salario mensal
* Nº de empresas que trabalhou
* Mais de 18 anos
* Horas Extra
* Aumento de salário percentual
* Avaliação de desempenho
  * 1 Baixo
  * 2 Bom
  * 3 Excelente
  * 4 Excepcional
* Satisfeito com relacionamento no trabalho
  * 1 Baixo
  * 2 Médio
  * 3 Alto
  * 4 Muito Alto
* Carga horária
* Nível de ações na empresa
* Tempo de registro em carteira
* Tempo de treinamento no ano passado
* Equilibrio entre o trabalho e vida pessoal
  * 1 Ruim
  * 2 Bom
  * 3 Melhor
  * 4 Melhor
* Tempo na empresa atual
* Ano desde da ultima promoção
* Anos com o mesmo gerente


https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset


Desenvolvi uma notebook com as análise exploratória, transformação dos dados, normalização dos dados, aplicação de técnicas para balanceamento de classe minoritária, treinamento e validação do modelo. Foi aplicado 3 modelos de Machine Learning tendo se destacado o modelo XGBClassifier.
A partir desse notebook, implementei o modelo treinado em produção onde ficará disponível para os gestores do departamento de RH coletar previsões e probabilidade dos registros imputados. 

Deploy da aplicação: http://rh-ml.herokuapp.com/

Essa aplicação usamos os parâmetros padrão do modelo, mas para melhorar os resultados poderiamos testar:

* Usar outros modelos de classificação;

* Usar técnicas de hiperparamentro para melhorar o modelo;

* Combinação de modelos;
