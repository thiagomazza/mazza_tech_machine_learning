# Importando bibliotecas
import pandas               as pd
import streamlit            as st
import numpy                as np
import pandas               as pd
import matplotlib.pyplot    as plt
from PIL                    import Image
from sklearn.neighbors      import KNeighborsClassifier
from sklearn.tree           import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble       import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model   import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.metrics        import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error, mean_absolute_error, silhouette_score
from sklearn.preprocessing  import PolynomialFeatures
from math                   import sqrt
from sklearn.cluster        import KMeans, AffinityPropagation

# Carregando DataFrames
results_classification = pd.read_csv('Classificacao/resultados_classificacao.csv')
results_regression     = pd.read_csv('Regressao/resultados_regressao.csv')
results_cluster        = pd.read_csv('Clusterizacao/resultados_clusterizacao.csv')

###################################################################
###################### Montando Streamlit #########################
###################################################################

# Função para criar a barra lateral
image = Image.open("img/Logo_2.png")
col1, col2 = st.sidebar.columns([1, 2], gap="small")
col1.image(image, width=100)
col2.markdown("# Machine Learning")
st.sidebar.markdown(
    f"""
    <div style="text-align: justify; font-size: 18px;">
    <br>Como Cientista de Dados recém-contratado pela Data Money, meu foco é conduzir três ensaios abrangendo algoritmos de Classificação, Regressão e Clusterização.<br><br>
    Essa pesquisa visa aprofundar nosso entendimento sobre o desempenho desses algoritmos em diferentes cenários, capacitando-nos a oferecer soluções mais precisas e financeiramente vantajosas para nossos clientes.<br>
    </div>
    """,
    unsafe_allow_html=True
)

# Montando as Tabs
tab1, tab2 = st.tabs(["O Problema", "O Resultado"])

with tab1:
    st.markdown(
        f"""
    <div style="text-align: justify; font-size: 32px;">
	    1. A empresa Data Money
    </div><br>
    <div style="text-align: justify; font-size: 16px;">
        A empresa Data Money fornece serviços de consultoria de Análise e Ciência de Dados para grandes
        empresas no Brasil e no exterior.
        O seu principal diferencial de mercado em relação aos concorrentes é o alto retorno financeiro para as
        empresas clientes, graças a performance de seus algoritmos de Machine Learning.
        A Data Money acredita que a expertise no treinamento e ajuste fino dos algoritmos, feito pelos Cientistas de
        Dados da empresa, é a principal motivo dos ótimos resultados que as consultorias vem entregando aos seus
        clientes.
        Para continuar crescendo a expertise do time, os Cientistas de Dados acreditam que é extremamente
        importante realizar ensaios nos algoritmos de Machine Learning para adquirir uma experiência cada vez
        maior sobre o seu funcionamento e em quais cenários as performances são máximas e mínimas, para que a
        escolha do algoritmo para cada situação seja a mais correta possível.
        Como Cientista de Dados recém contratado pela empresa, a sua principal tarefa será realizar 3 ensaios com
        algoritmos de Classificação, Regressão e Clusterização, a fim de extrair aprendizados sobre o seu
        funcionamento em determinados cenário e conseguir transmitir esse conhecimento para o restante do time.
   </div><br>
   <div style="text-align: justify; font-size: 32px;">
	    2. O Ensaio de Machine Learning
   </div><br>
    <div style="text-align: justify; font-size: 26px;">
	    2.1. Descrição do Ensaio
    </div>
    <div style="text-align: justify; font-size: 16px;">
        O ensaio de Machine Learning ajuda os Cientistas de Dados a ganhar mais experiência na aplicação dos
        algoritmos. Nesse ensaio, em específico, cada algoritmo será treinado com os dados de treinamento e
        sua performance será medida usando 3 conjuntos de dados:<br><br>
        1. Os próprios dados de treinamento<br>
        2. Os dados de validação<br>
        3. Os dados de teste.<br><br>
    </div>
    <div style="text-align: justify; font-size: 16px;">
        A performance de cada algoritmo será medida, utilizando diferentes métricas de performance.
        O seu trabalho nesse ensaio será construir uma tabela mostrando os valores das métricas de
        performance para cada algoritmo de Machine Learning.
        Cada tabela vai armazenar os resultados da performance sobre um conjunto de dados diferentes, ou
        seja, você precisa criar 3 tabelas:<br><br>
    </div>
    <div style="text-align: justify; font-size: 16px;">
        1) Performance sobre os dados de treinamento<br>
        2) Performance sobre os dados de validação<br>
        3) Performance sobre os dados de teste para o Ensaio de classificação, regressão e clusterização;<br><br>
    </div>
    <div style="text-align: justify; font-size: 26px;">
        2.2. Os algoritmos e métricas do ensaio<br>
    </div>
    <div style="text-align: justify; font-size: 16px;">
        2.2.1 - Classificação:<br>
        Algoritmos: KNN, Decision Tree, Random Forest e Logistic Regression<br>
        Métricas de performance: Accuracy, Precision, Recall e F1-Score<br>
        2.2.2 - Regressão:<br>
        Algoritmos: Linear Regression, Decision Tree Regressor, Random Forest Regressor, Polinomial<br>
        Regression, Linear Regression Lasso, Linear Regression Ridge, Linear Regression Elastic Net,<br>
        Polinomial Regression Lasso, Polinomial Regression Ridge e Polinomial Regression Elastic Net<br>
        Métricas de performance: R2, MSE, RMSE, MAE e MAPE<br>
        2.2.3 - Agrupamento:<br>
        Algoritmos: K-Means e Affinity Propagation<br>
        Métricas de performance: Silhouette Score<br><br>
    </div>
    <div style="text-align: justify; font-size: 26px;">
        2.3. Os parâmetros do ensaio
    </div>
    <div style="text-align: justify; font-size: 16px;">
        2.3.1 - Classificação<br>
        Random Forest Classifier: n_estimators, max_depth<br>
        K-Neighbors Classifier: n_neighbors<br>
        Logistic Regression C, Solver, max_iter<br>
        Decision Tree Classifier: max_depth<br>
        2.3.2 - Regressão<br>
        Decision Tree Regressor: max_depth<br>
        Polinomial Regression: degree<br>
        Polinomial Regression Lasso e Ridge: degree, alpha, max_iter<br>
        Polinomial Regression Elastic Net: degree, alpha, l1_ratio, max_iter<br>
        Random Forest Regressor: n_estimators, max_depth<br>
        Linear Regression Lasso e Ridge: alpha, max_iter, Linear<br>
        Regression Elastic Net: alpha, l1_ratio, max_iter<br>
        2.3.3 - Agrupamento<br>
        K-Means: k<br>
        Affinity Propagation: preference<br>
    </div>
        """,
        unsafe_allow_html=True
    )

with tab2:
    # Exibir classificação
    st.markdown(f"""<div style="text-align: justify; font-size: 24px;"> Métricas alcançadas pelo algoritmo de Classificação</div>""",unsafe_allow_html=True)
    st.dataframe(results_classification)
    # Exibir Regressão
    st.markdown(f"""<div style="text-align: justify; font-size: 24px;"> Métricas alcançadas pelo algoritmo de Regressão</div>""",unsafe_allow_html=True)
    st.dataframe(results_regression)
    # Exibir Clusterização
    st.markdown(f"""<div style="text-align: justify; font-size: 24px;"> Métricas alcançadas pelo algoritmo de Clusterização</div>""",unsafe_allow_html=True)
    st.dataframe(results_cluster)
    
    