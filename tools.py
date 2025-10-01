import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- Ferramentas Básicas (suas originais melhoradas) ---

def mostrar_tipos_de_dados(df: pd.DataFrame):
    """Exibe os tipos de dados e informações básicas de cada coluna."""
    st.write("### 📊 Tipos de Dados das Colunas")
    
    # Criar dataframe com informações detalhadas
    info_colunas = []
    for col in df.columns:
        info_colunas.append({
            'Coluna': col,
            'Tipo': str(df[col].dtype),
            'Não-Nulos': df[col].count(),
            'Nulos': df[col].isnull().sum(),
            'Únicos': df[col].nunique(),
            'Amostra': str(df[col].iloc[0]) if len(df) > 0 else 'N/A'
        })
    
    info_df = pd.DataFrame(info_colunas)
    st.dataframe(info_df, use_container_width=True)
    
    # Resumo por tipo
    col1, col2, col3 = st.columns(3)
    with col1:
        numericas = len(df.select_dtypes(include=['number']).columns)
        st.metric("Colunas Numéricas", numericas)
    with col2:
        categoricas = len(df.select_dtypes(include=['object']).columns)
        st.metric("Colunas Categóricas", categoricas)  
    with col3:
        st.metric("Total de Colunas", len(df.columns))

def mostrar_estatisticas_descritivas(df: pd.DataFrame):
    """Calcula e exibe estatísticas descritivas completas."""
    st.write("### 📈 Estatísticas Descritivas Completas")
    
    # Estatísticas para colunas numéricas
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        st.write("#### Variáveis Numéricas:")
        st.dataframe(numeric_df.describe(), use_container_width=True)
        
        # Medidas adicionais
        st.write("#### Medidas Adicionais:")
        medidas_extras = pd.DataFrame({
            'Variância': numeric_df.var(),
            'Coef. Variação': (numeric_df.std() / (numeric_df.mean().abs() + 0.0001)) * 100,
            'Assimetria': numeric_df.skew(),
            'Curtose': numeric_df.kurtosis()
        }).round(4)
        st.dataframe(medidas_extras, use_container_width=True)
    
    # Estatísticas para colunas categóricas
    categorical_df = df.select_dtypes(include=['object'])
    if not categorical_df.empty:
        st.write("#### Variáveis Categóricas:")
        cat_stats = pd.DataFrame({
            'Únicos': categorical_df.nunique(),
            'Mais Frequente': categorical_df.mode().iloc[0] if len(categorical_df) > 0 else 'N/A',
            'Frequência': [categorical_df[col].value_counts().iloc[0] if len(categorical_df) > 0 else 0 for col in categorical_df.columns]
        })
        st.dataframe(cat_stats, use_container_width=True)

def gerar_histograma(df: pd.DataFrame, coluna: str):
    """Gera histograma interativo com estatísticas."""
    if coluna not in df.columns:
        st.error(f"Coluna '{coluna}' não encontrada!")
        return
        
    st.write(f"### 📊 Distribuição da Coluna: {coluna}")
    
    if pd.api.types.is_numeric_dtype(df[coluna]):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Histograma interativo
            fig = px.histogram(df, x=coluna, nbins=30, title=f'Distribuição de {coluna}')
            fig.add_vline(x=df[coluna].mean(), line_dash="dash", line_color="red", 
                         annotation_text="Média")
            fig.add_vline(x=df[coluna].median(), line_dash="dash", line_color="green",
                         annotation_text="Mediana")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("#### Estatísticas:")
            st.metric("Média", f"{df[coluna].mean():.2f}")
            st.metric("Mediana", f"{df[coluna].median():.2f}")
            st.metric("Desvio Padrão", f"{df[coluna].std():.2f}")
            st.metric("Min - Max", f"{df[coluna].min():.2f} - {df[coluna].max():.2f}")
            
            # Interpretação
            skewness = df[coluna].skew()
            if abs(skewness) < 0.5:
                distribuicao = "Aproximadamente simétrica"
            elif skewness > 0.5:
                distribuicao = "Assimétrica à direita"
            else:
                distribuicao = "Assimétrica à esquerda"
            st.info(f"**Forma:** {distribuicao}")
    else:
        # Para variáveis categóricas
        fig = px.bar(df[coluna].value_counts().head(10), title=f'Frequência de {coluna}')
        st.plotly_chart(fig, use_container_width=True)

def gerar_mapa_de_calor_correlacao(df: pd.DataFrame):
    """Gera mapa de calor da correlação com análise."""
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        st.error("Não há colunas numéricas para calcular correlação!")
        return
        
    st.write("### 🔥 Mapa de Calor da Correlação")
    
    correlation_matrix = numeric_df.corr()
    
    # Mapa de calor interativo
    fig = px.imshow(correlation_matrix, 
                    text_auto=True, 
                    color_continuous_scale='RdBu',
                    title='Matriz de Correlação')
    st.plotly_chart(fig, use_container_width=True)
    
    # Análise das correlações mais fortes
    st.write("#### 🔍 Correlações Mais Relevantes:")
    
    # Pegar correlações acima de 0.5 (exceto diagonal)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    correlation_matrix_masked = correlation_matrix.mask(mask)
    
    correlations_list = []
    for i in range(len(correlation_matrix_masked.columns)):
        for j in range(len(correlation_matrix_masked.columns)):
            if not pd.isna(correlation_matrix_masked.iloc[i, j]):
                corr_val = correlation_matrix_masked.iloc[i, j]
                if abs(corr_val) > 0.3:  # Threshold para correlações relevantes
                    correlations_list.append({
                        'Variável 1': correlation_matrix_masked.columns[i],
                        'Variável 2': correlation_matrix_masked.index[j], 
                        'Correlação': corr_val,
                        'Força': 'Forte' if abs(corr_val) > 0.7 else 'Moderada' if abs(corr_val) > 0.5 else 'Fraca'
                    })
    
    if correlations_list:
        corr_df = pd.DataFrame(correlations_list).sort_values('Correlação', key=abs, ascending=False)
        st.dataframe(corr_df, use_container_width=True)
    else:
        st.info("Não foram encontradas correlações relevantes (>0.3) entre as variáveis.")

def gerar_boxplot(df: pd.DataFrame, coluna_x: str, coluna_y: str):
    """Gera boxplot comparativo com análises estatísticas."""
    if coluna_x not in df.columns or coluna_y not in df.columns:
        st.error("Uma ou ambas as colunas não foram encontradas!")
        return
        
    st.write(f"### 📦 Boxplot: {coluna_y} por {coluna_x}")
    
    # Boxplot interativo
    fig = px.box(df, x=coluna_x, y=coluna_y, title=f'{coluna_y} por {coluna_x}')
    st.plotly_chart(fig, use_container_width=True)
    
    # Análise estatística por grupo
    st.write("#### 📊 Estatísticas por Grupo:")
    group_stats = df.groupby(coluna_x)[coluna_y].agg(['count', 'mean', 'median', 'std']).round(2)
    st.dataframe(group_stats, use_container_width=True)
    
    # Teste de significância (se aplicável)
    grupos = df.groupby(coluna_x)[coluna_y].apply(list)
    if len(grupos) == 2:
        from scipy.stats import ttest_ind
        stat, p_value = ttest_ind(*grupos.values)
        st.write("#### 🧪 Teste t para Diferença entre Grupos:")
        st.write(f"**Estatística t:** {stat:.4f}")
        st.write(f"**P-valor:** {p_value:.4f}")
        if p_value < 0.05:
            st.success("✅ Diferença estatisticamente significativa (p < 0.05)")
        else:
            st.info("ℹ️ Não há diferença estatisticamente significativa (p ≥ 0.05)")

def encontrar_outliers_zscore(df: pd.DataFrame):
    """Encontra e analisa outliers usando múltiples métodos."""
    st.write("### 🎯 Detecção de Outliers")
    
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        st.error("Não há colunas numéricas para detectar outliers!")
        return
    
    tab1, tab2, tab3 = st.tabs(["Z-Score", "IQR", "Visualização"])
    
    with tab1:
        st.write("#### Método Z-Score (> 3)")
        z_scores = np.abs(stats.zscore(numeric_df.fillna(0)))
        outlier_count_zscore = pd.Series((z_scores > 3).sum(axis=0), index=numeric_df.columns)
        outlier_df_zscore = outlier_count_zscore[outlier_count_zscore > 0].sort_values(ascending=False)
        
        if not outlier_df_zscore.empty:
            st.dataframe(pd.DataFrame({
                'Coluna': outlier_df_zscore.index,
                'Quantidade de Outliers': outlier_df_zscore.values,
                'Percentual': (outlier_df_zscore.values / len(df) * 100).round(2)
            }), use_container_width=True)
        else:
            st.success("✅ Nenhum outlier detectado pelo método Z-Score!")
    
    with tab2:
        st.write("#### Método IQR (Interquartile Range)")
        outliers_iqr = {}
        for col in numeric_df.columns:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = numeric_df[(numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)][col]
            if len(outliers) > 0:
                outliers_iqr[col] = len(outliers)
        
        if outliers_iqr:
            outliers_iqr_df = pd.DataFrame({
                'Coluna': list(outliers_iqr.keys()),
                'Quantidade de Outliers': list(outliers_iqr.values()),
                'Percentual': [v/len(df)*100 for v in outliers_iqr.values()]
            }).round(2)
            st.dataframe(outliers_iqr_df, use_container_width=True)
        else:
            st.success("✅ Nenhum outlier detectado pelo método IQR!")
    
    with tab3:
        st.write("#### Visualização de Outliers")
        col_selecionada = st.selectbox("Selecione uma coluna para visualizar:", numeric_df.columns)
        if col_selecionada:
            col1, col2 = st.columns(2)
            with col1:
                fig_box = px.box(df, y=col_selecionada, title=f'Boxplot - {col_selecionada}')
                st.plotly_chart(fig_box, use_container_width=True)
            with col2:
                fig_hist = px.histogram(df, x=col_selecionada, title=f'Histograma - {col_selecionada}')
                st.plotly_chart(fig_hist, use_container_width=True)