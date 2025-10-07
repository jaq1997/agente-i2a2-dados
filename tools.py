import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
from groq import Groq

# --- Caixa de Ferramentas ---
caixa_de_ferramentas = {}

# --- Função auxiliar: Obter informações do DataFrame ---
def obter_info_completa_dataframe(df):
    """Retorna informações completas sobre o DataFrame"""
    colunas_numericas = list(df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns)
    colunas_categoricas = list(df.select_dtypes(include=['object', 'category', 'bool']).columns)
    
    info = {
        "colunas": list(df.columns),
        "tipos": df.dtypes.astype(str).to_dict(),
        "colunas_numericas": colunas_numericas,
        "colunas_categoricas": colunas_categoricas,
        "shape": df.shape,
        "memoria_mb": round(df.memory_usage(deep=True).sum() / (1024*1024), 2),
        "valores_nulos": df.isnull().sum().to_dict(),
        "primeiras_linhas": df.head(3).to_dict('records') if len(df) > 0 else []
    }
    return info

# --- Classificador de Perguntas ---
def classificar_pergunta(pergunta):
    """Classifica o tipo de pergunta para melhor roteamento"""
    pergunta_lower = pergunta.lower()
    
    # Ordem importa! Verificações mais específicas primeiro
    if any(word in pergunta_lower for word in ["conclusão", "conclusao", "conclusões", "conclusoes", "insight", "o que você descobriu", "o que voce descobriu", "descobri", "aprende", "resumo geral", "síntese", "sintese"]):
        return "conclusoes"
    
    # Proporção/distribuição de classes
    if any(phrase in pergunta_lower for phrase in ["proporção", "proporcao", "distribuição da classe", "distribuicao da classe", "balanceamento", "quantas fraudes", "quantos"]):
        return "distribuicao_classes"
    
    # Tipos de dados
    if any(phrase in pergunta_lower for phrase in ["quais colunas", "tipos de dados", "tipos de coluna", "colunas numéricas", "colunas categóricas", "tipo de cada coluna", "estrutura do dataset"]):
        return "tipos_dados"
    
    # Scatter plot (dispersão)
    if any(phrase in pergunta_lower for phrase in ["dispersão", "dispersao", "scatter", "relação entre", "relacao entre", " vs ", " versus "]):
        return "scatter"
    
    # Comparação (boxplot)
    if any(word in pergunta_lower for word in ["compar", "boxplot", " por ", " x ", "diferença", "diferenca", "entre grupos"]):
        return "comparacao"
    
    # Outliers
    if any(word in pergunta_lower for word in ["outlier", "atípico", "atipico", "anomal", "discrepan", "valores extremos"]):
        return "outliers"
    
    # Correlação
    if any(word in pergunta_lower for word in ["correlação", "correlacao", "relacao", "relaciona", "mapa de calor"]):
        return "correlacao"
    
    # Distribuição (histograma)
    if any(word in pergunta_lower for word in ["distribui", "histograma", "frequen"]):
        return "distribuicao"
    
    # Estatísticas
    if any(word in pergunta_lower for word in ["estatística", "estatistica", "média", "media", "mediana", "desvio", "resumo estatístico"]):
        return "estatisticas"
    
    return "geral"

# --- Roteador Inteligente ---
def agente_roteador_inteligente(pergunta_usuario, ferramentas, info_df, tipo_pergunta):
    """Roteador baseado na classificação da pergunta"""
    st.session_state.pensamento_agentes.append(f"Tipo de pergunta identificado: {tipo_pergunta}")
    
    mapeamento = {
        "tipos_dados": "mostrar_tipos_de_dados",
        "estatisticas": "mostrar_estatisticas_descritivas",
        "distribuicao": "gerar_histograma",
        "correlacao": "gerar_mapa_de_calor_correlacao",
        "outliers": "encontrar_outliers_zscore",
        "comparacao": "gerar_boxplot",
        "scatter": "gerar_scatter_plot",
        "distribuicao_classes": "analisar_distribuicao_classes",
        "geral": "mostrar_estatisticas_descritivas"
    }
    
    ferramenta_escolhida = mapeamento.get(tipo_pergunta, "mostrar_estatisticas_descritivas")
    st.session_state.pensamento_agentes.append(f"Ferramenta selecionada: {ferramenta_escolhida}")
    
    return ferramenta_escolhida

# --- Extrator de Parâmetros ---
def agente_extrator_parametros_melhorado(pergunta_usuario, ferramenta, info_df):
    """Extrator de parâmetros mais inteligente"""
    import inspect
    
    funcao = caixa_de_ferramentas[ferramenta]
    assinatura = inspect.signature(funcao)
    parametros_funcao = [p for p in assinatura.parameters.keys() if p != 'df']
    
    if not parametros_funcao:
        st.session_state.pensamento_agentes.append("Função não precisa de parâmetros adicionais")
        return {}
    
    st.session_state.pensamento_agentes.append(f"Buscando parâmetros: {parametros_funcao}")
    
    params_extraidos = {}
    pergunta_lower = pergunta_usuario.lower()
    
    # Para histogramas e análise de classes - buscar nome da coluna
    if "coluna" in parametros_funcao:
        # Procura por nomes de colunas mencionados
        for col in info_df['colunas']:
            if col.lower() in pergunta_lower:
                params_extraidos["coluna"] = col
                break
        
        # Se não encontrou, usa heurística
        if "coluna" not in params_extraidos:
            # Para distribuição de classes, procura coluna categórica
            if ferramenta == "analisar_distribuicao_classes":
                # Procura por 'class', 'classe', 'churn', 'fraud', etc
                for col in info_df['colunas_categoricas']:
                    if any(palavra in col.lower() for palavra in ['class', 'churn', 'fraud', 'target', 'label']):
                        params_extraidos["coluna"] = col
                        break
                
                # Se ainda não achou, usa a última categórica (geralmente é o target)
                if "coluna" not in params_extraidos and info_df['colunas_categoricas']:
                    params_extraidos["coluna"] = info_df['colunas_categoricas'][-1]
            else:
                # Para histograma, usa primeira numérica
                if info_df['colunas_numericas']:
                    params_extraidos["coluna"] = info_df['colunas_numericas'][0]
            
            if "coluna" in params_extraidos:
                st.session_state.pensamento_agentes.append(f"Coluna não especificada, usando: {params_extraidos['coluna']}")
    
    # Para scatter plots e boxplots - buscar duas colunas
    if "coluna_x" in parametros_funcao and "coluna_y" in parametros_funcao:
        colunas_mencionadas = []
        for col in info_df['colunas']:
            if col.lower() in pergunta_lower:
                colunas_mencionadas.append(col)
        
        if len(colunas_mencionadas) >= 2:
            params_extraidos["coluna_x"] = colunas_mencionadas[0]
            params_extraidos["coluna_y"] = colunas_mencionadas[1]
        else:
            # Defaults diferentes para scatter e boxplot
            if ferramenta == "gerar_scatter_plot":
                # Scatter: duas numéricas
                if len(info_df['colunas_numericas']) >= 2:
                    params_extraidos["coluna_x"] = info_df['colunas_numericas'][0]
                    params_extraidos["coluna_y"] = info_df['colunas_numericas'][1]
            else:
                # Boxplot: categórica vs numérica
                if info_df['colunas_categoricas'] and info_df['colunas_numericas']:
                    params_extraidos["coluna_x"] = info_df['colunas_categoricas'][0]
                    params_extraidos["coluna_y"] = info_df['colunas_numericas'][0]
            
            if "coluna_x" in params_extraidos:
                st.session_state.pensamento_agentes.append(f"Colunas não especificadas, usando: {params_extraidos['coluna_x']} vs {params_extraidos['coluna_y']}")
    
    st.session_state.pensamento_agentes.append(f"Parâmetros extraídos: {params_extraidos}")
    return params_extraidos

# --- Extrator de Insights ---
def extrair_insights_do_resultado(ferramenta, df, parametros):
    """Extrai insights automáticos baseados na ferramenta executada"""
    insights = []
    
    try:
        if ferramenta == "mostrar_tipos_de_dados":
            num_colunas = len(df.columns)
            num_numericas = len(df.select_dtypes(include=['number']).columns)
            num_categoricas = len(df.select_dtypes(include=['object', 'category', 'bool']).columns)
            insights.append(f"Dataset possui {num_colunas} colunas: {num_numericas} numéricas e {num_categoricas} categóricas")
            
        elif ferramenta == "mostrar_estatisticas_descritivas":
            numeric_df = df.select_dtypes(include=['number'])
            for col in numeric_df.columns[:3]:
                media = numeric_df[col].mean()
                mediana = numeric_df[col].median()
                if abs(media - mediana) / (abs(media) + 0.0001) > 0.2:
                    insights.append(f"{col}: Diferença significativa entre média ({media:.2f}) e mediana ({mediana:.2f})")
                    
        elif ferramenta == "gerar_histograma":
            if "coluna" in parametros:
                col = parametros["coluna"]
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    skew = df[col].skew()
                    if abs(skew) > 1:
                        insights.append(f"{col}: Distribuição altamente assimétrica (skewness={skew:.2f})")
                    
        elif ferramenta == "gerar_mapa_de_calor_correlacao":
            numeric_df = df.select_dtypes(include=['number'])
            corr = numeric_df.corr()
            correlacoes_fortes = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    if abs(corr.iloc[i, j]) > 0.7:
                        correlacoes_fortes.append(f"{corr.columns[i]} e {corr.columns[j]} ({corr.iloc[i, j]:.2f})")
            if correlacoes_fortes:
                insights.append(f"Correlações fortes: {', '.join(correlacoes_fortes[:3])}")
            else:
                insights.append("Não foram detectadas correlações fortes (>0.7)")
                        
        elif ferramenta == "encontrar_outliers_zscore":
            numeric_df = df.select_dtypes(include=['number'])
            z_scores = np.abs(stats.zscore(numeric_df.fillna(0)))
            outliers = (z_scores > 3).sum()
            total_outliers = outliers.sum()
            if total_outliers > 0:
                colunas_com_outliers = outliers[outliers > 0]
                for col in colunas_com_outliers.index[:3]:
                    insights.append(f"{col}: {colunas_com_outliers[col]} outliers ({colunas_com_outliers[col]/len(df)*100:.1f}%)")
            else:
                insights.append("Nenhum outlier significativo detectado")
                
    except Exception as e:
        st.session_state.pensamento_agentes.append(f"Erro ao extrair insights: {e}")
    
    return insights

# --- Analisador com LLM ---
def analisar_resultado_com_llm(ferramenta, resultado_textual, insights_tecnicos, pergunta_usuario, df):
    """Usa a LLM para interpretar os resultados de forma conversacional"""
    insights_str = "\n".join([f"- {i}" for i in insights_tecnicos]) if insights_tecnicos else "Nenhum insight técnico específico"
    
    prompt = f"""Você é um analista de dados experiente e amigável. Acabou de realizar uma análise e está explicando os resultados para alguém interessado.

Use uma linguagem natural, como se estivesse explicando para um colega de trabalho. Pode usar frases como "o que me chamou atenção foi...", "olha só que interessante...", "percebi que...".

Evite parecer robótico ou formal demais. Seja claro, direto, mas com um tom leve e humano.

PERGUNTA DO USUÁRIO: "{pergunta_usuario}"

ANÁLISE REALIZADA: {ferramenta}

RESULTADO DA ANÁLISE:
{resultado_textual}

INSIGHTS TÉCNICOS ADICIONAIS:
{insights_str}

Sua tarefa:
1. Responda DIRETAMENTE à pergunta do usuário
2. Explique o que a análise mostrou de forma conversacional
3. Destaque os pontos mais importantes ou surpreendentes
4. Se houver algo preocupante (outliers, dados faltantes, etc), mencione
5. Sugira brevemente o que pode ser interessante investigar em seguida

Seja conciso (máximo 6-7 linhas) e responda em português do Brasil."""

    try:
        st.session_state.pensamento_agentes.append("Enviando resultados para análise da LLM...")
        response = st.session_state.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.7,
            max_tokens=600
        )
        st.session_state.pensamento_agentes.append("LLM analisou os resultados com sucesso")
        return response.choices[0].message.content
    except Exception as e:
        st.session_state.pensamento_agentes.append(f"Erro ao analisar com LLM: {e}")
        return f"Análise concluída. Principais descobertas:\n{insights_str}"

# --- Agente de Conclusões ---
def agente_conclusoes_completo(df, analises_realizadas, insights_descobertos):
    """Gera conclusões abrangentes sobre o dataset"""
    if not analises_realizadas and not insights_descobertos:
        return "Ainda não realizei análises suficientes. Por favor, faça algumas perguntas sobre os dados primeiro."
    
    info_df = obter_info_completa_dataframe(df)
    numeric_df = df.select_dtypes(include=['number'])
    
    estatisticas_basicas = ""
    if not numeric_df.empty:
        for col in numeric_df.columns[:5]:
            estatisticas_basicas += f"\n- {col}: média={numeric_df[col].mean():.2f}, mediana={numeric_df[col].median():.2f}"
    
    tipos_analises = list(set([analise.get('tipo', 'desconhecido') for analise in analises_realizadas]))
    resumo_analises = f"{len(analises_realizadas)} análises realizadas: {', '.join(tipos_analises)}"
    
    top_insights = insights_descobertos[-15:] if len(insights_descobertos) > 15 else insights_descobertos
    resumo_insights = "\n".join([f"- {insight}" for insight in top_insights])
    
    nulos_importantes = {k: v for k, v in info_df['valores_nulos'].items() if v > 0}
    resumo_nulos = ""
    if nulos_importantes:
        resumo_nulos = "\nVALORES NULOS:\n" + "\n".join([f"- {k}: {v} ({v/info_df['shape'][0]*100:.1f}%)" for k, v in list(nulos_importantes.items())[:5]])
    
    prompt = f"""Você é um cientista de dados experiente e amigável. Acabou de analisar um dataset completo e está explicando suas conclusões.

Use linguagem natural e conversacional. Seja claro, direto, mas com tom leve e humano.

INFORMAÇÕES DO DATASET:
- Total de registros: {info_df['shape'][0]:,}
- Total de colunas: {info_df['shape'][1]}
- Colunas numéricas: {', '.join(info_df['colunas_numericas'][:5])}
- Colunas categóricas: {', '.join(info_df['colunas_categoricas'][:5])}

ESTATÍSTICAS:
{estatisticas_basicas}
{resumo_nulos}

ANÁLISES: {resumo_analises}

DESCOBERTAS:
{resumo_insights if resumo_insights else '- Análises básicas realizadas'}

Forneça uma síntese conversacional:

**🔍 O que descobri sobre os dados**
(2-3 linhas sobre estrutura, tamanho, tipos)

**📊 Padrões interessantes**
(3-4 linhas sobre padrões, correlações, distribuições)

**⚠️ Pontos de atenção**
(2-3 linhas sobre problemas: nulos, outliers, assimetrias)

**💡 Próximos passos**
(2-3 sugestões práticas)

Responda em português do Brasil."""

    try:
        st.session_state.pensamento_agentes.append("Gerando conclusões com LLM...")
        response = st.session_state.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.7,
            max_tokens=1200
        )
        return response.choices[0].message.content
    except Exception as e:
        st.session_state.pensamento_agentes.append(f"Erro ao gerar conclusões: {e}")
        conclusao_fallback = f"Baseado nas análises:\n\n**Dados:** {info_df['shape'][0]:,} registros, {info_df['shape'][1]} colunas\n\n**Descobertas:**\n"
        for insight in insights_descobertos[-5:]:
            conclusao_fallback += f"- {insight}\n"
        return conclusao_fallback

# =============================================================================
# FERRAMENTAS DE ANÁLISE
# =============================================================================

def mostrar_tipos_de_dados(df):
    """Mostra os tipos de dados de cada coluna"""
    st.subheader("🔎 Tipos de Dados")
    tipos = pd.DataFrame({
        "Coluna": df.columns,
        "Tipo de Dado": df.dtypes.astype(str),
        "Valores Nulos": df.isnull().sum(),
    })
    st.dataframe(tipos)
    
    colunas_numericas = df.select_dtypes(include=['number']).columns.tolist()
    colunas_categoricas = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    resumo = f"""TIPOS DE DADOS DO DATASET:

Colunas Numéricas ({len(colunas_numericas)}): {', '.join(colunas_numericas) if colunas_numericas else 'Nenhuma'}

Colunas Categóricas ({len(colunas_categoricas)}): {', '.join(colunas_categoricas) if colunas_categoricas else 'Nenhuma'}

Valores Nulos:
{df.isnull().sum().to_string()}
"""
    return resumo

def mostrar_estatisticas_descritivas(df):
    """Mostra estatísticas descritivas"""
    st.subheader("📊 Estatísticas Descritivas")
    desc = df.describe(include="all").T
    st.dataframe(desc)
    
    numeric_df = df.select_dtypes(include=['number'])
    resumo = "ESTATÍSTICAS DESCRITIVAS:\n\n"
    for col in numeric_df.columns[:10]:
        resumo += f"{col}:\n"
        resumo += f"  - Média: {numeric_df[col].mean():.2f}\n"
        resumo += f"  - Mediana: {numeric_df[col].median():.2f}\n"
        resumo += f"  - Desvio: {numeric_df[col].std():.2f}\n"
        resumo += f"  - Min: {numeric_df[col].min():.2f}, Max: {numeric_df[col].max():.2f}\n\n"
    return resumo

def gerar_histograma(df, coluna):
    """Gera histograma melhorado"""
    st.subheader(f"📈 Distribuição: {coluna}")
    
    col_data = df[coluna].dropna()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(col_data, kde=True, ax=ax, color='steelblue', alpha=0.6, bins=30)
    
    media = col_data.mean()
    mediana = col_data.median()
    
    ax.axvline(media, color='red', linestyle='--', linewidth=2, label=f'Média: {media:.2f}')
    ax.axvline(mediana, color='green', linestyle='--', linewidth=2, label=f'Mediana: {mediana:.2f}')
    
    ax.set_xlabel(coluna, fontsize=12)
    ax.set_ylabel('Frequência', fontsize=12)
    ax.set_title(f'Distribuição de {coluna}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    skewness = col_data.skew()
    kurtosis = col_data.kurtosis()
    q1 = col_data.quantile(0.25)
    q3 = col_data.quantile(0.75)
    
    if abs(skewness) < 0.5:
        interpretacao_skew = "Distribuição SIMÉTRICA"
        sugestao = "Não é necessária transformação"
    elif skewness > 1:
        interpretacao_skew = "Distribuição ASSIMÉTRICA À DIREITA"
        sugestao = "Considere transformação logarítmica (log) ou raiz quadrada"
    elif skewness > 0:
        interpretacao_skew = "Distribuição levemente assimétrica à direita"
        sugestao = "Assimetria moderada, transformação opcional"
    else:
        interpretacao_skew = "Distribuição ASSIMÉTRICA À ESQUERDA"
        sugestao = "Considere transformação exponencial"
    
    resumo = f"""DISTRIBUIÇÃO DA COLUNA '{coluna}':

📊 Estatísticas:
- Média: {media:.2f}
- Mediana: {mediana:.2f}
- Desvio Padrão: {col_data.std():.2f}
- Mínimo: {col_data.min():.2f}
- Máximo: {col_data.max():.2f}
- Q1: {q1:.2f}, Q3: {q3:.2f}

📐 Forma:
- Assimetria: {skewness:.2f} → {interpretacao_skew}
- Curtose: {kurtosis:.2f}

💡 Sugestão: {sugestao}

⚠️ Diferença Média vs Mediana: {abs(media - mediana):.2f} ({abs(media - mediana) / media * 100:.1f}%)
"""
    return resumo

def gerar_mapa_de_calor_correlacao(df):
    """Gera mapa de calor de correlações"""
    st.subheader("🔥 Mapa de Calor das Correlações")
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        st.warning("Nenhuma coluna numérica encontrada.")
        return "Não foi possível gerar o mapa de calor."

    corr = numeric_df.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)

    correlacoes_fortes = []
    correlacoes_moderadas = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            var1, var2 = corr.columns[i], corr.columns[j]
            valor = corr.iloc[i, j]
            if abs(valor) > 0.7:
                correlacoes_fortes.append(f"{var1} e {var2}: {valor:.2f}")
            elif abs(valor) > 0.4:
                correlacoes_moderadas.append(f"{var1} e {var2}: {valor:.2f}")

    resumo = "ANÁLISE DE CORRELAÇÕES:\n\n"
    if correlacoes_fortes:
        resumo += f"Correlações Fortes (|r| > 0.7): {len(correlacoes_fortes)}\n"
        for c in correlacoes_fortes[:10]:
            resumo += f"- {c}\n"
    else:
        resumo += "Nenhuma correlação forte detectada.\n"
    
    if correlacoes_moderadas:
        resumo += f"\nCorrelações Moderadas (0.4 < |r| < 0.7): {len(correlacoes_moderadas)}\n"
        for c in correlacoes_moderadas[:5]:
            resumo += f"- {c}\n"
    
    return resumo

def gerar_boxplot(df, coluna_x, coluna_y):
    """Gera boxplot"""
    st.subheader(f"📦 Boxplot: {coluna_y} por {coluna_x}")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x=coluna_x, y=coluna_y, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    grupos = df.groupby(coluna_x)[coluna_y].agg(['mean', 'median', 'std', 'count'])
    
    resumo = f"""COMPARAÇÃO: {coluna_y} por {coluna_x}

Estatísticas por Grupo:
{grupos.to_string()}

Maior média: {grupos['mean'].idxmax()} ({grupos['mean'].max():.2f})
Menor média: {grupos['mean'].idxmin()} ({grupos['mean'].min():.2f})
Diferença: {grupos['mean'].max() - grupos['mean'].min():.2f}
"""
    return resumo

def encontrar_outliers_zscore(df):
    """Detecta outliers usando Z-score"""
    st.subheader("🚨 Detecção de Outliers (Z-score)")
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        st.warning("Nenhuma coluna numérica encontrada.")
        return "Não foi possível detectar outliers."

    z_scores = np.abs(stats.zscore(numeric_df.fillna(0)))
    outliers = (z_scores > 3).sum()
    resultado = pd.DataFrame({"Coluna": numeric_df.columns, "Outliers": outliers})
    st.dataframe(resultado)

    total_outliers = outliers.sum()
    colunas_com_outliers = resultado[resultado['Outliers'] > 0]

    resumo = f"""DETECÇÃO DE OUTLIERS (Z-score > 3):

Total: {total_outliers}
Percentual: {(total_outliers / (len(df) * len(numeric_df.columns)) * 100):.2f}%

Colunas com outliers:
"""
    if len(colunas_com_outliers) > 0:
        for _, row in colunas_com_outliers.iterrows():
            pct = (row['Outliers'] / len(df)) * 100
            resumo += f"- {row['Coluna']}: {int(row['Outliers'])} ({pct:.2f}%)\n"
    else:
        resumo += "Nenhum outlier significativo detectado.\n"
    
    return resumo

def gerar_scatter_plot(df, coluna_x, coluna_y):
    """Gera gráfico de dispersão"""
    st.subheader(f"📍 Gráfico de Dispersão: {coluna_x} vs {coluna_y}")
    
    if not pd.api.types.is_numeric_dtype(df[coluna_x]) or not pd.api.types.is_numeric_dtype(df[coluna_y]):
        st.warning("Ambas as colunas precisam ser numéricas.")
        return "Não foi possível gerar scatter plot."
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df[coluna_x], df[coluna_y], alpha=0.5, s=20, color='steelblue')
    ax.set_xlabel(coluna_x)
    ax.set_ylabel(coluna_y)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    correlacao = df[[coluna_x, coluna_y]].corr().iloc[0, 1]
    
    if correlacao > 0.7:
        interp = "Correlação positiva FORTE"
    elif correlacao > 0.4:
        interp = "Correlação positiva moderada"
    elif correlacao < -0.7:
        interp = "Correlação negativa FORTE"
    elif correlacao < -0.4:
        interp = "Correlação negativa moderada"
    else:
        interp = "Correlação fraca"
    
    resumo = f"""GRÁFICO DE DISPERSÃO: {coluna_x} vs {coluna_y}

Correlação de Pearson: {correlacao:.3f}
Interpretação: {interp}

Estatísticas:
- {coluna_x}: média={df[coluna_x].mean():.2f}, desvio={df[coluna_x].std():.2f}
- {coluna_y}: média={df[coluna_y].mean():.2f}, desvio={df[coluna_y].std():.2f}
"""
    return resumo

def analisar_distribuicao_classes(df, coluna):
    """Analisa distribuição de classes"""
    st.subheader(f"📊 Distribuição de Classes: {coluna}")
    
    if coluna not in df.columns:
        st.warning(f"Coluna '{coluna}' não encontrada.")
        return f"Coluna '{coluna}' não encontrada."
    
    contagem = df[coluna].value_counts()
    percentual = df[coluna].value_counts(normalize=True) * 100
    
    resumo_df = pd.DataFrame({
        'Classe': contagem.index,
        'Quantidade': contagem.values,
        'Percentual (%)': percentual.values.round(2)
    })
    
    st.dataframe(resumo_df)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(contagem.index.astype(str), contagem.values, color='steelblue', alpha=0.7)
    ax.set_xlabel(coluna)
    ax.set_ylabel('Quantidade')
    ax.set_title(f'Distribuição de {coluna}')
    
    for i, v in enumerate(contagem.values):
        ax.text(i, v + max(contagem.values)*0.01, str(v), ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    total = len(df)
    n_classes = len(contagem)
    
    resumo = f"""DISTRIBUIÇÃO DE CLASSES: {coluna}

Total de registros: {total:,}
Número de classes: {n_classes}

Distribuição:
"""
    
    for classe, qtd, pct in zip(resumo_df['Classe'], resumo_df['Quantidade'], resumo_df['Percentual (%)']):
        resumo += f"- {classe}: {qtd:,} ({pct:.2f}%)\n"
    
    resumo += f"""
Classe majoritária: {contagem.index[0]} ({percentual.iloc[0]:.2f}%)
Classe minoritária: {contagem.index[-1]} ({percentual.iloc[-1]:.2f}%)

Balanceamento:
"""
    
    ratio = percentual.iloc[0] / percentual.iloc[-1]
    if ratio > 10:
        resumo += f"- Dataset MUITO DESBALANCEADO (ratio {ratio:.1f}:1)\n"
        resumo += "- Recomenda-se SMOTE, undersampling ou ajuste de pesos\n"
    elif ratio > 3:
        resumo += f"- Dataset DESBALANCEADO (ratio {ratio:.1f}:1)\n"
        resumo += "- Considerar técnicas de balanceamento\n"
    else:
        resumo += f"- Dataset BALANCEADO (ratio {ratio:.1f}:1)\n"
    
    return resumo

# Registrar todas as ferramentas
caixa_de_ferramentas = {
    "mostrar_tipos_de_dados": mostrar_tipos_de_dados,
    "mostrar_estatisticas_descritivas": mostrar_estatisticas_descritivas,
    "gerar_histograma": gerar_histograma,
    "gerar_mapa_de_calor_correlacao": gerar_mapa_de_calor_correlacao,
    "gerar_boxplot": gerar_boxplot,
    "encontrar_outliers_zscore": encontrar_outliers_zscore,
    "gerar_scatter_plot": gerar_scatter_plot,
    "analisar_distribuicao_classes": analisar_distribuicao_classes,
}