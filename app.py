import streamlit as st
import pandas as pd
import ollama
import tools
import inspect
import json
from datetime import datetime

# --- Configuração da Página ---
st.set_page_config(page_title="Sistema Multiagente - EDA Genérico", page_icon="🔍", layout="wide")

# --- Estado da Sessão ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "analises_realizadas" not in st.session_state:
    st.session_state.analises_realizadas = []
if "insights_descobertos" not in st.session_state:
    st.session_state.insights_descobertos = []
if "pensamento_agentes" not in st.session_state:
    st.session_state.pensamento_agentes = []

# --- Ferramentas Expandidas ---
caixa_de_ferramentas = {
    "mostrar_tipos_de_dados": tools.mostrar_tipos_de_dados,
    "mostrar_estatisticas_descritivas": tools.mostrar_estatisticas_descritivas,
    "gerar_histograma": tools.gerar_histograma,
    "gerar_mapa_de_calor_correlacao": tools.gerar_mapa_de_calor_correlacao,
    "gerar_boxplot": tools.gerar_boxplot,
    "encontrar_outliers_zscore": tools.encontrar_outliers_zscore,
}

# --- Função para classificar perguntas ---
def classificar_pergunta(pergunta):
    """Classifica o tipo de pergunta para melhor roteamento"""
    pergunta_lower = pergunta.lower()
    
    # Ordem importa! Verificar conclusões PRIMEIRO
    if any(word in pergunta_lower for word in ["conclusão", "conclusao", "conclusões", "conclusoes", "insight", "o que você", "o que voce", "descobri", "aprende"]):
        return "conclusoes"
    elif any(word in pergunta_lower for word in ["compar", "boxplot", "versus", "vs", " por ", " x ", "diferença", "diferenca"]):
        return "comparacao"
    elif any(word in pergunta_lower for word in ["outlier", "atípico", "atipico", "anomal", "discrepan", "valores extremos"]):
        return "outliers"
    elif any(word in pergunta_lower for word in ["correlação", "correlacao", "relacao", "relaciona", "mapa de calor"]):
        return "correlacao"
    elif any(word in pergunta_lower for word in ["distribui", "histograma", "frequen"]):
        return "distribuicao"
    elif any(word in pergunta_lower for word in ["tipo", "tipos", "categori", "numeri"]):
        return "tipos_dados"
    elif any(word in pergunta_lower for word in ["estatística", "estatistica", "média", "media", "mediana", "desvio", "resumo"]):
        return "estatisticas"
    else:
        return "geral"

# --- Função para obter informações detalhadas do DataFrame ---
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

# --- Função para extrair insights de resultados ---
def extrair_insights_do_resultado(ferramenta, df, parametros):
    """Extrai insights automáticos baseados na ferramenta executada"""
    insights = []
    
    try:
        if ferramenta == "mostrar_tipos_de_dados":
            num_colunas = len(df.columns)
            num_numericas = len(df.select_dtypes(include=['number']).columns)
            insights.append(f"Dataset possui {num_colunas} colunas, sendo {num_numericas} numéricas")
            
        elif ferramenta == "mostrar_estatisticas_descritivas":
            numeric_df = df.select_dtypes(include=['number'])
            for col in numeric_df.columns:
                media = numeric_df[col].mean()
                mediana = numeric_df[col].median()
                if abs(media - mediana) / (abs(media) + 0.0001) > 0.2:
                    insights.append(f"{col}: Diferença significativa entre média ({media:.2f}) e mediana ({mediana:.2f}), indica distribuição assimétrica")
                    
        elif ferramenta == "gerar_histograma":
            if "coluna" in parametros:
                col = parametros["coluna"]
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    skew = df[col].skew()
                    if abs(skew) > 1:
                        insights.append(f"{col}: Distribuição altamente assimétrica (skewness={skew:.2f})")
                    media = df[col].mean()
                    mediana = df[col].median()
                    insights.append(f"{col}: Média={media:.2f}, Mediana={mediana:.2f}")
                    
        elif ferramenta == "gerar_mapa_de_calor_correlacao":
            numeric_df = df.select_dtypes(include=['number'])
            corr = numeric_df.corr()
            correlacoes_fortes = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    if abs(corr.iloc[i, j]) > 0.7:
                        correlacoes_fortes.append(f"{corr.columns[i]} e {corr.columns[j]} ({corr.iloc[i, j]:.2f})")
            if correlacoes_fortes:
                insights.append(f"Correlações fortes detectadas: {', '.join(correlacoes_fortes[:3])}")
            else:
                insights.append("Não foram detectadas correlações fortes (>0.7) entre as variáveis")
                        
        elif ferramenta == "encontrar_outliers_zscore":
            from scipy import stats
            import numpy as np
            numeric_df = df.select_dtypes(include=['number'])
            z_scores = np.abs(stats.zscore(numeric_df.fillna(0)))
            outliers = (z_scores > 3).sum()
            total_outliers = outliers.sum()
            if total_outliers > 0:
                colunas_com_outliers = outliers[outliers > 0]
                for col in colunas_com_outliers.index[:3]:
                    insights.append(f"{col}: {colunas_com_outliers[col]} outliers ({colunas_com_outliers[col]/len(df)*100:.1f}%)")
            else:
                insights.append("Nenhum outlier significativo detectado (Z-score > 3)")
                
        elif ferramenta == "gerar_boxplot":
            if "coluna_x" in parametros and "coluna_y" in parametros:
                col_x = parametros["coluna_x"]
                col_y = parametros["coluna_y"]
                if col_x in df.columns and col_y in df.columns:
                    grupos = df.groupby(col_x)[col_y].agg(['mean', 'count'])
                    maior_media = grupos['mean'].idxmax()
                    insights.append(f"Grupo '{maior_media}' apresenta maior média em {col_y} ({grupos.loc[maior_media, 'mean']:.2f})")
                
    except Exception as e:
        st.session_state.pensamento_agentes.append(f"Erro ao extrair insights: {e}")
    
    return insights

# --- Agente Decisor de Gráficos ---
def agente_deve_gerar_grafico(pergunta_usuario, ferramenta_escolhida):
    """Decide se deve gerar gráfico baseado no contexto"""
    
    # Sempre gerar gráfico para estas ferramentas
    ferramentas_graficas = ["gerar_histograma", "gerar_mapa_de_calor_correlacao", "gerar_boxplot"]
    
    if ferramenta_escolhida in ferramentas_graficas:
        return True
    
    # Verificar se usuário pediu explicitamente
    palavras_grafico = ["gráfico", "grafico", "visualiza", "plota", "mostra", "desenha"]
    if any(palavra in pergunta_usuario.lower() for palavra in palavras_grafico):
        return True
    
    # Para perguntas sobre distribuição, correlação, comparação
    palavras_visual = ["distribui", "correlação", "correlacao", "compara", "relação", "relacao"]
    if any(palavra in pergunta_usuario.lower() for palavra in palavras_visual):
        return True
        
    return False

# --- Agente Roteador Inteligente ---
def agente_roteador_inteligente(pergunta_usuario, ferramentas, info_df, tipo_pergunta):
    """Roteador baseado na classificação da pergunta"""
    
    st.session_state.pensamento_agentes.append(f"Tipo de pergunta identificado: {tipo_pergunta}")
    
    # Mapeamento direto baseado no tipo
    mapeamento = {
        "tipos_dados": "mostrar_tipos_de_dados",
        "estatisticas": "mostrar_estatisticas_descritivas", 
        "distribuicao": "gerar_histograma",
        "correlacao": "gerar_mapa_de_calor_correlacao",
        "outliers": "encontrar_outliers_zscore",
        "comparacao": "gerar_boxplot",
        "geral": "mostrar_estatisticas_descritivas"
    }
    
    ferramenta_escolhida = mapeamento.get(tipo_pergunta, "mostrar_estatisticas_descritivas")
    st.session_state.pensamento_agentes.append(f"Ferramenta selecionada: {ferramenta_escolhida}")
    
    return ferramenta_escolhida

# --- Agente Extrator Melhorado ---
def agente_extrator_parametros_melhorado(pergunta_usuario, ferramenta, info_df):
    """Extrator de parâmetros mais inteligente"""
    
    funcao = caixa_de_ferramentas[ferramenta]
    assinatura = inspect.signature(funcao)
    parametros_funcao = [p for p in assinatura.parameters.keys() if p != 'df']
    
    if not parametros_funcao:
        st.session_state.pensamento_agentes.append("Função não precisa de parâmetros adicionais")
        return {}
    
    st.session_state.pensamento_agentes.append(f"Buscando parâmetros: {parametros_funcao}")
    
    params_extraidos = {}
    pergunta_lower = pergunta_usuario.lower()
    
    # Para histogramas - buscar nome da coluna
    if "coluna" in parametros_funcao:
        for col in info_df['colunas']:
            if col.lower() in pergunta_lower:
                params_extraidos["coluna"] = col
                break
        
        # Se não encontrou, usa a primeira numérica
        if "coluna" not in params_extraidos and info_df['colunas_numericas']:
            params_extraidos["coluna"] = info_df['colunas_numericas'][0]
            st.session_state.pensamento_agentes.append(f"Coluna não especificada, usando: {params_extraidos['coluna']}")
    
    # Para boxplots - buscar duas colunas
    if "coluna_x" in parametros_funcao and "coluna_y" in parametros_funcao:
        colunas_mencionadas = []
        for col in info_df['colunas']:
            if col.lower() in pergunta_lower:
                colunas_mencionadas.append(col)
        
        if len(colunas_mencionadas) >= 2:
            params_extraidos["coluna_x"] = colunas_mencionadas[0]
            params_extraidos["coluna_y"] = colunas_mencionadas[1]
        else:
            # Usa defaults: primeira categórica e primeira numérica
            if info_df['colunas_categoricas'] and info_df['colunas_numericas']:
                params_extraidos["coluna_x"] = info_df['colunas_categoricas'][0]
                params_extraidos["coluna_y"] = info_df['colunas_numericas'][0]
                st.session_state.pensamento_agentes.append(f"Colunas não especificadas, usando: {params_extraidos['coluna_x']} vs {params_extraidos['coluna_y']}")
    
    st.session_state.pensamento_agentes.append(f"Parâmetros extraídos: {params_extraidos}")
    return params_extraidos

# --- Agente de Conclusões COMPLETO ---
def agente_conclusoes_completo(df, analises_realizadas, insights_descobertos):
    """Gera conclusões abrangentes sobre TODO o dataset analisado"""
    
    if not analises_realizadas and not insights_descobertos:
        return "Ainda não realizei análises suficientes. Por favor, faça algumas perguntas sobre os dados primeiro para que eu possa gerar conclusões."
    
    # Coletar informações do dataset
    info_df = obter_info_completa_dataframe(df)
    numeric_df = df.select_dtypes(include=['number'])
    
    # Preparar contexto rico para o LLM
    contexto_dataset = f"""
INFORMAÇÕES DO DATASET:
- Total de registros: {info_df['shape'][0]:,}
- Total de colunas: {info_df['shape'][1]}
- Colunas numéricas ({len(info_df['colunas_numericas'])}): {', '.join(info_df['colunas_numericas'][:5])}{'...' if len(info_df['colunas_numericas']) > 5 else ''}
- Colunas categóricas ({len(info_df['colunas_categoricas'])}): {', '.join(info_df['colunas_categoricas'][:5])}{'...' if len(info_df['colunas_categoricas']) > 5 else ''}
"""

    # Estatísticas básicas selecionadas
    estatisticas_basicas = ""
    if not numeric_df.empty:
        # Pegar apenas colunas mais relevantes
        colunas_relevantes = numeric_df.columns[:3]  # Primeiras 3
        for col in colunas_relevantes:
            estatisticas_basicas += f"\n- {col}: média={numeric_df[col].mean():.2f}, mediana={numeric_df[col].median():.2f}, min={numeric_df[col].min():.2f}, max={numeric_df[col].max():.2f}"

    # Resumir análises realizadas
    tipos_analises = list(set([analise['tipo'] for analise in analises_realizadas[-15:]]))
    resumo_analises = f"Realizadas {len(analises_realizadas)} análises: {', '.join(tipos_analises)}"
    
    # Top insights
    top_insights = insights_descobertos[-10:] if len(insights_descobertos) > 10 else insights_descobertos
    resumo_insights = "\n".join([f"- {insight}" for insight in top_insights])
    
    prompt = f"""Você é um cientista de dados. Forneça conclusões OBJETIVAS e DIRETAS sobre o dataset.

{contexto_dataset}

AMOSTRA DE ESTATÍSTICAS:
{estatisticas_basicas}

ANÁLISES: {resumo_analises}

PRINCIPAIS DESCOBERTAS:
{resumo_insights if resumo_insights else '- Análises estatísticas básicas realizadas'}

INSTRUÇÕES CRÍTICAS:
1. NÃO repita as estatísticas já mostradas
2. NÃO gere novas tabelas ou análises
3. SINTETIZE os padrões encontrados
4. Seja DIRETO e OBJETIVO

Forneça suas conclusões em 4 seções curtas:

**1. CARACTERÍSTICAS DO DATASET** (2-3 linhas sobre tamanho, estrutura, qualidade)

**2. PADRÕES IDENTIFICADOS** (2-3 principais descobertas dos insights)

**3. QUALIDADE DOS DADOS** (valores nulos, outliers, problemas encontrados)

**4. RECOMENDAÇÕES** (2 sugestões práticas para análises futuras)

Responda em português do Brasil. Seja conciso."""

    try:
        st.session_state.pensamento_agentes.append("Conectando ao LLM local (phi3:mini)...")
        response = ollama.chat(
            model='phi3:mini',
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.3, 'num_predict': 1000}
        )
        st.session_state.pensamento_agentes.append("LLM respondeu com sucesso")
        return response['message']['content'].strip()
    except Exception as e:
        st.session_state.pensamento_agentes.append(f"Erro ao gerar conclusões: {e}")
        
        # Fallback: gerar conclusões baseadas apenas nos insights
        conclusao_fallback = "Baseado nas análises realizadas:\n\n"
        conclusao_fallback += "PRINCIPAIS DESCOBERTAS:\n"
        for insight in insights_descobertos[-5:]:
            conclusao_fallback += f"- {insight}\n"
        conclusao_fallback += f"\nForam realizadas {len(analises_realizadas)} análises no total."
        return conclusao_fallback

# --- Interface Principal ---
st.title("Sistema Multiagente - EDA Genérico")
st.markdown("*Sistema inteligente para análise exploratória de qualquer dataset CSV*")

# --- Sidebar ---
with st.sidebar:
    st.header("Configuração")
    uploaded_file = st.file_uploader("Upload do arquivo CSV:", type=["csv"])
    
    if st.session_state.df is not None:
        info = obter_info_completa_dataframe(st.session_state.df)
        st.success("Dataset carregado!")
        
        with st.expander("Informações do Dataset", expanded=False):
            st.metric("Linhas", info['shape'][0])
            st.metric("Colunas", info['shape'][1])
            st.metric("Memória (MB)", info['memoria_mb'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Numéricas:**")
                st.write(info['colunas_numericas'][:3] + (['...'] if len(info['colunas_numericas']) > 3 else []))
            with col2:
                st.write("**Categóricas:**")  
                st.write(info['colunas_categoricas'][:3] + (['...'] if len(info['colunas_categoricas']) > 3 else []))
        
        # Progresso das análises
        if st.session_state.analises_realizadas:
            st.write("### Análises Realizadas")
            for i, analise in enumerate(st.session_state.analises_realizadas[-5:], 1):
                st.write(f"{i}. {analise['tipo']}")
        
        # Insights descobertos
        if st.session_state.insights_descobertos:
            st.write("### Insights")
            for insight in st.session_state.insights_descobertos[-3:]:
                st.info(insight)

# --- Área Principal ---
if uploaded_file is None:
    st.info("Faça o upload de um arquivo CSV para começar")
    
    st.write("### Exemplos de perguntas:")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Descrição dos Dados:**")
        st.write("- Quais são os tipos de dados?")
        st.write("- Mostre as estatísticas descritivas")
        
        st.write("**Detecção de Anomalias:**") 
        st.write("- Existem valores atípicos?")
        st.write("- Encontre outliers nos dados")
    
    with col2:
        st.write("**Padrões e Tendências:**")
        st.write("- Mostre a distribuição da variável X")
        st.write("- Gere um histograma")
        
        st.write("**Relações e Conclusões:**")
        st.write("- Como as variáveis se relacionam?")
        st.write("- Quais suas conclusões sobre os dados?")

else:
    # Carregamento do arquivo
    if st.session_state.df is None or st.session_state.get('file_name') != uploaded_file.name:
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.session_state.file_name = uploaded_file.name
            st.session_state.messages = [{"role": "assistant", "content": f"Dataset '{uploaded_file.name}' carregado com sucesso! O que gostaria de descobrir?"}]
            st.session_state.analises_realizadas = []
            st.session_state.insights_descobertos = []
            st.session_state.pensamento_agentes = []
        except Exception as e:
            st.error(f"Erro ao carregar: {e}")
            st.stop()
    
    df = st.session_state.df
    info_df = obter_info_completa_dataframe(df)

    # --- Exibir pensamento dos agentes ---
    if st.session_state.pensamento_agentes:
        with st.expander("Ver Pensamento dos Agentes", expanded=False):
            for pensamento in st.session_state.pensamento_agentes[-10:]:
                st.text(pensamento)

    # --- Chat Interface ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt_usuario := st.chat_input("Faça sua pergunta sobre os dados..."):
        st.session_state.messages.append({"role": "user", "content": prompt_usuario})
        st.session_state.pensamento_agentes = []
        
        with st.chat_message("user"):
            st.write(prompt_usuario)

        with st.chat_message("assistant"):
            with st.spinner("Agentes analisando..."):
                try:
                    # 1. Classificar pergunta
                    tipo_pergunta = classificar_pergunta(prompt_usuario)
                    
                    # 2. Verificar se é pergunta sobre conclusões
                    if tipo_pergunta == "conclusoes":
                        st.write("### 📊 Gerando Conclusões Finais...")
                        
                        conclusoes = agente_conclusoes_completo(
                            df,
                            st.session_state.analises_realizadas, 
                            st.session_state.insights_descobertos
                        )
                        
                        st.markdown(conclusoes)
                        st.session_state.messages.append({"role": "assistant", "content": conclusoes})
                        st.rerun()
                    
                    # 3. Processar normalmente
                    ferramenta_escolhida = agente_roteador_inteligente(
                        prompt_usuario, 
                        list(caixa_de_ferramentas.keys()), 
                        info_df, 
                        tipo_pergunta
                    )
                    
                    parametros_extraidos = agente_extrator_parametros_melhorado(
                        prompt_usuario, 
                        ferramenta_escolhida, 
                        info_df
                    )
                    
                    # VERIFICAÇÃO: Se for boxplot, garantir que tem os parâmetros necessários
                    if ferramenta_escolhida == "gerar_boxplot":
                        if "coluna_x" not in parametros_extraidos or "coluna_y" not in parametros_extraidos:
                            # Tentar extrair manualmente da pergunta
                            palavras = prompt_usuario.lower().split()
                            for col in info_df['colunas']:
                                if col.lower() in palavras:
                                    if col in info_df['colunas_categoricas'] and "coluna_x" not in parametros_extraidos:
                                        parametros_extraidos["coluna_x"] = col
                                    elif col in info_df['colunas_numericas'] and "coluna_y" not in parametros_extraidos:
                                        parametros_extraidos["coluna_y"] = col
                            
                            # Se ainda não tem, usar defaults inteligentes
                            if "coluna_x" not in parametros_extraidos and info_df['colunas_categoricas']:
                                parametros_extraidos["coluna_x"] = info_df['colunas_categoricas'][0]
                            if "coluna_y" not in parametros_extraidos and info_df['colunas_numericas']:
                                # Procurar "amount" ou similar na pergunta
                                for col in info_df['colunas_numericas']:
                                    if 'amount' in col.lower() or 'valor' in col.lower():
                                        parametros_extraidos["coluna_y"] = col
                                        break
                                if "coluna_y" not in parametros_extraidos:
                                    parametros_extraidos["coluna_y"] = info_df['colunas_numericas'][0]
                    
                    # 4. Decidir se deve gerar gráfico
                    deve_gerar_grafico = agente_deve_gerar_grafico(prompt_usuario, ferramenta_escolhida)
                    
                    # 5. Executar ferramenta
                    funcao_da_ferramenta = caixa_de_ferramentas[ferramenta_escolhida]
                    funcao_da_ferramenta(df=df, **parametros_extraidos)
                    
                    # 6. Extrair insights automáticos
                    novos_insights = extrair_insights_do_resultado(ferramenta_escolhida, df, parametros_extraidos)
                    st.session_state.insights_descobertos.extend(novos_insights)
                    
                    # 7. Registrar análise com insights
                    st.session_state.analises_realizadas.append({
                        "tipo": ferramenta_escolhida,
                        "parametros": parametros_extraidos,
                        "insights": novos_insights,
                        "timestamp": datetime.now()
                    })
                    
                    # 8. Gerar resposta textual inteligente usando LLM
                    prompt_resposta = f"""Responda à pergunta do usuário de forma direta e clara baseado nas análises realizadas.

PERGUNTA DO USUÁRIO: "{prompt_usuario}"

ANÁLISE EXECUTADA: {ferramenta_escolhida}
PARÂMETROS: {parametros_extraidos if parametros_extraidos else 'Nenhum'}

DESCOBERTAS:
{chr(10).join([f"- {insight}" for insight in novos_insights]) if novos_insights else 'Análise executada com sucesso'}

Responda à pergunta de forma direta em 2-3 frases, mencionando os principais achados. Seja objetivo e em português do Brasil."""

                    try:
                        response = ollama.chat(
                            model='phi3:mini',
                            messages=[{'role': 'user', 'content': prompt_resposta}],
                            options={'temperature': 0.3, 'num_predict': 200}
                        )
                        resposta = response['message']['content'].strip()
                    except:
                        # Fallback se LLM falhar
                        resposta = f"Análise concluída usando **{ferramenta_escolhida}**"
                        if novos_insights:
                            resposta += "\n\n**Principais descobertas:**\n"
                            for insight in novos_insights[:3]:
                                resposta += f"- {insight}\n"
                    
                    st.write(resposta)
                    st.session_state.messages.append({"role": "assistant", "content": resposta})

                except Exception as e:
                    error_msg = f"Erro: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.session_state.pensamento_agentes.append(f"Erro na execução: {e}")