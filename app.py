import streamlit as st
import pandas as pd
import tools as tools
from datetime import datetime
from groq import Groq

# --- Configuração da Página ---
st.set_page_config(page_title="Sistema Multiagente - EDA Genérico", page_icon="🔍", layout="wide")

# --- Inicializar cliente Groq ---
if "groq_client" not in st.session_state:
    try:
        api_key = st.secrets.get("GROQ_API_KEY", "")
        if not api_key:
            st.error("⚠️ GROQ_API_KEY não configurada. Configure em .streamlit/secrets.toml")
            st.stop()
        st.session_state.groq_client = Groq(api_key=api_key)
    except Exception as e:
        st.error(f"Erro ao inicializar Groq: {e}")
        st.stop()

# --- Estado da Sessão ---
for key in ["messages", "df", "analises_realizadas", "insights_descobertos", "pensamento_agentes"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key != "df" else None

# --- Sidebar ---
with st.sidebar:
    st.header("Configuração")
    uploaded_file = st.file_uploader("Upload do arquivo CSV:", type=["csv"])
    
    if st.session_state.df is not None:
        info = tools.obter_info_completa_dataframe(st.session_state.df)
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
        
        if st.session_state.analises_realizadas:
            st.write("### Análises Realizadas")
            for i, analise in enumerate(st.session_state.analises_realizadas[-5:], 1):
                st.write(f"{i}. {analise.get('tipo', 'desconhecido')}")
        
        if st.session_state.insights_descobertos:
            st.write("### Insights")
            for insight in st.session_state.insights_descobertos[-3:]:
                st.info(insight)
    
    # Modo Debug
    if st.checkbox("Modo Debug", value=False):
        if st.session_state.pensamento_agentes:
            st.write("### Pensamento dos Agentes")
            for pensamento in st.session_state.pensamento_agentes[-10:]:
                st.text(pensamento)

# --- Área Principal ---
st.title("Sistema Multiagente - EDA Genérico")
st.markdown("*Sistema inteligente para análise exploratória de qualquer dataset CSV*")

if uploaded_file is None:
    st.info("📎 Faça o upload de um arquivo CSV para começar")
    
    st.write("### Exemplos de perguntas:")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Descrição dos Dados:**")
        st.write("- Quais são os tipos de dados?")
        st.write("- Quais colunas são numéricas e categóricas?")
        st.write("- Mostre as estatísticas descritivas")
        st.write("- Qual a distribuição da coluna X?")
        
        st.write("**Detecção de Anomalias:**") 
        st.write("- Existem valores atípicos?")
        st.write("- Há outliers nos dados?")
    
    with col2:
        st.write("**Análise de Relações:**")
        st.write("- Existe correlação entre as variáveis?")
        st.write("- Mostre a relação entre X e Y")
        st.write("- Compare X por Y")
        
        st.write("**Síntese:**")
        st.write("- Quais são as conclusões?")
        st.write("- Me dê um resumo geral")
        st.write("- O que você descobriu?")
        st.write("- Qual a proporção da classe alvo?")

else:
    # Carregar dataset
    if st.session_state.df is None or st.session_state.get('file_name') != uploaded_file.name:
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.session_state.file_name = uploaded_file.name
            st.session_state.messages = [{"role": "assistant", "content": f"Opa! Dataset **'{uploaded_file.name}'** carregado com sucesso! 🎉\n\nO que você gostaria de descobrir sobre esses dados?"}]
            st.session_state.analises_realizadas = []
            st.session_state.insights_descobertos = []
            st.session_state.pensamento_agentes = []
        except Exception as e:
            st.error(f"Erro ao carregar arquivo: {e}")
            st.stop()
    
    df = st.session_state.df

    # Mostrar histórico de mensagens
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Input do usuário
    if prompt_usuario := st.chat_input("Faça sua pergunta sobre os dados..."):
        st.session_state.messages.append({"role": "user", "content": prompt_usuario})
        st.session_state.pensamento_agentes = []  # Limpar pensamentos anteriores

        tipo_pergunta = tools.classificar_pergunta(prompt_usuario)

        with st.chat_message("user"):
            st.write(prompt_usuario)

        with st.chat_message("assistant"):
            with st.spinner("Analisando..."):
                try:
                    # Se for pergunta de conclusões, usa o agente especializado
                    if tipo_pergunta == "conclusoes":
                        resposta = tools.agente_conclusoes_completo(
                            df, 
                            st.session_state.analises_realizadas, 
                            st.session_state.insights_descobertos
                        )
                        st.write(resposta)
                        st.session_state.messages.append({"role": "assistant", "content": resposta})
                    
                    else:
                        # Fluxo normal: roteador -> extrator -> execução -> análise com LLM
                        info_df = tools.obter_info_completa_dataframe(df)
                        
                        # 1. Escolher ferramenta
                        ferramenta_escolhida = tools.agente_roteador_inteligente(
                            prompt_usuario, 
                            list(tools.caixa_de_ferramentas.keys()), 
                            info_df,
                            tipo_pergunta
                        )
                        
                        # 2. Extrair parâmetros
                        parametros = tools.agente_extrator_parametros_melhorado(
                            prompt_usuario, 
                            ferramenta_escolhida, 
                            info_df
                        )
                        
                        # 3. Verificação especial para funções que precisam de 2 parâmetros
                        if ferramenta_escolhida in ["gerar_boxplot", "gerar_scatter_plot"]:
                            if "coluna_x" not in parametros or "coluna_y" not in parametros:
                                if ferramenta_escolhida == "gerar_boxplot":
                                    resposta = "Para gerar um boxplot, preciso que você especifique duas colunas: uma categórica e uma numérica. Por exemplo: 'compare Amount por Class'."
                                else:
                                    resposta = "Para gerar um gráfico de dispersão, preciso que você especifique duas colunas numéricas. Por exemplo: 'mostre a relação entre Time e Amount'."
                                
                                st.warning(resposta)
                                st.session_state.messages.append({"role": "assistant", "content": resposta})
                                st.stop()

                        # 4. Executar ferramenta
                        funcao = tools.caixa_de_ferramentas[ferramenta_escolhida]
                        resultado_textual = funcao(df, **parametros)
                        
                        # 5. Extrair insights técnicos
                        novos_insights = tools.extrair_insights_do_resultado(
                            ferramenta_escolhida, 
                            df, 
                            parametros
                        )
                        st.session_state.insights_descobertos.extend(novos_insights)
                        
                        # 6. Analisar resultado com LLM
                        analise_llm = tools.analisar_resultado_com_llm(
                            ferramenta_escolhida,
                            resultado_textual,
                            novos_insights,
                            prompt_usuario,
                            df
                        )
                        
                        # 7. Mostrar análise conversacional
                        st.markdown("---")
                        st.markdown("### 💬 Análise")
                        st.write(analise_llm)
                        
                        # 8. Salvar no histórico
                        resposta_completa = f"[Análise: {ferramenta_escolhida}]\n\n{analise_llm}"
                        st.session_state.messages.append({"role": "assistant", "content": resposta_completa})
                        
                        # 9. Registrar análise
                        st.session_state.analises_realizadas.append({
                            "tipo": tipo_pergunta,
                            "ferramenta": ferramenta_escolhida,
                            "pergunta": prompt_usuario,
                            "insights": novos_insights,
                            "timestamp": datetime.now().isoformat()
                        })

                except Exception as e:
                    error_msg = f"Ops, encontrei um erro ao processar sua solicitação: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.session_state.pensamento_agentes.append(f"Erro na execução: {e}")
                    
                    # Mostra traceback no modo debug
                    if st.session_state.get('debug_mode', False):
                        import traceback
                        st.code(traceback.format_exc())