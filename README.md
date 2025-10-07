# 🤖 Agente Analista de Dados - EDA

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Sistema inteligente de análise exploratória de dados baseado em agentes de IA autônomos. Permite análise conversacional de datasets CSV através de linguagem natural, com geração automática de gráficos, detecção de padrões e síntese de insights.

**🔗 [Demo Online](https://agente-cientista-dados-i2a2.streamlit.app/)** 

---

## 📋 Índice

- [Funcionalidades](#-funcionalidades)
- [Arquitetura](#️-arquitetura)
- [Instalação](#-instalação)
- [Uso](#-uso)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Tecnologias](#-tecnologias)
- [Exemplos](#-exemplos)
- [Limitações](#-limitações)
- [Autor](#-autor)

---

## ✨ Funcionalidades

### Análises Disponíveis

| Análise | Descrição | Visualização |
|---------|-----------|--------------|
| **Tipos de Dados** | Identifica colunas numéricas e categóricas automaticamente | Tabela |
| **Estatísticas Descritivas** | Média, mediana, desvio padrão, assimetria, curtose | Tabela + Métricas |
| **Distribuições** | Análise de distribuição com histogramas e interpretação de assimetria | Gráfico + Análise |
| **Correlações** | Matriz de correlação com identificação de relações fortes/moderadas | Heatmap |
| **Detecção de Outliers** | Z-Score com análise de impacto e recomendações | Tabela |
| **Comparações** | Boxplots para análise entre grupos | Gráfico + Estatísticas |
| **Scatter Plots** | Análise de relações bivariadas | Gráfico + Correlação |
| **Distribuição de Classes** | Análise de balanceamento para variáveis categóricas | Gráfico + Métricas |

### Capacidades do Sistema

🧠 **4 Agentes Especializados**
- **Classificador**: Identifica automaticamente o tipo de pergunta
- **Roteador**: Seleciona a ferramenta adequada
- **Extrator**: Identifica parâmetros e colunas mencionadas
- **Sintetizador**: Gera conclusões baseadas em todo histórico de análises

💬 **Análise Conversacional**
- Respostas em linguagem natural usando Groq LLM (llama-3.1-8b-instant)
- Interpretação contextual dos resultados
- Sugestões proativas de próximas análises

📊 **8 Ferramentas de Análise**
- Cada ferramenta retorna dados estruturados + visualizações
- Extração automática de insights após cada análise
- Memória persistente de descobertas

🎨 **Interface Intuitiva**
- Chat em tempo real estilo ChatGPT
- Upload simples de CSV
- Visualização do raciocínio dos agentes (modo debug)
- Sidebar com progresso e insights acumulados

---

## 🏗️ Arquitetura

### Fluxo de Execução

```
Usuário faz pergunta
        ↓
[Classificador] → Identifica tipo (correlação, outliers, conclusões, etc)
        ↓
[Roteador] → Seleciona ferramenta apropriada
        ↓
[Extrator] → Identifica colunas e parâmetros mencionados
        ↓
[Execução] → Ferramenta gera análise + visualização
        ↓
[Extração de Insights] → Descobre padrões automaticamente
        ↓
[Groq LLM] → Interpreta resultados em linguagem natural
        ↓
Usuário recebe resposta conversacional + gráficos
```

### Agentes

**1. Classificador de Perguntas**
- Analisa pergunta do usuário
- Identifica intenção (8 tipos: tipos_dados, estatisticas, distribuicao, correlacao, outliers, comparacao, scatter, conclusoes)
- Baseado em keywords com priorização

**2. Roteador Inteligente**
- Mapeia tipo de pergunta → ferramenta
- Validação de pré-requisitos (ex: colunas numéricas para correlação)

**3. Extrator de Parâmetros**
- Identifica colunas mencionadas na pergunta
- Usa heurísticas inteligentes para defaults
- Diferencia categóricas vs numéricas

**4. Agente de Conclusões**
- Sintetiza todo histórico de análises
- Usa Groq LLM para gerar síntese estruturada
- 4 seções: características, padrões, qualidade, recomendações

---

## 🚀 Instalação

### Pré-requisitos

- Python 3.8+
- Conta Groq (gratuita) - [Criar conta](https://console.groq.com)

### Passos

```bash
# 1. Clone o repositório
git clone https://github.com/jaq1997/agente-i2a2-dados.git
cd agente-i2a2-dados

# 2. Crie ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate   # Windows

# 3. Instale dependências
pip install -r requirements.txt

# 4. Configure API Key
# Crie arquivo .streamlit/secrets.toml
mkdir .streamlit
echo 'GROQ_API_KEY = "sua_chave_aqui"' > .streamlit/secrets.toml

# 5. Execute
streamlit run app.py
```

Acesse: http://localhost:8501

---

## 💻 Uso

### Passo a Passo

1. **Upload**: Faça upload de um arquivo CSV
2. **Pergunte**: Digite sua pergunta em linguagem natural
3. **Analise**: Veja gráficos, tabelas e interpretação
4. **Explore**: Faça novas perguntas baseadas nos insights
5. **Conclua**: Peça "Quais suas conclusões?" para síntese final

### Exemplos de Perguntas

**Exploração Inicial**
```
Quais são os tipos de dados?
Mostre as estatísticas descritivas
Quantas linhas e colunas tem o dataset?
```

**Análise de Distribuições**
```
Mostre a distribuição de Age
Gere um histograma de Amount
A variável X está normalmente distribuída?
```

**Detecção de Padrões**
```
Como as variáveis se relacionam?
Existe correlação entre vendas e publicidade?
Mostre o mapa de calor
```

**Identificação de Problemas**
```
Existem valores atípicos?
Quais colunas têm mais outliers?
Há valores nulos?
```

**Comparações**
```
Compare Age por Survived
Mostre a diferença de salary entre departamentos
Relacione Time com Amount
```

**Classes & Balanceamento**
```
Qual a proporção da classe alvo?
O dataset está balanceado?
Quantas fraudes existem?
```

**Síntese (Crítico!)**
```
Quais suas conclusões sobre os dados?
O que você descobriu analisando?
Resuma os principais insights
```

---

## 📁 Estrutura do Projeto

```
agente-i2a2-dados/
├── app.py                    # Aplicação principal Streamlit
│                             # - Sistema de chat
│                             # - Gerenciamento de estado
│                             # - Orquestração de agentes
│
├── tools.py                  # Ferramentas e agentes
│                             # - 8 ferramentas de análise
│                             # - 4 agentes (classificador, roteador, extrator, conclusões)
│                             # - Extração automática de insights
│                             # - Integração com Groq LLM
│
├── requirements.txt          # Dependências Python
├── README.md                 # Este arquivo
└── .streamlit/
    └── secrets.toml          # Configurações (não versionado)
```

### Detalhes dos Módulos

**app.py (220 linhas)**
- Interface Streamlit com chat
- Estado da sessão (mensagens, análises, insights)
- Sidebar com informações e progresso
- Modo debug para desenvolvimento

**tools.py (650 linhas)**
- Ferramentas: `mostrar_tipos_de_dados`, `mostrar_estatisticas_descritivas`, `gerar_histograma`, `gerar_mapa_de_calor_correlacao`, `gerar_boxplot`, `encontrar_outliers_zscore`, `gerar_scatter_plot`, `analisar_distribuicao_classes`
- Agentes: `classificar_pergunta`, `agente_roteador_inteligente`, `agente_extrator_parametros_melhorado`, `agente_conclusoes_completo`
- Helpers: `obter_info_completa_dataframe`, `extrair_insights_do_resultado`, `analisar_resultado_com_llm`

---

## 🛠️ Tecnologias

| Camada | Tecnologia | Versão | Uso |
|--------|-----------|--------|-----|
| **Interface** | Streamlit | latest | UI web interativa |
| **LLM** | Groq | latest | Análises conversacionais (llama-3.1-8b-instant) |
| **Dados** | Pandas | latest | Manipulação de datasets |
| **Viz Estática** | Matplotlib, Seaborn | latest | Gráficos |
| **Stats** | Scipy, NumPy | latest | Estatísticas avançadas |
| **ML** | Scikit-learn | latest | Detecção de outliers |

### Dependências Completas

```txt
streamlit
pandas
numpy
scipy
matplotlib
seaborn
plotly
scikit-learn
groq
```

---

## 📊 Exemplos

### Caso 1: Análise de Fraude (Credit Card)

**Dataset**: 284,807 transações, 31 features (Time, V1-V28, Amount, Class)

**Perguntas feitas:**
1. "Quais são os tipos de dados?" → Identificou 31 numéricas
2. "Mostre as estatísticas descritivas" → Detectou assimetria em Amount
3. "Qual a proporção da classe alvo?" → 0.17% fraudes (desbalanceado)
4. "Existem valores atípicos?" → V27, V6, V20 com ~1.6% outliers cada
5. "Quais suas conclusões?" → Síntese completa gerada pelo LLM

**Insights gerados automaticamente:**
- Dataset altamente desbalanceado (ratio 581:1)
- Variáveis V1-V28 não correlacionadas (PCA efetivo)
- Amount com distribuição assimétrica à direita
- Recomendação: SMOTE ou undersampling

### Caso 2: Análise de Sobrevivência (Titanic)

**Dataset**: 891 passageiros, 12 features

**Perguntas:**
1. "Compare Age por Survived" → Boxplot mostrando diferenças
2. "Como as variáveis se relacionam?" → Heatmap de correlação
3. "Existem outliers em Fare?" → Detectados 116 outliers (12.9%)

**Descobertas:**
- Fare altamente assimétrico (cauda longa)
- Idade não é fator determinante isolado
- Pclass correlacionado negativamente com Survived

---

## ⚠️ Limitações Conhecidas

1. **Perguntas Diretas**: Funciona melhor com menções explícitas de colunas
   - ✅ "Mostre a distribuição de Age"
   - ⚠️ "Mostre a distribuição da primeira variável"

2. **Tamanho de Dataset**: Recomendado até 500k linhas para performance ideal

3. **Tipos de Arquivo**: Apenas CSV suportado

4. **Dependência de API**: Requer conexão com Groq para análises conversacionais
   - Fallbacks implementados para respostas básicas

5. **Idioma**: Sistema otimizado para português brasileiro

### Próximos Passos

- [ ] Suporte a Excel (.xlsx)
- [ ] Análises de séries temporais
- [ ] Exportação de relatórios em PDF
- [ ] Sugestões proativas de análises

---

## 🔧 Configuração Avançada

### Variáveis de Ambiente

```bash
# .streamlit/secrets.toml
GROQ_API_KEY = "gsk_..."
```

### Modo Debug

Ative o checkbox "Modo Debug" na sidebar para ver:
- Tipo de pergunta identificado
- Ferramenta selecionada
- Parâmetros extraídos
- Chamadas ao LLM
- Tempo de execução

---

## 🚨 Troubleshooting

### Erro: "GROQ_API_KEY não configurada"
**Solução**: Configure em `.streamlit/secrets.toml` ou obtenha chave em https://console.groq.com/keys

### Análise travando
**Soluções**:
1. Verifique conexão com internet
2. Recarregue a página (F5)
3. Tente pergunta mais simples
4. Ative Modo Debug

### CSV não carrega
**Soluções**:
1. Verifique encoding (deve ser UTF-8)
2. Confirme que não há colunas duplicadas
3. Teste com CSV menor

### Respostas genéricas
**Soluções**:
1. Mencione nome das colunas explicitamente
2. Use perguntas dos exemplos como base
3. Evite perguntas muito abstratas

---

## 👤 Autor

**Jaqueline Aguiar Kunzel**

Desenvolvido como parte do curso de desenvolvimento de Agentes Autônomos - I2A2 (2025)

- 🔗 GitHub: @jaq1997 (https://github.com/jaq1997)
- 📧 Email: aguiarjaqueline1997@gmail.com
- 💼 LinkedIn: https://www.linkedin.com/in/jaqueline-aguiar/

---

## 📝 Licença

Este projeto foi desenvolvido para fins educacionais como parte do curso de desenvolvimento de Agentes Autônomos - I2A2 (2025).

---

**⭐ Se este projeto foi útil, considere dar uma estrela!**

---

Última atualização: Outubro 2025
