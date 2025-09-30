def extrair_partes(texto):
    """Separa explicação de código"""
    if '```python' in texto:
        partes = texto.split('```python')
        explicacao = partes[0].strip()
        
        if len(partes) > 1:
            codigo = partes[1].split('```')[0].strip()
            return explicacao, codigo
    
    return texto.strip(), None

import pandas as pd
import requests
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import re
from datetime import datetime

print("="*70)
print("🤖 AGENTE DE ANÁLISE EXPLORATÓRIA DE DADOS (EDA)")
print("="*70)

# Leitura interativa do CSV
caminho_csv = input("\n📂 Digite o caminho completo do CSV: ")
if not os.path.isfile(caminho_csv):
    print("❌ Arquivo não encontrado. Verifique o caminho e tente novamente.")
    exit()

print("\n⏳ Carregando dados...")
df = pd.read_csv(caminho_csv)

print("\n✅ Dados carregados com sucesso!")
print(f"📏 Dimensões: {df.shape[0]} linhas × {df.shape[1]} colunas")
print(f"📋 Colunas: {', '.join(df.columns.tolist())}")
print("\n📊 Prévia dos primeiros registros:")
print(df.head())

# Memória do agente - armazena análises realizadas
memoria_agente = {
    "analises_realizadas": [],
    "insights_descobertos": [],
    "dados_estatisticos": {},
    "inicio_sessao": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

def salvar_na_memoria(tipo, conteudo):
    """Salva informações importantes na memória do agente"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    if tipo == "analise":
        memoria_agente["analises_realizadas"].append({
            "timestamp": timestamp,
            "descricao": conteudo
        })
    elif tipo == "insight":
        memoria_agente["insights_descobertos"].append({
            "timestamp": timestamp,
            "insight": conteudo
        })
    elif tipo == "estatistica":
        memoria_agente["dados_estatisticos"].update(conteudo)

def gerar_contexto_memoria():
    """Gera um resumo da memória para incluir no prompt"""
    if not memoria_agente["analises_realizadas"]:
        return ""
    
    resumo = "\n--- MEMÓRIA DO AGENTE (Análises Anteriores) ---\n"
    
    # Últimas 3 análises
    for analise in memoria_agente["analises_realizadas"][-3:]:
        resumo += f"[{analise['timestamp']}] {analise['descricao']}\n"
    
    # Insights descobertos
    if memoria_agente["insights_descobertos"]:
        resumo += "\nInsights importantes descobertos:\n"
        for insight in memoria_agente["insights_descobertos"][-3:]:
            resumo += f"• {insight['insight']}\n"
    
    return resumo

def limpar_codigo(codigo):
    """Remove caracteres problemáticos e corrige erros comuns"""
    # Remove caracteres unicode malformados
    codigo = re.sub(r'\\u[0-9a-fA-F]{4}', '', codigo)
    codigo = re.sub(r'\\x[0-9a-fA-F]{2}', '', codigo)
    
    # Corrige erros comuns
    codigo = codigo.replace("plt0.", "plt.")
    codigo = codigo.replace("df0.", "df.")
    
    # Remove linhas problemáticas
    linhas = codigo.split('\n')
    linhas_filtradas = []
    
    for linha in linhas:
        # Remove imports de bibliotecas não disponíveis
        if 'import' in linha.lower():
            if any(lib in linha for lib in ['seaborn', 'scipy', 'sklearn', 'sns', 'plotly', 'sns as']):
                continue
        
        # Remove tentativas de recriar DataFrame
        if 'pd.read_csv' in linha or 'StringIO' in linha or 'io.StringIO' in linha:
            continue
        
        # Remove chamadas de funções inexistentes
        if 'spearman_kendall' in linha or '.spearmanr(' in linha:
            continue
            
        linhas_filtradas.append(linha)
    
    codigo = '\n'.join(linhas_filtradas)
    
    # Adiciona plt.show() se necessário
    if 'plt.' in codigo and 'plt.show()' not in codigo:
        codigo += '\nplt.show()'
    
    return codigo

def gerar_codigo_fallback(pergunta, df):
    """Gera código automaticamente quando o LLM falha"""
    pergunta_lower = pergunta.lower()
    
    # Histograma
    if any(palavra in pergunta_lower for palavra in ['histograma', 'distribuição', 'distribuicao']):
        # Encontra coluna mencionada
        colunas = [col for col in df.columns if col.lower() in pergunta_lower]
        if colunas:
            col = colunas[0]
            return f"""plt.figure(figsize=(10, 6))
plt.hist(df['{col}'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
plt.title('Distribuição de {col}')
plt.xlabel('{col}')
plt.ylabel('Frequência')
plt.grid(True, alpha=0.3)
plt.show()"""
    
    # Scatter plot / Dispersão
    if any(palavra in pergunta_lower for palavra in ['dispersão', 'dispersao', 'scatter', 'relação', 'relacao']):
        colunas = [col for col in df.columns if col.lower() in pergunta_lower]
        if len(colunas) >= 2:
            return f"""plt.figure(figsize=(10, 6))
plt.scatter(df['{colunas[0]}'], df['{colunas[1]}'], alpha=0.5, c='steelblue')
plt.title('Dispersão: {colunas[0]} vs {colunas[1]}')
plt.xlabel('{colunas[0]}')
plt.ylabel('{colunas[1]}')
plt.grid(True, alpha=0.3)
plt.show()"""
    
    # Correlação
    if 'correlação' in pergunta_lower or 'correlacao' in pergunta_lower:
        colunas = [col for col in df.columns if col.lower() in pergunta_lower]
        if len(colunas) >= 2:
            return f"""correlacao = df[['{colunas[0]}', '{colunas[1]}']].corr()
print("\\nMatriz de Correlação:")
print(correlacao)
print(f"\\nCorrelação entre {colunas[0]} e {colunas[1]}: {{correlacao.iloc[0,1]:.4f}}")"""
        elif len(colunas) == 0:
            return """correlacao = df.select_dtypes(include=[np.number]).corr()
print("\\nMatriz de Correlação:")
print(correlacao)"""
    
    # Estatísticas
    if any(palavra in pergunta_lower for palavra in ['estatística', 'estatistica', 'describe', 'resumo']):
        colunas = [col for col in df.columns if col.lower() in pergunta_lower]
        if colunas:
            return f"""print(df['{colunas[0]}'].describe())"""
        else:
            return "print(df.describe())"
    
    # Boxplot
    if 'boxplot' in pergunta_lower or 'outlier' in pergunta_lower:
        colunas = [col for col in df.columns if col.lower() in pergunta_lower]
        if colunas:
            col = colunas[0]
            return f"""plt.figure(figsize=(10, 6))
plt.boxplot(df['{col}'].dropna())
plt.title('Boxplot de {col}')
plt.ylabel('{col}')
plt.grid(True, alpha=0.3)
plt.show()"""
    
    return None

def extrair_partes(texto):
    """Separa explicação de código"""
    if '```python' in texto:
        partes = texto.split('```python')
        explicacao = partes[0].strip()
        
        if len(partes) > 1:
            codigo = partes[1].split('```')[0].strip()
            return explicacao, codigo
    
    return texto.strip(), None

def eh_pergunta_sobre_conclusoes(pergunta):
    """Verifica se é pergunta sobre conclusões/insights do agente"""
    palavras_chave = [
        'conclus', 'insight', 'aprend', 'descobr', 'observ',
        'padrão', 'tendência', 'opinião', 'análise geral',
        'resumo', 'principais', 'importante', 'destaque'
    ]
    pergunta_lower = pergunta.lower()
    return any(palavra in pergunta_lower for palavra in palavras_chave)

def analisar_ambiguidade(pergunta, colunas_df):
    """Verifica se a pergunta é ambígua e sugere esclarecimentos"""
    pergunta_lower = pergunta.lower()
    
    # Perguntas sobre TODAS as colunas ou análise geral são VÁLIDAS
    perguntas_gerais_validas = [
        'tipos de dados', 'tipos de coluna', 'tipo das coluna',
        'quais coluna', 'quantas coluna', 'estrutura',
        'numéricas', 'categóricas', 'dtypes', 'info',
        'todas as coluna', 'todas coluna', 'cada coluna',
        'variância', 'variacao', 'desvio padrão',
        'estatísticas', 'estatistica', 'describe',
        'valores nulos', 'missing', 'correlação geral',
        'matriz de correlação', 'correlacao geral'
    ]
    
    if any(termo in pergunta_lower for termo in perguntas_gerais_validas):
        return {"ambigua": False}
    
    # Se menciona "todas", "cada", "geral" - permite
    if any(palavra in pergunta_lower for palavra in ['todas', 'todos', 'cada', 'geral', 'comparar']):
        return {"ambigua": False}
    
    # Detecta perguntas REALMENTE muito vagas (menos de 3 palavras e sem contexto)
    perguntas_vagas = ['analise', 'mostre', 'me fale']
    if any(vaga in pergunta_lower for vaga in perguntas_vagas) and len(pergunta.split()) <= 2:
        return {
            "ambigua": True,
            "motivo": "muito_vaga",
            "sugestoes": [
                "Qual aspecto específico você quer analisar?",
                "Você quer ver: distribuição, correlação, outliers ou estatísticas?"
            ]
        }
    
    # Detecta menção a "uma coluna" ou "a coluna" sem especificar qual
    if (('uma coluna' in pergunta_lower or 'a coluna' in pergunta_lower or 'essa coluna' in pergunta_lower) 
        and not any(col.lower() in pergunta_lower for col in colunas_df)):
        return {
            "ambigua": True,
            "motivo": "coluna_nao_especificada",
            "sugestoes": [
                f"Colunas disponíveis: {', '.join(colunas_df[:10])}",
                "Qual coluna específica você quer analisar?"
            ]
        }
    
    # Detecta "correlação" entre duas colunas específicas incompleta
    if 'correlação entre' in pergunta_lower or 'correlacao entre' in pergunta_lower:
        colunas_mencionadas = [col for col in colunas_df if col.lower() in pergunta_lower]
        if len(colunas_mencionadas) == 1:
            colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
            return {
                "ambigua": True,
                "motivo": "correlacao_incompleta",
                "sugestoes": [
                    f"Correlação de '{colunas_mencionadas[0]}' com qual outra coluna?",
                    f"Colunas numéricas disponíveis: {', '.join(colunas_numericas[:8])}"
                ]
            }
    
    # Detecta "faça um gráfico" sem mais informações
    if ('faça um gráfico' in pergunta_lower or 'faca um grafico' in pergunta_lower) and len(pergunta.split()) <= 4:
        return {
            "ambigua": True,
            "motivo": "grafico_incompleto",
            "sugestoes": [
                "Que tipo de gráfico e de qual coluna?",
                "Ex: 'Faça um histograma de Amount' ou 'Gráfico de dispersão entre Time e Amount'"
            ]
        }
    
    return {"ambigua": False}

# Análise inicial automática dos dados
print("\n🔍 Realizando análise inicial dos dados...\n")
analise_inicial = {
    "tipos_colunas": df.dtypes.to_dict(),
    "valores_nulos": df.isnull().sum().to_dict(),
    "linhas_totais": len(df),
    "colunas_numericas": df.select_dtypes(include=[np.number]).columns.tolist(),
    "colunas_categoricas": df.select_dtypes(include=['object']).columns.tolist()
}
salvar_na_memoria("estatistica", analise_inicial)
print("✅ Análise inicial concluída e armazenada na memória.\n")

# Loop de perguntas
print("="*70)
print("💬 Você pode fazer perguntas sobre os dados.")
print("💡 Exemplos:")
print("   - Qual a distribuição da variável Amount?")
print("   - Existe correlação entre Time e Amount?")
print("   - Quais são suas conclusões sobre os dados até agora?")
print("   - Existem outliers em Amount?")
print("="*70)

while True:
    pergunta = input("\n🗣️  Sua pergunta (ou 'sair'): ")
    
    if pergunta.lower() in ['sair', 'exit', 'quit']:
        print("\n" + "="*70)
        print("📊 RESUMO DA SESSÃO")
        print("="*70)
        print(f"Total de análises realizadas: {len(memoria_agente['analises_realizadas'])}")
        print(f"Total de insights descobertos: {len(memoria_agente['insights_descobertos'])}")
        print("\n👋 Encerrando. Até logo!")
        break
    
    if not pergunta.strip():
        continue
    
    # NOVO: Verifica ambiguidade antes de processar
    analise_ambig = analisar_ambiguidade(pergunta, df.columns.tolist())
    
    if analise_ambig["ambigua"]:
        print("\n🤔 " + "="*68)
        print("Hmm, preciso de mais informações para responder bem!")
        print("="*68)
        
        for sugestao in analise_ambig["sugestoes"]:
            print(f"\n💡 {sugestao}")
        
        print("\n" + "="*68)
        print("Por favor, reformule sua pergunta com mais detalhes.")
        print("="*68)
        continue
    
    # Se for pergunta sobre conclusões, responde da memória
    if eh_pergunta_sobre_conclusoes(pergunta):
        print("\n🧠 Consultando memória do agente...\n")
        print("="*70)
        print("💭 CONCLUSÕES E INSIGHTS DO AGENTE")
        print("="*70)
        
        if memoria_agente["analises_realizadas"]:
            print(f"\n📈 Análises realizadas: {len(memoria_agente['analises_realizadas'])}")
            print("\n🔍 Principais descobertas:")
            
            for i, insight in enumerate(memoria_agente["insights_descobertos"], 1):
                print(f"\n{i}. {insight['insight']}")
            
            if not memoria_agente["insights_descobertos"]:
                print("\n⚠️  Ainda não registrei insights específicos.")
                print("Continue fazendo análises para que eu possa formar conclusões!")
        else:
            print("\n⚠️  Ainda não realizei análises suficientes para ter conclusões.")
            print("Faça perguntas sobre os dados para que eu possa analisá-los!")
        
        print("\n" + "="*70)
        continue
    
    # Gerar contexto com memória
    contexto_dados = df.head(5).to_string()
    contexto_memoria = gerar_contexto_memoria()
    
    # Prompt otimizado com memória e instruções para conclusões
    prompt = f"""Você é um agente de análise de dados inteligente e reflexivo.

DADOS DISPONÍVEIS:
- DataFrame 'df' JÁ CARREGADO na memória
- Total de linhas: {len(df)}
- Colunas: {', '.join(df.columns.tolist())}
- Tipos: {len(analise_inicial['colunas_numericas'])} numéricas, {len(analise_inicial['colunas_categoricas'])} categóricas

PRÉVIA DOS DADOS:
{contexto_dados}

{contexto_memoria}

BIBLIOTECAS DISPONÍVEIS:
✅ pandas (pd), numpy (np), matplotlib.pyplot (plt)
❌ NÃO use: seaborn, scipy, sklearn, plotly

INSTRUÇÕES CRÍTICAS:
1. Você DEVE SEMPRE gerar código Python executável
2. TODO código deve estar entre ```python e ``` - SEM EXCEÇÕES!
3. NÃO escreva apenas explicações - SEMPRE inclua código funcional
4. Seja DIRETO: 2-3 linhas de explicação + código completo

FORMATO OBRIGATÓRIO:
Explicação breve (2-3 linhas)

```python
# Código completo aqui
plt.figure(figsize=(10, 6))
plt.hist(df['Amount'], bins=50)
plt.title('Título')
plt.show()
```

Conclusão (1 linha)

EXEMPLOS COMPLETOS:

Histograma:
```python
plt.figure(figsize=(10, 6))
plt.hist(df['Amount'], bins=50, color='steelblue', edgecolor='black')
plt.title('Distribuição de Amount')
plt.xlabel('Valor')
plt.ylabel('Frequência')
plt.grid(True, alpha=0.3)
```

Scatter:
```python
plt.figure(figsize=(10, 6))
plt.scatter(df['Time'], df['Amount'], alpha=0.5)
plt.title('Time vs Amount')
plt.xlabel('Time')
plt.ylabel('Amount')
plt.grid(True, alpha=0.3)
```

Correlação:
```python
corr = df[['Time', 'Amount']].corr()
print(corr)
```

IMPORTANTE: Se a pergunta pede gráfico/análise, você DEVE gerar código entre ```python ```!

EXEMPLOS DE CÓDIGO CORRETO:
- Correlação: df[['col1', 'col2']].corr()
- Dispersão: plt.scatter(df['x'], df['y'])
- Histograma: plt.hist(df['col'], bins=30)
- Estatísticas: df['col'].describe()

PERGUNTA DO USUÁRIO: {pergunta}

Sua resposta (explicação + código se necessário + conclusão):"""

    print("\n🤖 Analisando...\n")
    
    try:
        resposta = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3",
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.6,
                    "num_predict": 700,
                    "top_p": 0.9,
                }
            },
            stream=True,
            timeout=120
        )
        
        saida = ""
        for linha in resposta.iter_lines():
            if linha:
                try:
                    dados = json.loads(linha.decode("utf-8"))
                    if "response" in dados:
                        saida += dados["response"]
                        if len(saida) > 5000:
                            break
                except json.JSONDecodeError:
                    continue
        
        if not saida.strip():
            print("⚠️  Resposta vazia. Tente reformular.")
            continue
        
        # Extrair explicação e código
        explicacao, codigo = extrair_partes(saida)
        
        # Mostra explicação
        if explicacao:
            print("💬 " + "="*68)
            print(explicacao)
            print("="*68 + "\n")
            
            # Salvar na memória
            salvar_na_memoria("analise", pergunta[:80])
        
        # Se não gerou código mas pergunta precisa, usa fallback
        if not codigo:
            print("⚙️  Gerando código automaticamente...\n")
            codigo = gerar_codigo_fallback(pergunta, df)
            
            if codigo:
                print("💡 O modelo não gerou código, usando solução automática.\n")
        
        # Executar código se houver
        if codigo:
            codigo_limpo = limpar_codigo(codigo)
            
            if codigo_limpo.strip():
                print("🔧 Executando código...\n")
                
                try:
                    # Executar código
                    exec(codigo_limpo, {
                        "df": df,
                        "plt": plt,
                        "pd": pd,
                        "np": np,
                        "print": print
                    })
                    
                    print("\n✅ Código executado com sucesso!")
                    
                    # Tentar extrair insight da explicação
                    if len(explicacao) > 50:
                        # Pega última frase como possível insight
                        frases = explicacao.split('.')
                        if len(frases) > 1:
                            possivel_insight = frases[-2].strip()
                            if len(possivel_insight) > 20:
                                salvar_na_memoria("insight", possivel_insight)
                    
                except Exception as e:
                    print(f"\n⚠️  Erro: {type(e).__name__}: {str(e)}")
                    print("💡 A explicação acima ainda é válida!\n")
        
    except requests.exceptions.RequestException as e:
        print(f"\n⚠️  Erro de conexão: {e}")
        print("Verifique se Ollama está rodando: ollama serve\n")
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrompido.\n")
        break
    except Exception as e:
        print(f"\n⚠️  Erro: {type(e).__name__}: {e}\n")