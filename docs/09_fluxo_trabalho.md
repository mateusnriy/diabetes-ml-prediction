# 09 — Fluxo de Trabalho com o Antigravity CLI

> Guia operacional completo para usar o `agy` no desenvolvimento deste projeto.
> Execute os passos nesta ordem exata.

---

## Pré-requisitos

Antes de começar, certifique-se de ter:

- [X] Conta Google (Pro, Ultra ou gratuita)
- [X] Antigravity CLI instalado (`agy --version` funciona no terminal)
- [X] Git instalado e configurado
- [X] Python 3.10+ instalado
- [X] Arquivo `diabetes.csv` baixado do Kaggle

---

## Fase 0 — Instalação do agy

```bash
# Linux / macOS
curl -fsSL https://antigravity.google/cli/install.sh | bash

# Adicionar ao PATH (se necessário)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Verificar
agy --version

# Autenticar (abre browser automaticamente)
agy
# → Escolher "Google OAuth"
# → Clicar no link exibido
# → Aceitar termos
```

---

## Fase 1 — Inicialização do Repositório

### Passo 1.1 — Criar repositório no GitHub

1. Acesse github.com → "New repository"
2. Nome: `diabetes-ml-prediction`
3. Visibilidade: **Public** (obrigatório para entrega)
4. Adicionar `README.md` ✅
5. `.gitignore`: Python ✅
6. Licença: MIT ✅
7. Clicar em "Create repository"

### Passo 1.2 — Clonar localmente

```bash
git clone https://github.com/SEU_USUARIO/diabetes-ml-prediction.git
cd diabetes-ml-prediction
```

### Passo 1.3 — Copiar a pasta de documentação

```bash
# Copiar os arquivos docs/ para dentro do repositório clonado
cp -r /caminho/para/docs/ ./docs/

# Confirmar que os arquivos estão lá
ls docs/
# → 00_AGENTS.md  01_requisitos.md  02_arquitetura.md  ...
```

### Passo 1.4 — Criar o AGENTS.md na raiz

```bash
cp docs/00_AGENTS.md ./AGENTS.md
```

### Passo 1.5 — Adicionar o dataset (sem versionar)

```bash
mkdir data
cp /caminho/para/diabetes.csv data/
# Verificar que .gitignore já inclui data/*.csv
cat .gitignore | grep "*.csv"
```

---

## Fase 2 — Geração da Estrutura com o agy

### Passo 2.1 — Iniciar sessão do agy na pasta do projeto

```bash
cd diabetes-ml-prediction
agy
```

### Passo 2.2 — Criar estrutura de pastas e arquivos base

Cole este prompt no agy:

```
Leia o arquivo docs/03_estrutura_pastas.md e docs/04_convencoes.md.

Em seguida, crie toda a estrutura de diretórios e arquivos-base do projeto
conforme especificado em docs/03_estrutura_pastas.md.

Para cada arquivo marcado com [GERAR], crie o arquivo com conteúdo
apropriado seguindo as convenções de docs/04_convencoes.md.

Arquivos a criar:
- requirements.txt (com versões exatas conforme docs/03_estrutura_pastas.md)
- .gitignore (conforme especificação)
- data/README.md
- results/graficos/.gitkeep
- src/__init__.py
- .agents/skills/ (criar pasta com .gitkeep)

Não crie notebooks nem módulos src/ ainda. Apenas a estrutura base.
```

### Passo 2.3 — Verificar e commitar estrutura

```bash
# Verificar o que foi criado
tree -a -I '__pycache__|.git|*.pyc'

# Adicionar ao Git
git add .
git commit -m "chore: inicializa estrutura do projeto conforme documentação"
git push origin main
```

---

## Fase 3 — Geração dos Módulos `src/`

### Passo 3.1 — Gerar config.py

```
Leia docs/02_arquitetura.md, especificamente a seção "src/config.py".
Leia também docs/04_convencoes.md para seguir as convenções de código.

Crie o arquivo src/config.py com todas as constantes do projeto conforme
especificado. Inclua docstring completa no topo do arquivo.
```

### Passo 3.2 — Gerar data_loader.py

```
Leia docs/02_arquitetura.md (seção src/data_loader.py) e docs/05_dados.md.

Crie src/data_loader.py com as funções load_dataset() e summarize_dataset()
conforme especificado. Inclua type hints, docstrings em português, validações
de arquivo inexistente e número de colunas incorreto.
```

### Passo 3.3 — Gerar preprocessor.py

```
Leia docs/02_arquitetura.md (seção src/preprocessor.py) e docs/05_dados.md
completo (especialmente seção 5 — pré-processamento detalhado).

Crie src/preprocessor.py com todas as funções especificadas. A função
run_full_pipeline() deve executar todas as etapas na ordem correta e retornar
um dicionário com X_train, X_test, y_train, y_test e o scaler.

Siga rigorosamente a ordem: zeros→NaN, mediana, outliers IQR, seleção de
features, normalização MinMaxScaler, split 85/15 estratificado.
```

### Passo 3.4 — Gerar models.py

```
Leia docs/02_arquitetura.md (seção src/models.py) e docs/06_modelos.md
(seção 1 — Modelos Clássicos).

Crie src/models.py com as funções especificadas. A função get_classical_models()
deve retornar o dicionário CLASSICAL_MODELS com os 7 modelos configurados.
```

### Passo 3.5 — Gerar neural_network.py

```
Leia docs/02_arquitetura.md (seção src/neural_network.py) e docs/06_modelos.md
(seção 2 — Redes Neurais).

Crie src/neural_network.py com as funções build_nn1(), build_nn2(), build_nn3()
e run_all_nn_experiments(). Cada função build_nn*() deve criar e compilar o
modelo Keras conforme arquitetura especificada.
```

### Passo 3.6 — Gerar evaluator.py e visualizer.py

```
Leia docs/02_arquitetura.md (seções evaluator e visualizer), docs/07_experimentos.md
(seções 3 e 4) e docs/04_convencoes.md (seção 5 — Convenções de Gráficos).

Crie src/evaluator.py com as funções de métricas e persistência de resultados.
Crie src/visualizer.py com todas as funções de geração de gráficos especificadas
em docs/01_requisitos.md (RF-07). Todo gráfico deve ser salvo em PNG em
results/graficos/ antes de ser exibido.
```

### Passo 3.7 — Commitar módulos

```bash
git add src/
git commit -m "feat(src): implementa módulos de carregamento, preprocessamento, modelos e avaliação"
git push origin main
```

---

## Fase 4 — Geração dos Notebooks

### Passo 4.1 — Notebook 01: EDA

```
Leia docs/05_dados.md (seção 7 — Análise Exploratória Esperada) e
docs/04_convencoes.md (seção 2 — Convenções de Células de Notebook).

Crie notebooks/01_eda.ipynb com:
- Célula 1: markdown de título conforme padrão
- Célula 2: instalação de dependências
- Célula 3: imports e configuração de caminhos
- Células de EDA: responder cada uma das 6 perguntas da seção 7 de docs/05_dados.md
- Última célula: markdown com resumo

O notebook deve ser completamente autocontido e executável no Google Colab.
```

### Passo 4.2 — Notebook 02: Pré-processamento

```
Leia docs/05_dados.md (seção 5 completa) e docs/02_arquitetura.md (padrão de importação).

Crie notebooks/02_preprocessamento.ipynb implementando o pipeline completo.
Para cada etapa, exibir: shape antes, operação realizada, shape depois.
Ao final, exibir estatísticas de X_train e X_test normalizados.
```

### Passo 4.3 — Notebook 03: Modelos Clássicos

```
Leia docs/06_modelos.md (seção 1) e docs/07_experimentos.md (EXP-01 e EXP-02).

Crie notebooks/03_modelos_classicos.ipynb que:
1. Execute EXP-01: K-fold para os 7 modelos, salvando resultados
2. Execute EXP-02: Split 85/15 para os 7 modelos, salvando resultados
3. Gere os gráficos 01_acuracia_comparativa.png e 02_f1_comparativo.png
4. Gere o gráfico 06_feature_importance.png (Random Forest)
5. Salve todos os resultados em results/resultados.csv
```

### Passo 4.4 — Notebook 04: Rede Neural

```
Leia docs/06_modelos.md (seção 2) e docs/07_experimentos.md (EXP-03 e EXP-04).

Crie notebooks/04_rede_neural.ipynb que:
1. Execute EXP-03: todas as 9 combinações (3 arq. × 3 épocas)
2. Gere o gráfico 05_learning_curves_nn.png
3. Avalie o melhor modelo (NN-2, 400 épocas)
4. Gere gráficos 03_matriz_confusao_nn.png e 04_curva_roc.png
5. Execute EXP-04: gere o gráfico 07_comparacao_artigo.png
6. Atualize results/resultados.csv com resultados de NN
```

### Passo 4.5 — Commitar notebooks

```bash
git add notebooks/
git commit -m "feat(notebooks): adiciona notebooks EDA, preprocessamento, modelos e rede neural"
git push origin main
```

---

## Fase 5 — Geração do Relatório

```
Leia docs/08_relatorio.md completo. Leia também results/resultados.csv
para obter os resultados obtidos nos experimentos.

Gere o arquivo reports/relatorio_tecnico.md preenchendo todas as seções
do template com os resultados reais obtidos nos notebooks. Onde indicado
[Inserir gráfico X], usar referência markdown ao arquivo em results/graficos/.

O relatório deve estar em português, com linguagem acadêmica e formal.
Todas as seções obrigatórias devem ser preenchidas.
```

---

## Fase 6 — Finalização e Entrega

### Passo 6.1 — Gerar README.md final

```
Leia o arquivo reports/relatorio_tecnico.md e results/resultados.csv.

Gere o README.md do repositório com:
- Badge de Python version, licença MIT e status
- Descrição do projeto em 2-3 parágrafos
- Referência ao artigo base com DOI
- Tabela de resultados obtidos (melhor modelo em destaque)
- Instruções de instalação e execução
- Estrutura do repositório
- Nomes dos integrantes do grupo
```

### Passo 6.2 — Commit e push final

```bash
git add .
git commit -m "docs: adiciona relatório técnico e README final com resultados"
git push origin main

# Verificar repositório no GitHub
echo "✓ Repositório disponível em: https://github.com/mateusnriy/diabetes-ml-prediction"
```

### Passo 6.3 — Checklist final de entrega

- [ ] Todos os 4 notebooks executam do zero sem erros
- [ ] `results/resultados.csv` com todos os modelos
- [ ] Todos os 7 gráficos em `results/graficos/`
- [ ] `reports/relatorio_tecnico.md` completo
- [ ] `README.md` atualizado com resultados finais
- [ ] Repositório público no GitHub
- [ ] Nenhum arquivo `.csv` versionado (verificar com `git ls-files data/`)
- [ ] Slides da apresentação preparados

---

## Comandos de Emergência (se algo der errado)

```bash
# Verificar o que o agy está fazendo
agy inspect

# Ver changelog da versão atual
agy changelog

# Reiniciar sessão
# Ctrl+C → agy

# Usar modelo mais capaz para tarefa difícil
agy --model "claude-opus-4-6"

# Modo não interativo para prompt longo
agy -p "$(cat docs/06_modelos.md)" 

# Ver uso de quota
# /usage  (dentro do agy)
```