# AGENTS.md — Projeto: Predição de Diabetes com Machine Learning

> Arquivo principal de instruções para o Antigravity CLI (agy).
> Este arquivo deve estar na raiz do repositório e será lido automaticamente pelo agente em cada sessão.

---

## Identidade do Projeto

| Campo               | Valor                                                                 |
|---------------------|-----------------------------------------------------------------------|
| Nome                | diabetes-ml-prediction                                                |
| Tipo                | Trabalho Acadêmico — Reprodução de Artigo Científico                  |
| Disciplina          | Inteligência Artificial              |
| Artigo base         | Khanam & Foo (2021) — ICT Express, DOI: 10.1016/j.icte.2021.02.004   |
| Dataset             | Pima Indians Diabetes — UCI Machine Learning Repository               |
| Entrega             | 19/06/2026                                                            |
| Linguagem principal | Python 3.10+                                                          |

---

## Documentação de Referência

Toda documentação técnica do projeto está em `docs/`. Leia os arquivos relevantes antes de gerar código:

| Arquivo                          | Conteúdo                                              |
|----------------------------------|-------------------------------------------------------|
| `docs/01_requisitos.md`          | Requisitos funcionais e não funcionais                |
| `docs/02_arquitetura.md`         | Arquitetura do projeto e fluxo de dados               |
| `docs/03_estrutura_pastas.md`    | Organização completa do repositório                   |
| `docs/04_convencoes.md`          | Convenções de código, commits e nomenclatura          |
| `docs/05_dados.md`               | Especificação completa do dataset                     |
| `docs/06_modelos.md`             | Especificação dos modelos de ML a implementar         |
| `docs/07_experimentos.md`        | Plano de experimentos e métricas                      |
| `docs/08_relatorio.md`           | Estrutura e diretrizes do relatório técnico           |
| `docs/09_fluxo_trabalho.md`      | Fluxo de trabalho com o Antigravity CLI               |

---

## Regras Absolutas (nunca violar)

1. **Reprodutibilidade:** sempre usar `random_state=42` em todo código com aleatoriedade.
2. **Idioma:** comentários e docstrings em português; código (variáveis, funções, classes) em português.
3. **Notebooks autocontidos:** cada notebook deve rodar do zero sem depender de outro.
4. **Sem dados no repositório:** nunca fazer commit do arquivo `diabetes.csv`. Usar `.gitignore`.
5. **Resultados rastreáveis:** toda execução deve salvar métricas em `results/resultados.csv`.
6. **Células de instalação:** todo notebook deve ter célula inicial com `pip install -r requirements.txt`.
7. **Sem magic numbers:** constantes como `TEST_SIZE`, `N_SPLITS`, `EPOCHS` devem estar em `src/config.py`.

---

## Stack Tecnológica

```
Python          3.10+
pandas          2.x
numpy           1.26+
scikit-learn    1.4+
tensorflow      2.16+
keras           (incluso no tensorflow)
matplotlib      3.8+
seaborn         0.13+
jupyter         1.0+
imbalanced-learn 0.12+
shap            0.45+
```

---

## Comandos Rápidos para o agy

```bash
# Gerar estrutura completa de pastas
agy -p "Leia docs/03_estrutura_pastas.md e crie todos os diretórios e arquivos base do projeto"

# Gerar notebook de EDA
agy -p "Leia docs/05_dados.md e docs/04_convencoes.md e crie notebooks/01_eda.ipynb"

# Gerar pré-processamento
agy -p "Leia docs/05_dados.md e crie notebooks/02_preprocessamento.ipynb"

# Gerar modelos clássicos
agy -p "Leia docs/06_modelos.md e docs/07_experimentos.md e crie notebooks/03_modelos_classicos.ipynb"

# Gerar rede neural
agy -p "Leia docs/06_modelos.md e crie notebooks/04_rede_neural.ipynb"

# Gerar código modular
agy -p "Leia docs/02_arquitetura.md e crie todos os módulos em src/"

# Gerar relatório
agy -p "Leia docs/08_relatorio.md e gere o relatório em reports/relatorio_tecnico.md"
```