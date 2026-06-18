# Skill: Relatório Técnico e Visualização de Resultados

Esta skill auxilia o Antigravity CLI (agy) na persistência de métricas, geração de gráficos e estruturação do relatório científico.

## Instruções para o agy

1. **Persistência de Resultados**:
   - Salvar os dados agregados de métricas no arquivo `results/resultados.csv`.
   - Manter as colunas obrigatórias: `modelo, metodo, acuracia, precisao, recall, f1, auc`.
   - Formatar os valores numéricos com até 4 casas decimais.
   - Atualizar registros antigos para o mesmo par (modelo, metodo) ao invés de simplesmente concatenar ou duplicar.

2. **Geração de Gráficos**:
   - Salvar os 7 gráficos requeridos no formato PNG sob o diretório `results/graficos/`.
   - Seguir rigorosamente o padrão estético de cores (`#1A56A0`, `#E65100`, `#2E7D32`, `#6A1B9A`) e fontes definidas no projeto.

3. **Geração de Relatório Técnico**:
   - Elaborar o documento `reports/relatorio_tecnico.md` seguindo o template do artigo científico.
   - Preencher todas as 8 seções com linguagem acadêmica em português.
   - Incluir imagens dos gráficos salvos e as tabelas com os números reais obtidos.
