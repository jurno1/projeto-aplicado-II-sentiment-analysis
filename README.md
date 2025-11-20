
# Projeto Aplicado II - Sentiment Analysis

Repositório gerado com base na Etapa 3 do projeto (código e descrição extraídos do documento do projeto). Veja o PDF original para referência. fileciteturn1file0

## Como usar

1. Ajuste/coloque seu dataset em `data/amazon_reviews.csv` (colunas: `text`, `label` onde label é 1 para positivo e 0 para negativo).
2. Treine modelos (opcional, se você quiser gerar os .pkl):
   ```
   python src/train_model.py
   ```
3. Rode o dashboard:
   ```
   pip install -r requirements.txt
   streamlit run src/app_streamlit.py
   ```

## Estrutura
- src/ -> código fonte (app + train)
- models/ -> modelos salvos (.pkl)
- data/ -> dataset exemplo
- docs/imagens -> imagens para relatório

