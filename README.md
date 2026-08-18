# Análise de Desempenho de Modelos Léxicos e Probabilísticos na Classificação de Sentimentos

Trabalho de Iniciação Científica — Centro de Inteligência Artificial e Ciência de Dados  
Pontifícia Universidade Católica do Rio Grande do Sul (PUCRS)  
Autor: Rafael Andrés Riveros Radomsky  
Orientador: Prof. Rodrigo Goulart

## Sobre

Este repositório contém o código dos experimentos realizados para o estudo de limitações em modelos tradicionais de classificação de sentimentos utilizando a biblioteca NLTK.

Foram testados dois modelos:
- **VADER** — modelo léxico baseado em dicionário e regras empíricas
- **Naive Bayes** — classificador probabilístico treinado sobre o corpus Subjectivity (Pang & Lee, 2004)

## Experimentos

| # | Descrição |
|---|-----------|
| 1 | Comportamento do VADER em casos básicos, negação, sarcasmo, emojis e português |
| 2 | Avaliação do Naive Bayes com métricas de acurácia, precisão, recall e F-measure |
| 3 | Testes de estresse por categoria de problema: sarcasmo, negação composta, sentimento misto, intensidade gradual e contexto de domínio |

## Requisitos

- Python 3.x
- NLTK 3.10+

## Instalação

```bash
pip install nltk
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('subjectivity'); nltk.download('punkt')"
```

## Execução

```bash
python classificador_sentimentos.py
```

## Referências

- BIRD, S.; KLEIN, E.; LOPER, E. *Natural Language Processing with Python*. O'Reilly Media, 2009.
- HUTTO, C. J.; GILBERT, E. VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text. ICWSM, 2014.
- PANG, B.; LEE, L. A Sentimental Education. ACL, 2004.
