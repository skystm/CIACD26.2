# Iniciacao Cientifica - Classificacao de Sentimentos com NLTK
# Baseado em: https://www.nltk.org/howto/sentiment.html

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import mark_negation
from nltk.corpus import subjectivity
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy


# ==============================================================
# PARTE 1: VADER - analisador rapido baseado em dicionario
# ==============================================================

def experimento_vader():
    print("=" * 60)
    print("EXPERIMENTO 1: VADER SentimentIntensityAnalyzer")
    print("=" * 60)

    sia = SentimentIntensityAnalyzer()

    frases = [
        # Casos basicos
        ("VADER is smart, handsome, and funny!", "positivo esperado"),
        ("VADER is smart, handsome, and funny.",  "positivo (sem !)"),
        ("VADER is very bad.",                    "negativo esperado"),

        # Maiusculas amplificam sentimento
        ("VADER is VERY SMART",                   "positivo com CAPS"),
        ("vader is very smart",                   "positivo sem CAPS"),

        # Negacao
        ("The food is not good.",                 "negacao simples"),
        ("At least it isn't a horrible book.",    "negacao de negativo"),

        # Ironia / sarcasmo (problema classico)
        ("Yeah right, that was SO helpful.",      "sarcasmo (dificil)"),
        ("Great, another Monday.",                "sarcasmo implicito"),

        # Girias e emoticons
        ("Today sux :(",                          "giria + emoticon neg"),
        (":) and :D",                             "emoticons positivos"),

        # Frases em portugues (problema: VADER e treinado em ingles)
        ("Que dia lindo!",                        "portugues positivo"),
        ("Esse produto e terrivel.",              "portugues negativo"),
    ]

    print(f"\n{'Frase':<45} {'compound':>9} {'pos':>6} {'neu':>6} {'neg':>6}  Contexto")
    print("-" * 90)

    for frase, contexto in frases:
        scores = sia.polarity_scores(frase)
        print(
            f"{frase:<45} "
            f"{scores['compound']:>9.4f} "
            f"{scores['pos']:>6.3f} "
            f"{scores['neu']:>6.3f} "
            f"{scores['neg']:>6.3f}  "
            f"{contexto}"
        )


# ==============================================================
# PARTE 2: NaiveBayes - classificador treinado com corpus
# ==============================================================

def experimento_naive_bayes():
    print("\n" + "=" * 60)
    print("EXPERIMENTO 2: NaiveBayes com corpus Subjectivity")
    print("=" * 60)

    n_instances = 100

    subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
    obj_docs  = [(sent, 'obj')  for sent in subjectivity.sents(categories='obj')[:n_instances]]

    # Divisao treino (80%) / teste (20%)
    train_subj = subj_docs[:80]
    test_subj  = subj_docs[80:]
    train_obj  = obj_docs[:80]
    test_obj   = obj_docs[80:]

    training_docs = train_subj + train_obj
    testing_docs  = test_subj  + test_obj

    sentim_analyzer = SentimentAnalyzer()

    # Extrai palavras negadas e cria unigramas como features
    all_words_neg = sentim_analyzer.all_words(
        [mark_negation(doc) for doc, _ in training_docs]
    )
    unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)

    # Extrator customizado usando o conjunto de unigramas ja filtrado
    def feat_extractor(document):
        feats = {}
        for word in mark_negation(document):
            feats[word] = (word in unigram_feats)
        return feats

    sentim_analyzer.add_feat_extractor(feat_extractor)

    training_set = sentim_analyzer.apply_features(training_docs)
    test_set     = sentim_analyzer.apply_features(testing_docs)

    trainer = NaiveBayesClassifier.train
    classifier = sentim_analyzer.train(trainer, training_set)

    print("\nMetricas no conjunto de teste:")
    for key, value in sorted(sentim_analyzer.evaluate(test_set).items()):
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")


# ==============================================================
# PARTE 3: Experimentos sobre problemas conhecidos
# ==============================================================

def experimento_problemas():
    print("\n" + "=" * 60)
    print("EXPERIMENTO 3: Problemas classicos de classificacao")
    print("=" * 60)

    sia = SentimentIntensityAnalyzer()

    casos = {
        "Sarcasmo / Ironia": [
            "What a wonderful disaster.",
            "Oh great, the server is down again.",
            "Sure, because that always works.",
            "Absolutely love waiting an hour for cold food.",
            "Yeah, losing my keys was the highlight of my day.",
        ],
        "Negacao composta": [
            "I don't think this is not a good idea.",
            "Nobody said it wasn't fine.",
            "It's not like I didn't enjoy it at all.",
            "I can't say I don't disagree with you.",
            "She never said it was not worth trying.",
        ],
        "Sentimento misto": [
            "The movie had great effects but a terrible story.",
            "I love the design but hate the price.",
            "The hotel was beautiful but the service was awful.",
            "Great product, horrible customer support.",
            "The concert was amazing but way too loud.",
        ],
        "Intensidade gradual": [
            "The food was okay.",
            "The food was good.",
            "The food was great.",
            "The food was amazing.",
            "The food was absolutely outstanding.",
        ],
        "Contexto de dominio": [
            "This stock is killing it.",
            "The bacteria is killing the host.",
            "That pitch was sick.",
            "The patient is getting better.",
            "We need to execute the plan.",
        ],
    }

    for problema, frases in casos.items():
        print(f"\n  [{problema}]")
        for frase in frases:
            s = sia.polarity_scores(frase)
            sentimento = "POS" if s['compound'] >= 0.05 else ("NEG" if s['compound'] <= -0.05 else "NEU")
            print(f"    {sentimento} ({s['compound']:+.3f})  \"{frase}\"")


# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":
    experimento_vader()
    experimento_naive_bayes()
    experimento_problemas()

    print("\n" + "=" * 60)
    print("Todos os experimentos concluidos.")
    print("=" * 60)
