from flask import Flask, jsonify, request
from natasha import Doc, NewsNERTagger, NewsEmbedding, Segmenter
from pyaspeller import YandexSpeller
import re
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords
from gensim import corpora
from gensim.summarization.summarizer import summarize
import gensim
import torch

app = Flask(__name__)

model, example_texts, languages, punct, apply_te = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                                                  model='silero_te')

@app.route('/', methods=['POST'])
def asr_with_nlp_models():
    text = request.json.get('text')
    try:
        speller = YandexSpeller()
        text = speller.spelled(text)
        text = model.enhance_text(text, 'ru')
        tags = get_tags(text)
        ner = get_text_ner(text)
        annotation = summarize(text.replace(',', '.')) if text.split('.') < 4 else summarize(text)
        return jsonify({
            'text': ner,
            'tags': tags,
            'annotation': annotation
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 401


def clean_text(doc: str):
    patterns = r'|[-;,"`\'(){}–.—«»“”\n„]'
    stopwords_ = stopwords.words("russian")
    stopwords_.extend(["это"])
    morph = MorphAnalyzer()
    doc = re.sub(patterns, '', doc)
    tokens = []
    for token in doc.split():
        if token:
            token = token.strip()
            token = morph.normal_forms(token)[0]
            if token not in stopwords_:
                tokens.append(token)
    if len(tokens) > 2:
        return tokens
    return []


def get_tags(text):
    texts = text.replace('\n', '').replace(',', '').split('.')
    texts_norm = [clean_text(elem) for elem in texts]
    id2word = corpora.Dictionary(texts_norm)
    corpus = [id2word.doc2bow(text_) for text_ in texts_norm]
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=5,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=10,
                                                passes=1,
                                                alpha='symmetric',
                                                per_word_topics=True)
    top_tags = lda_model.show_topics(formatted=False)
    tags = [topic[1][0][0] for topic in top_tags]
    tags += [topic[1][1][0] for topic in top_tags]
    return list(set(tags))


def get_text_ner(text):
    emb = NewsEmbedding()
    ner_tagger = NewsNERTagger(emb)
    segmenter = Segmenter()
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)
    text_ner = text
    for elem in doc.spans:
        t = elem.type
        text = elem.text
        text_ner = text_ner.replace(text, f'<{t}> {text} <{t}>')
    return text_ner


if __name__ == '__main__':
    app.run()
