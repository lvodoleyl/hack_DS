from flask import Flask, jsonify, request
from vosk import Model, KaldiRecognizer
from natasha import Doc, NewsNERTagger, NewsEmbedding, Segmenter
import os
import wave
import re
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords
from gensim import corpora
from gensim.summarization.summarizer import summarize
import gensim

app = Flask(__name__)

# if not os.path.exists("vosk-model-small-ru-0.15"):
#     print(
#         "Please download the model from https://github.com/alphacep/kaldi-android-demo/releases and unpack as 'model-en' in the current folder.")
#     exit(1)

mock_text = """Несколько примеров диалогов из классики. Для этого я взяла пока Лермонтова "Герой нашего времени". Он отлично умел показать характер в диалогах.
1)
- Ты видел? - сказал он, крепко пожимая мне руку, - это просто ангел!
- Отчего? - спросил я с видом чистейшего простодушия.
- Разве ты не видал?
- Нет, видел: она подняла твой стакан. Если бы был тут сторож, то он
сделал бы то же самое, и еще поспешнее, надеясь получить на водку. Впрочем,
очень понятно, что ей стало тебя жалко: ты сделал такую ужасную гримасу,
когда ступил на простреленную ногу...
- И ты не был нисколько тронут, глядя на нее в эту минуту, когда душа
сияла на лице ее?..
- Нет.
Я лгал; но мне хотелось его побесить. У меня врожденная страсть
противоречить; целая моя жизнь была только цепь грустных и неудачных
противоречий сердцу или рассудку.

2) Ей стало лучше; она хотела освободиться от моей руки, но я еще крепче
обвил ее нежный мягкий стан; моя щека почти касалась ее щеки; от нее веяло
пламенем.
- Что вы со мною делаете? Боже мой!..
Я не обращал внимания на ее трепет и смущение, и губы мои коснулись ее
нежной щечки; она вздрогнула, но ничего не сказала; мы ехали сзади; никто не
видал. Когда мы выбрались на берег, то все пустились рысью. Княжна удержала
свою лошадь; я остался возле нее; видно было, что ее беспокоило мое
молчание, но я поклялся не говорить ни слова - из любопытства. Мне хотелось
видеть, как она выпутается из этого затруднительного положения.
- Или вы меня презираете, или очень любите! - сказала она наконец
голосом, в котором были слезы. - Может быть, вы хотите посмеяться надо мной,
возмутить мою душу и потом оставить.-. Это было бы так подло, так низко, что
одно предположение... о нет! не правда ли, - прибавила она голосом нежной
доверенности, - не правда ли, во мне нет ничего такого, что бы исключало
уважение? Ваш дерзкий поступок... я должна, я должна вам его простить,
потому что позволила... Отвечайте, говорите же, я хочу слышать ваш голос!..
- В последних словах было такое женское нетерпение, что я невольно
улыбнулся; к счастию, начинало смеркаться. Я ничего не отвечал.
- Вы молчите? - продолжала она, - вы, может быть, хотите, чтоб я первая
вам сказала, что я вас люблю?..
Я молчал...
- Хотите ли этого? - продолжала она, быстро обратясь ко мне... В
решительности ее взора и голоса было что-то страшное...
- Зачем? - отвечал я, пожав плечами.
Она ударила хлыстом свою лошадь и пустилась во весь дух по узкой,
опасной дороге; это произошло так скоро, что я едва мог ее догнать, и то,
когда она уж она присоединилась к остальному обществу. До самого дома она
говорила и смеялась поминутно. В ее движениях было что-то лихорадочное; На
меня не взглянула ни разу.

3) - Написали ли вы свое завещание? - вдруг спросил Вернер.
- Нет.
- А если будете убиты?..
- Наследники отыщутся сами.
- Неужели у вас нет друзей, которым бы вы хотели послать свое последнее
прости?..
Я покачал головой.
- Неужели нет на свете женщины, которой вы хотели бы оставить
что-нибудь на память?..
- Хотите ли, доктор, - отвечал я ему, - чтоб я раскрыл вам мою душу?..
Видите ли, я выжил из тех лет, когда умирают, произнося имя своей любезной и
завещая другу клочок напомаженных или ненапомаженных волос. Думая о близкой
и возможной смерти, я думаю об одном себе: иные не делают и этого. Друзья,
которые завтра меня забудут или, хуже, возведут на мой счет бог знает какие
небылицы; женщины, которые, обнимая другого, будут смеяться надо мною, чтоб
не возбудить в нем ревности к усопшему, - бог с ними! Из жизненной бури я
вынес только несколько идей - и ни одного чувства. Я давно уж живу не
сердцем, а головою.""".replace("!", '.').replace("?", ".").replace('\n', ' ')


@app.route('/', methods=['POST'])
def asr_with_nlp_models():
    file_wav = request.json.get('file_wav')
    # if file_wav is None:
    #     return jsonify({}), 410
    try:
        # wf = wave.open(file_wav, "rb")
        # if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        #     return jsonify({'error': 'Audio file must be WAV format mono PCM'}), 401
        # model = Model("vosk-model-small-ru-0.15")
        # rec = KaldiRecognizer(model, wf.getframerate())
        # text = ""
        # while True:
        #     data = wf.readframes(10000)
        #     if len(data) == 0:
        #         break
        #     if rec.AcceptWaveform(data):
        #         text += '. ' + rec.Result()['text']
        text = mock_text
        tags = get_tags(text)
        ner = get_text_ner(text)
        annotation = summarize(text)
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
    texts = text.replace('\n', '').split('.')
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
