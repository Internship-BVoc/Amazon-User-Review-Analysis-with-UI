import flask
from flask_ngrok import run_with_ngrok
from flask import render_template, request, url_for, flash, redirect
from nlp_mod import sentimentAn, pred, keyword, mainf

app = flask.Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
run_with_ngrok(app)


@app.route("/")
def index():
    global revtxt
    revtxt = request.args.get('revent')
    return render_template("main.html", revtxt=revtxt)


@app.route("/sentiment/", methods=['GET', 'POST'])
def sentiment():
    sent_score, key_l = sentimentAn(revtxt)
    return render_template("main.html", revtxt=revtxt, score=sent_score, kd=key_l, le=len(key_l))


@app.route("/category/", methods=['GET', 'POST'])
def category():
    cat_arr = pred(revtxt)
    return render_template("main.html", ar=cat_arr[:5], revtxt=revtxt)


@app.route("/keyword/", methods=['GET', 'POST'])
def key():
    _, key_arr = keyword(revtxt)
    return render_template("main.html", kar=key_arr, revtxt=revtxt)


@app.route("/entity/", methods=['GET', 'POST'])
def en():
    _, _, comar = mainf(revtxt)
    print(comar)
    return render_template("main.html", comar=comar, revtxt=revtxt)

@app.after_request
def add_header(response):
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'no-store'
    return response

if __name__ == '__main__':
    app.run()
