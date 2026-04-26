from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/new-scan")
def new_scan():
    return render_template("new_scan.html")

@app.route("/history")
def history():
    return render_template("history.html")

@app.route("/results")
def results():
    return render_template("results.html")

if __name__ == "__main__":
    app.run(debug=True)
