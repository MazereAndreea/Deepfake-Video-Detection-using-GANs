from flask import Flask

app = Flask(__name__)

@app.route("/")
def index():
    return '''
            <html>
                <body>
                    <h1>Generated Image</h1>
                    <img src="/generated-image" alt="Generated Image">
                </body>
            </html>
        '''

if __name__ == '__main__':
    app.run()