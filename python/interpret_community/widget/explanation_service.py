from gevent.pywsgi import WSGIServer
from flask import Flask, render_template
from IPython.display import display, HTML
import threading

class LocalExplanationDash:
    __service = None
    explanations = {}
    
    class DashboardService:
        app = Flask(__name__)
        app.config['TESTING'] = True
        app.config["EXPLAIN_TEMPLATE_LOADING"] = True
        explanations = {}
        def __init__(self, base_url=None):
            self.config = {}
            self.call_count = 0

        def register(self, id, explanation):
            LocalExplanationDash.explanations[str(id)] = explanation

        @app.route('/')
        def hello():
            return "HelloWorld"

        @app.route('/<id>')
        def explanation_visual(id):
            if id in LocalExplanationDash.explanations:
                return render_template('dashboard.html', explanation=LocalExplanationDash.explanations[id])
            return "unknown"

    def __init__(self, explanationObject, model=None):
        print(__name__)
        if not LocalExplanationDash.__service:
            LocalExplanationDash.__service = LocalExplanationDash.DashboardService()
            self._thread = threading.Thread(target=LocalExplanationDash.__service.app.run, daemon=True)
            self._thread.start()
        LocalExplanationDash.__service.call_count += 1

        LocalExplanationDash.__service.register(LocalExplanationDash.__service.call_count, explanationObject)

        display(HTML("<iframe src='http://127.0.0.1:5000/{0}' width='100%' height='800px' frameBorder='0'></iframe>".format(LocalExplanationDash.__service.call_count)))
    

        
        