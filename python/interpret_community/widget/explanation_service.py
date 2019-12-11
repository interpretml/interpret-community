from gevent.pywsgi import WSGIServer
from flask import Flask
from IPython.display import display, HTML
import threading

class LocalExplanationDash:
    __service = None
    explanations = {}
    
    class DashboardService:
        app = Flask(__name__)
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
                return "Hi {0}, is ithis it ? {1}".format(id, LocalExplanationDash.explanations[id])
            return "unknown"

    def __init__(self, explanationObject, model=None):
        if not LocalExplanationDash.__service:
            LocalExplanationDash.__service = LocalExplanationDash.DashboardService()
            self._thread = threading.Thread(target=LocalExplanationDash.__service.app.run, daemon=True)
            self._thread.start()
        LocalExplanationDash.__service.call_count += 1

        LocalExplanationDash.__service.register(LocalExplanationDash.__service.call_count, explanationObject)

        display(HTML("<iframe src='http://127.0.0.1:5000/{0}'>".format(LocalExplanationDash.__service.call_count)))
    

        
        