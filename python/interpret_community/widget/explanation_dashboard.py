from flask import Flask, render_template, request
from jinja2 import Environment, PackageLoader
from IPython.display import display, IFrame, HTML
import threading
import socket
import requests
import os
import json
import atexit
from .explanation_dashboard_input import ExplanationDashboardInput
from ._internal.constants import DatabricksInterfaceConstants
try:
    from gevent.pywsgi import WSGIServer
except ModuleNotFoundError:
    raise RuntimeError("Error: gevent package is missing, please run 'conda install gevent' or"
                       "'pip install gevent' or 'pip install interpret-community[visualization]'")

"""Explanation Dashboard Class.

:param explanation: An object that represents an explanation.
:type explanation: ExplanationMixin
:param model: An object that represents a model. It is assumed that for the classification case
    it has a method of predict_proba() returning the prediction probabilities for each
    class and for the regression case a method of predict() returning the prediction value.
:type model: object
:param dataset:  A matrix of feature vector examples (# examples x # features), the same samples
    used to build the explanation. Overwrites any existing dataset on the explanation object.
:type dataset: numpy.array or list[][]
:param true_y: The true labels for the provided dataset. Overwrites any existing dataset on the
    explanation object.
:type true_y: numpy.array or list[]
:param classes: The class names.
:type classes: numpy.array or list[]
:param features: Feature names.
:type features: numpy.array or list[]
:param port: The port to use on locally hosted service.
:type port: number
:param use_cdn: Whether to load latest dashboard script from cdn, fall back to local script if False.
:type use_cdn: boolean
"""


class ExplanationDashboard:
    service = None
    explanations = {}
    model_count = 0
    _cdn_path = "v0.1.js"
    env = Environment(loader=PackageLoader(__name__, 'templates'))
    default_template = env.get_template("inlineDashboard.html")

    class DashboardService:
        app = Flask(__name__)

        def __init__(self, port):
            self.port = port
            self.ip = '127.0.0.1'
            self.use_cdn = True
            if self.port is None:
                # Try 100 different ports
                for port in range(5000, 5100):
                    available = ExplanationDashboard.DashboardService._local_port_available(self.ip, port, rais=False)
                    if available:
                        self.port = port
                        return
                error_message = """Ports 5000 to 5100 not available.
                    Please specify an open port for use via the 'port' parameter"""
                raise RuntimeError(
                    error_message.format(port)
                )
            else:
                ExplanationDashboard.DashboardService._local_port_available(self.ip, self.port)

        def run(self):
            class devnull:
                write = lambda _: None  # noqa: E731

            server = WSGIServer((self.ip, self.port), self.app, log=devnull)
            self.app.config["server"] = server
            server.serve_forever()

            # Closes server on program exit, including freeing all sockets
            def closeserver():
                server.stop()

            atexit.register(closeserver)

        def _local_port_available(ip, port, rais=True):
            """
            Borrowed from:
            https://stackoverflow.com/questions/19196105/how-to-check-if-a-network-port-is-open-on-linux
            """
            try:
                backlog = 5
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind((ip, port))
                sock.listen(backlog)
                sock.close()
                sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
                sock.bind(("::1", port))
                sock.listen(backlog)
                sock.close()
            except socket.error:  # pragma: no cover
                if rais:
                    error_message = """Port {0} is not available.
                    Please specify another port for use via the 'port' parameter"""
                    raise RuntimeError(
                        error_message.format(port)
                    )
                else:
                    return False
            return True

        @app.route('/')
        def hello():
            return "No global list view supported at this time."

        @app.route('/<id>')
        def explanation_visual(id):
            # if there is no internet or the static file exists, use that
            # main_js='http://127.0.0.1:5000/static/index.js'
            if id in ExplanationDashboard.explanations:
                using_fallback = False
                if ExplanationDashboard.service.use_cdn:
                    try:
                        url = 'https://interpret-cdn.azureedge.net/{0}'.format(ExplanationDashboard._cdn_path)
                        r = requests.get(url)
                        if not r.ok:
                            using_fallback = True
                            url = "http://{0}:{1}/static/index.js".format(
                                ExplanationDashboard.service.ip,
                                ExplanationDashboard.service.port)
                    except Exception:
                        using_fallback = True
                        url = "http://{0}:{1}/static/index.js".format(
                            ExplanationDashboard.service.ip,
                            ExplanationDashboard.service.port)
                else:
                    url = "http://{0}:{1}/static/index.js".format(
                        ExplanationDashboard.service.ip,
                        ExplanationDashboard.service.port)
                serialized_explanation = json.dumps(ExplanationDashboard.explanations[id].dashboard_input)
                return render_template('dashboard.html',
                                       explanation=serialized_explanation,
                                       main_js=url,
                                       app_id='app_123',
                                       using_fallback=using_fallback)
            else:
                return "Unknown model id."

        @app.route('/<id>/predict', methods=['POST'])
        def predict(id):
            data = request.get_json(force=True)
            if id in ExplanationDashboard.explanations:
                return ExplanationDashboard.explanations[id].on_predict(data)

    def __init__(self, explanation, model=None, *, dataset=None,
                 true_y=None, classes=None, features=None, port=None, use_cdn=True,
                 datasetX=None, trueY=None, locale=None):
        # support legacy kwarg names
        if dataset is None and datasetX is not None:
            dataset = datasetX
        if true_y is None and trueY is not None:
            true_y = trueY
        explanation_input =\
            ExplanationDashboardInput(explanation, model, dataset, true_y, classes, features, None, locale)
        html = self._generate_inline_html(explanation_input)

        display(HTML(html))
        return
        if not ExplanationDashboard.service:
            try:
                ExplanationDashboard.service = ExplanationDashboard.DashboardService(port)
                self._thread = threading.Thread(target=ExplanationDashboard.service.run, daemon=True)
                self._thread.start()
            except Exception as e:
                ExplanationDashboard.service = None
                raise e
        ExplanationDashboard.service.use_cdn = use_cdn
        ExplanationDashboard.model_count += 1
        predict_url = "http://{0}:{1}/{2}/predict".format(
            ExplanationDashboard.service.ip,
            ExplanationDashboard.service.port,
            str(ExplanationDashboard.model_count))
        ExplanationDashboard.explanations[str(ExplanationDashboard.model_count)] =\
            ExplanationDashboardInput(explanation, model, dataset, true_y, classes, features, predict_url, locale)

        if "DATABRICKS_RUNTIME_VERSION" in os.environ:
            html = "<iframe src='http://{0}:{1}/{2}' width='100%' height='1200px' frameBorder='0'></iframe>".format(
                ExplanationDashboard.service.ip,
                ExplanationDashboard.service.port,
                ExplanationDashboard.model_count)
            _render_databricks(html)
        else:
            url = 'http://{0}:{1}/{2}'.format(
                ExplanationDashboard.service.ip,
                ExplanationDashboard.service.port,
                ExplanationDashboard.model_count)
            display(IFrame(url, "100%", 1200))

    def _generate_inline_html(self, explanation_input_object):
        explanation_input = json.dumps(explanation_input_object.dashboard_input)
        script_path = os.path.dirname(os.path.abspath(__file__))
        js_path = os.path.join(script_path, "static", "index.js")
        with open(js_path, "r", encoding="utf-8") as f:
            js = f.read()
        return ExplanationDashboard.default_template.render(explanation=explanation_input,
                                                            main_js=js,
                                                            app_id='app_123')


# NOTE: Code mostly derived from Plotly's databricks render as linked below:
# https://github.com/plotly/plotly.py/blob/01a78d3fdac14848affcd33ddc4f9ec72d475232/packages/python/plotly/plotly/io/_base_renderers.py
def _render_databricks(html):  # pragma: no cover
    import inspect

    if _render_databricks.displayHTML is None:
        found = False
        for frame in inspect.getouterframes(inspect.currentframe()):
            global_names = set(frame.frame.f_globals)
            target_names = {DatabricksInterfaceConstants.DISPLAY_HTML,
                            DatabricksInterfaceConstants.DISPLAY,
                            DatabricksInterfaceConstants.SPARK}
            if target_names.issubset(global_names):
                _render_databricks.displayHTML = frame.frame.f_globals[
                    DatabricksInterfaceConstants.DISPLAY_HTML]
                found = True
                break

        if not found:
            msg = "Could not find databrick's displayHTML function"
            raise RuntimeError(msg)

    _render_databricks.displayHTML(html)


_render_databricks.displayHTML = None
