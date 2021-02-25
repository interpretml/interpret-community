from flask import Flask, request, jsonify
from flask_cors import CORS
from jinja2 import Environment, PackageLoader
from IPython.display import display, HTML
from interpret.utils.environment import EnvironmentDetector, is_cloud_env
from interpret_community.common.serialization_utils import _serialize_json_safe
import threading
import socket
import re
import os
import json
import atexit
import warnings
from .explanation_dashboard_input import ExplanationDashboardInput
from ._internal.constants import ExplanationDashboardInterface, DatabricksInterfaceConstants

try:
    from gevent.pywsgi import WSGIServer
except ModuleNotFoundError:
    raise RuntimeError("Error: gevent package is missing, please run 'conda install gevent' or"
                       "'pip install gevent' or 'pip install interpret-community[visualization]'")


NBVM_FILE_PATH = "/mnt/azmnt/.nbvm"
CREDENTIALED_VM = 'credentialed_vm'
PUBLIC_VM = 'public_vm'
LOCALHOST = 'localhost'
VM_ENVS = {CREDENTIALED_VM, PUBLIC_VM}


def _get_nbvm():
    if not (os.path.exists(NBVM_FILE_PATH) and os.path.isfile(NBVM_FILE_PATH)):
        return None
    # regex to find items of the form key=value where value will be part of a url
    # the keys of interest to us are "instance" and domainsuffix"
    envre = re.compile(r'''^([^\s=]+)=(?:[\s"']*)(.+?)(?:[\s"']*)$''')
    result = {}
    with open(NBVM_FILE_PATH) as nbvm_variables:
        for line in nbvm_variables:
            match = envre.match(line)
            if match is not None:
                result[match.group(1)] = match.group(2)
    if "instance" not in result or "domainsuffix" not in result:
        return None
    return result


class ExplanationDashboard:
    """Explanation Dashboard Class.

    :param explanation: An object that represents an explanation.
    :type explanation: ExplanationMixin
    :param model: An object that represents a model. It is assumed that for the classification case
        it has a method of predict_proba() returning the prediction probabilities for each
        class and for the regression case a method of predict() returning the prediction value.
    :type model: object
    :param dataset:  A matrix of feature vector examples (# examples x # features), the same samples
        used to build the explanation. Overwrites any existing dataset on the explanation object. Must have fewer than
        10000 rows and fewer than 1000 columns.
    :type dataset: numpy.array or list[][]
    :param datasetX: Alias of the dataset parameter. If dataset is passed, this will have no effect. Must have fewer
        than 10000 rows and fewer than 1000 columns.
    :type datasetX: numpy.array or list[][]
    :param true_y: The true labels for the provided dataset. Overwrites any existing dataset on the
        explanation object.
    :type true_y: numpy.array or list[]
    :param classes: The class names.
    :type classes: numpy.array or list[]
    :param features: Feature names.
    :type features: numpy.array or list[]
    :param port: The port to use on locally hosted service.
    :type port: int
    :param use_cdn: Deprecated. Whether to load latest dashboard script from cdn, fall back to local script if False.
        .. deprecated:: 0.15.2
           Deprecated since 0.15.2, cdn has been removed.  Setting parameter to True or False will trigger warning.
    :type use_cdn: bool
    :param public_ip: Optional. If running on a remote vm, the external public ip address of the VM.
    :type public_ip: str
    :param with_credentials: Optional. If running on a remote vm, sets up CORS policy both on client and server.
    :type with_credentials: bool
    """

    service = None
    explanations = {}
    model_count = 0
    _dashboard_js = None
    env = Environment(loader=PackageLoader(__name__, 'templates'))
    default_template = env.get_template("inlineDashboard.html")

    class DashboardService:

        def __init__(self, port, public_ip, with_credentials=False):
            app = Flask(__name__)
            self.nbvm = _get_nbvm()
            if self.nbvm is None and not with_credentials:
                self.cors = CORS(app)
                self.with_credentials = False
                if public_ip is not None:
                    self.ip = public_ip
                    self.env = PUBLIC_VM
                else:
                    self.ip = LOCALHOST
                    self.env = 'local'
            elif self.nbvm is not None:
                # Note: for debugging CORS set logging.getLogger('flask_cors').level = logging.DEBUG
                instance_name = self.nbvm["instance"]
                domain_suffix = self.nbvm["domainsuffix"]
                nbvm_origin1 = "https://{}.{}".format(instance_name, domain_suffix)
                nbvm_origin2 = "https://{}-{}.{}".format(instance_name, port, domain_suffix)
                nbvm_origins = [nbvm_origin1, nbvm_origin2]
                headers = ['Content-Type']
                # Support credentials for notebook VM scenario
                self.cors = CORS(app, origins=nbvm_origins, expose_headers=headers, supports_credentials=True)
                self.with_credentials = True
                self.ip = LOCALHOST
                self.env = 'azure'
            else:
                if public_ip is not None:
                    self.ip = public_ip
                else:
                    # Attempt to get the ip, but this may fail since it may not get the external ip of
                    # the machine, just the private ip. Note we don't use LOCALHOST here because it
                    # doesn't make sense to run with CORS on local machine anyway.
                    host_name = socket.gethostname()
                    self.ip = socket.gethostbyname(host_name)
                origin = "https://{}:{}".format(self.ip, port)
                headers = ['Content-Type']
                self.cors = CORS(app, origins=[origin], expose_headers=headers, supports_credentials=True)
                self.with_credentials = True
                self.env = CREDENTIALED_VM

            @app.route('/')
            def hello():
                return "No global list view supported at this time."

            @app.route('/<id>')
            def explanation_visual(id):
                if id in ExplanationDashboard.explanations:
                    return generate_inline_html(ExplanationDashboard.explanations[id], None)
                else:
                    return "Unknown model id."

            @app.route('/<id>/predict', methods=['POST'])
            def predict(id):
                data = request.get_json(force=True)
                if id in ExplanationDashboard.explanations:
                    return jsonify(ExplanationDashboard.explanations[id].on_predict(data))

            self.app = app
            self.port = port
            if self.port is None:
                # Try 100 different ports
                for port in range(5000, 5100):
                    available = ExplanationDashboard.DashboardService._local_port_available(port, rais=False)
                    if available:
                        self.port = port
                        return
                error_message = """Ports 5000 to 5100 not available.
                    Please specify an open port for use via the 'port' parameter"""
                raise RuntimeError(
                    error_message.format(port)
                )
            else:
                ExplanationDashboard.DashboardService._local_port_available(self.port)

        def run(self):
            class devnull:
                write = lambda _: None  # noqa: E731
            ip = LOCALHOST
            # Note: for credentialed VM or public VM we need to use the private IP address
            if self.env in VM_ENVS:
                host_name = socket.gethostname()
                ip = socket.gethostbyname(host_name)
            server = WSGIServer((ip, self.port), self.app, log=devnull)
            self.app.config["server"] = server
            server.serve_forever()

            # Closes server on program exit, including freeing all sockets
            def closeserver():
                server.stop()

            atexit.register(closeserver)

        def get_base_url(self):
            env = EnvironmentDetector()
            detected_envs = env.detect()
            in_cloud_env = is_cloud_env(detected_envs)
            result = _get_nbvm()
            # First handle known cloud environments
            if result is None:
                # special case azure, since the azure sdk can set this env setting on local runs
                if not in_cloud_env or self.env in VM_ENVS:
                    return "http://{0}:{1}".format(
                        self.ip,
                        self.port)
                # all non-specified, non-credentialed cloud environments are not handled
                if not self.with_credentials:
                    self.env = 'cloud'
                return None

            instance_name = result["instance"]
            domain_suffix = result["domainsuffix"]
            return "https://{}-{}.{}".format(instance_name, self.port, domain_suffix)

        @staticmethod
        def _local_port_available(port, rais=True):
            """
            Borrowed from:
            https://stackoverflow.com/questions/19196105/how-to-check-if-a-network-port-is-open-on-linux
            """
            try:
                backlog = 5
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind((LOCALHOST, port))
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

    def __init__(self, explanation, model=None, *, dataset=None,
                 true_y=None, classes=None, features=None, port=None,
                 datasetX=None, trueY=None, locale=None, public_ip=None,
                 with_credentials=False, use_cdn=None):
        if use_cdn is not None:
            warnings.warn("""use_cdn parameter is deprecated, cdn has been deleted.
                          Constructor parameter will be removed in the future""")
        warnings.warn("""ExplanationDashboard in interpret-community package is deprecated and will be removed.
                      Please use the ExplanationDashboard from raiwidgets package instead.""")
        # support legacy kwarg names
        if dataset is None and datasetX is not None:
            dataset = datasetX
        if true_y is None and trueY is not None:
            true_y = trueY
        self._initialize_js()
        predict_url = None
        local_url = None
        if not ExplanationDashboard.service:
            try:
                ExplanationDashboard.service = ExplanationDashboard.DashboardService(port, public_ip, with_credentials)
                self._thread = threading.Thread(target=ExplanationDashboard.service.run, daemon=True)
                self._thread.start()
            except Exception as e:
                ExplanationDashboard.service = None
                raise e
        ExplanationDashboard.model_count += 1
        base_url = ExplanationDashboard.service.get_base_url()
        if base_url is not None:
            predict_url = "{0}/{1}/predict".format(
                base_url,
                str(ExplanationDashboard.model_count))
            local_url = "{0}/{1}".format(
                base_url,
                str(ExplanationDashboard.model_count))
        with_credentials = ExplanationDashboard.service.with_credentials
        explanation_input = ExplanationDashboardInput(explanation, model, dataset, true_y, classes,
                                                      features, predict_url, locale, with_credentials)
        # Due to auth, predict is only available in separate tab in cloud after login
        if ExplanationDashboard.service.env != "cloud":
            explanation_input.enable_predict_url()
        html = generate_inline_html(explanation_input, local_url)
        ExplanationDashboard.explanations[str(ExplanationDashboard.model_count)] = explanation_input

        if "DATABRICKS_RUNTIME_VERSION" in os.environ:
            _render_databricks(html)
        else:
            display(HTML(html))

    def _initialize_js(self):
        script_path = os.path.dirname(os.path.abspath(__file__))
        js_path = os.path.join(script_path, "static", "index.js")
        with open(js_path, "r", encoding="utf-8") as f:
            ExplanationDashboard._dashboard_js = f.read()


def generate_inline_html(explanation_input_object, local_url):
    data = explanation_input_object.dashboard_input[ExplanationDashboardInterface.TRAINING_DATA]
    data = _serialize_json_safe(data)
    explanation_input_object.dashboard_input[ExplanationDashboardInterface.TRAINING_DATA] = data
    explanation_input = json.dumps(explanation_input_object.dashboard_input)
    return ExplanationDashboard.default_template.render(explanation=explanation_input,
                                                        main_js=ExplanationDashboard._dashboard_js,
                                                        app_id='app_123',
                                                        local_url=local_url,
                                                        has_local_url=local_url is not None)


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
