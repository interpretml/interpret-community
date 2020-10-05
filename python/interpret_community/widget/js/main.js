import React from 'react';
import ReactDOM from 'react-dom';
import axios from 'axios';

import { NewExplanationDashboard } from 'interpret-dashboard';

const RenderDashboard = (divId, data) => {
  let generatePrediction = (postData) => {
    if (data.withCredentials) {
        var headers_data = {
            'Accept': 'application/json,text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'Content-Type': 'application/json',
        }
        axios.defaults.withCredentials = true
        var axios_options = { headers: headers_data, withCredentials: true }
        return axios.post(data.predictionUrl, JSON.stringify(postData), axios_options)
            .then((response) => {
                return response.data.data
            })
            .catch(function (error) {
                throw new Error(error)
            })
    } else {
        return fetch(data.predictionUrl, {method: "post", body: JSON.stringify(postData), headers: {
            'Content-Type': 'application/json'
          }}).then(resp => {
            if (resp.status >= 200 && resp.status < 300) {
              return resp.json()
            }
            return Promise.reject(new Error(resp.statusText))
          }).then(json => {
            if (json.error !== undefined) {
              throw new Error(json.error)
            }
            return Promise.resolve(json.data)
          })
    }
  }

  ReactDOM.render(<NewExplanationDashboard
      modelInformation={{modelClass: 'blackbox'}}
      dataSummary={{featureNames: data.featureNames, classNames: data.classNames}}
      testData={data.trainingData}
      predictedY={data.predictedY}
      probabilityY={data.probabilityY}
      trueY={data.trueY}
      precomputedExplanations={{
        localFeatureImportance: data.localExplanations,
        globalFeatureImportance: data.globalExplanation,
        ebmGlobalExplanation: data.ebmData
      }}
      requestPredictions={data.predictionUrl !== undefined ? generatePrediction : undefined}
      locale={data.locale}
      explanationMethod={data.explanation_method}
      key={new Date()}
    />, document.getElementById(divId));
}
  
export { RenderDashboard, NewExplanationDashboard };
