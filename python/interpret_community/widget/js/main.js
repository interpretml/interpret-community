import React from 'react';
import ReactDOM from 'react-dom';

import { ExplanationDashboard } from 'interpret-dashboard';


let generatePrediction = (postData) => {
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

ReactDOM.render(<ExplanationDashboard
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
    key={new Date()}
  />, document.getElementById(data.divId));