import React from 'react';
import ReactDOM from 'react-dom';

import { ExplanationDashboard } from 'mlchartlib';


let generatePrediction = (postData) => {
  return fetch(data.predictionUrl, {method: "post", body: postData})
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
    theme={"dark"}
    key={new Date()}
  />, document.getElementById(data.divId));