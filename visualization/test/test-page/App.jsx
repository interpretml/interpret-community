import React from 'react';
import { ExplanationDashboard } from 'mlchartlib';
import  {breastCancerData} from '../__mock_data/dummyData';
import {ibmData} from '../__mock_data/ibmData';
import {irisData} from '../__mock_data/irisData';
import {irisGlobal} from '../__mock_data/irisGlobal';
import {irisDataGlobal} from '../__mock_data/irisDataGlobal';
import {bostonData} from '../__mock_data/bostonData';
import {ebmData } from '../__mock_data/ebmData';
import {irisNoData} from '../__mock_data/irisNoData';
import {largeFeatureCount} from '../__mock_data/largeFeatureCount';

  var ibmNoClass = _.cloneDeep(ibmData);
  ibmNoClass.classNames = undefined;

  var irisNoFeatures = _.cloneDeep(irisData);
  irisNoFeatures.featureNames = undefined;

    class App extends React.Component {
      constructor(props) {
        super(props);
        this.state = {value: 3};
        this.handleChange = this.handleChange.bind(this);
        this.generateRandomScore = this.generateRandomScore.bind(this);
      }

      static choices = [
        {label: 'bostonData', data: bostonData},
        {label: 'irisData', data: irisData},
        {label: 'irisGlobal', data: irisGlobal},
        {label: 'irisDataGlobal', data: irisDataGlobal},
        {label: 'ibmData', data: ibmData},
        {label: 'breastCancer', data: breastCancerData},
        {label: 'ibmNoClass', data: ibmNoClass},
        {label: 'irisNoFeature', data: irisNoFeatures},
        {label: 'ebmData', data: ebmData},
        {label: 'irisNoData', data: irisNoData},
        {label: 'largeFeatureCount', data: largeFeatureCount}
      ]

      messages = {
        'LocalExpAndTestReq': [{displayText: 'LocalExpAndTestReq'}],
        'LocalOrGlobalAndTestReq': [{displayText: 'LocalOrGlobalAndTestReq'}],
        'TestReq': [{displayText: 'TestReq'}],
        'PredictorReq': [{displayText: 'PredictorReq'}]
      }

      handleChange(event){
        this.setState({value: event.target.value});
      }

      generateRandomScore(data) {
        return Promise.resolve(data.map(x => Math.random()));
      }

      generateRandomProbs(classDimensions, data, signal) {
        let promise = new Promise((resolve, reject) => {
          let timeout = setTimeout(() => {resolve(data.map(x => Array.from({length:classDimensions}, (unused) => Math.random())))}, 300);
          signal.addEventListener('abort', () => {
            clearTimeout(timeout);
            reject(new DOMException('Aborted', 'AbortError'));
          });
        });

        return promise;
      }

      generateExplanatins(explanations, data, signal) {
        let promise = new Promise((resolve, reject) => {
          let timeout = setTimeout(() => {resolve(explanations)}, 300);
          signal.addEventListener('abort', () => {
            clearTimeout(timeout);
            reject(new DOMException('Aborted', 'AbortError'));
          });
        });

        return promise;
      }


      render() {
        const data = _.cloneDeep(App.choices[this.state.value].data);
        // data.localExplanations = undefined;
        const classDimension = data.localExplanations && Array.isArray(data.localExplanations.scores[0][0]) ?
          data.localExplanations.length : 1;
        return (
          <div style={{backgroundColor: 'grey', height:'100%'}}>
            <label>
              Select dataset:
            </label>
            <select value={this.state.value} onChange={this.handleChange}>
              {App.choices.map((item, index) => <option key={item.label} value={index}>{item.label}</option>)}
            </select>
              <div style={{ width: '80vw', backgroundColor: 'white', margin:'50px auto'}}>
                  <div style={{ width: '100%'}}>
                      <ExplanationDashboard
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
                        requestPredictions={this.generateRandomProbs.bind(this, classDimension)}
                        stringParams={{contextualHelp: this.messages}}
                        requestLocalFeatureExplanations={this.generateExplanatins.bind(this, App.choices[this.state.value].data.localExplanations)}
                        theme={"dark"}
                        key={new Date()}
                      />
                  </div>
              </div>
          </div>
        );
      }
    }

    export default App;