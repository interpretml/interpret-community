import React from "react";
import { IExplanationDashboardProps, IMultiClassLocalFeatureImportance, ISingleClassLocalFeatureImportance } from "./Interfaces";
import { IFilter } from "./Interfaces/IFilter";
import { JointDataset } from "./JointDataset";
import { IGenericChartProps } from "./Controls/ChartWithControls";
import { ModelMetadata } from "mlchartlib";
import { localization } from "../Localization/localization";
import { IExplanationModelMetadata, ModelTypes } from "./IExplanationContext";
import * as memoize from "memoize-one";
import { IPivot, IPivotItemProps } from "office-ui-fabric-react/lib/Pivot";

export interface INewExplanationDashboardState {
    filters: IFilter[];
    activeGlobalTab: number;
    jointDataset: JointDataset;
    modelMetadata: IExplanationModelMetadata;
    chartConfigs: {[key: string]: IGenericChartProps};
}

export class NewExplanationDashboard extends React.PureComponent<IExplanationDashboardProps, INewExplanationDashboardState> {

    private static globalTabKeys: string[] = [
        "dataExploration",
        "globalImportance",
        "explanationExploration",
        "summaryImportance",
        "modelExplanation",
        "customVisualization"
    ];

    private static buildModelMetadata(props: IExplanationDashboardProps): IExplanationModelMetadata {
        const modelType = NewExplanationDashboard.getModelType(props);
        let featureNames = props.dataSummary.featureNames;
        let featureNamesAbridged: string[];
        const maxLength = 18;
        if (featureNames !== undefined) {
            if (!featureNames.every(name => typeof name === "string")) {
                featureNames = featureNames.map(x => x.toString());
            }
            featureNamesAbridged = featureNames.map(name => {
                return name.length <= maxLength ? name : `${name.slice(0, maxLength)}...`;
            });
        } else {
            let featureLength = 0;
            if (props.testData && props.testData[0] !== undefined) {
                featureLength = props.testData[0].length;
            } else if (props.precomputedExplanations && props.precomputedExplanations.globalFeatureImportance) {
                featureLength = props.precomputedExplanations.globalFeatureImportance.scores.length;
            } else if (props.precomputedExplanations && props.precomputedExplanations.localFeatureImportance) {
                const localImportances = props.precomputedExplanations.localFeatureImportance.scores;
                if ((localImportances as number[][][]).every(dim1 => {
                    return dim1.every(dim2 => Array.isArray(dim2));
                })) {
                    featureLength = (props.precomputedExplanations.localFeatureImportance.scores[0][0] as number[]).length;
                } else {
                    featureLength = (props.precomputedExplanations.localFeatureImportance.scores[0] as number[]).length;
                }
            } else if (props.precomputedExplanations && props.precomputedExplanations.ebmGlobalExplanation) {
                featureLength = props.precomputedExplanations.ebmGlobalExplanation.feature_list.length;
            }
            featureNames = NewExplanationDashboard.buildIndexedNames(featureLength, localization.defaultFeatureNames);
            featureNamesAbridged = featureNames;
        }
        const classNames = props.dataSummary.classNames || NewExplanationDashboard.buildIndexedNames(NewExplanationDashboard.getClassLength(props), localization.defaultClassNames);
        const featureIsCategorical = ModelMetadata.buildIsCategorical(featureNames.length, props.testData, props.dataSummary.categoricalMap);
        const featureRanges = ModelMetadata.buildFeatureRanges(props.testData, featureIsCategorical, props.dataSummary.categoricalMap);
        return {
            featureNames,
            featureNamesAbridged,
            classNames,
            featureIsCategorical,
            featureRanges,
            modelType,
        };
    }

    private static getClassLength: (props: IExplanationDashboardProps) => number
    = (memoize as any).default((props: IExplanationDashboardProps): number  => {
        if (props.probabilityY) {
            return props.probabilityY[0].length;
        }
        if (props.precomputedExplanations && props.precomputedExplanations.localFeatureImportance) {
            const localImportances = props.precomputedExplanations.localFeatureImportance.scores;
            if ((localImportances as number[][][]).every(dim1 => {
                return dim1.every(dim2 => Array.isArray(dim2));
            })) {
                return localImportances.length;
            }
        }
        if (props.precomputedExplanations && props.precomputedExplanations.globalFeatureImportance) {
            // determine if passed in vaules is 1D or 2D
            if ((props.precomputedExplanations.globalFeatureImportance.scores as number[][])
                .every(dim1 => Array.isArray(dim1))) {
                return (props.precomputedExplanations.globalFeatureImportance.scores as number[][]).length;
            }
        }
        // default to regression case
        return 1;
    });

    private static buildIndexedNames(length: number, baseString: string): string[] {
        return Array.from(Array(length).keys())
        .map(i => localization.formatString(baseString, i.toString()) as string);
    }

    private static getModelType(props: IExplanationDashboardProps): ModelTypes {
        // If python gave us a hint, use it
        if (props.modelInformation.method === "regressor") {
            return ModelTypes.regression;
        }
        switch(NewExplanationDashboard.getClassLength(props)) {
            case 1:
                return ModelTypes.regression;
            case 2:
                return ModelTypes.binary;
            default:
                return ModelTypes.multiclass;
        }
    }

    public static buildInitialExplanationContext(props: IExplanationDashboardProps): INewExplanationDashboardState {
        const modelMetadata = NewExplanationDashboard.buildModelMetadata(props);
        let localExplanations: IMultiClassLocalFeatureImportance | ISingleClassLocalFeatureImportance;
        if (props && props.precomputedExplanations && props.precomputedExplanations.localFeatureImportance &&
            props.precomputedExplanations.localFeatureImportance.scores) {
                localExplanations = props.precomputedExplanations.localFeatureImportance;
            }
        const jointDataset = new JointDataset({
            dataset: props.testData,
            predictedY: props.predictedY, 
            trueY: props.trueY,
            localExplanations,
            metadata: modelMetadata
        });
        return {
            filters: [],
            activeGlobalTab: 0,
            jointDataset,
            modelMetadata,
            chartConfigs: {}
        };
    }

    private pivotItems: IPivotItemProps[];
    private pivotRef: IPivot;
    constructor(props: IExplanationDashboardProps) {
        super(props);
        if (this.props.locale) {
            localization.setLanguage(this.props.locale)
        }
        const state: INewExplanationDashboardState = NewExplanationDashboard.buildInitialExplanationContext(props);

    }
}