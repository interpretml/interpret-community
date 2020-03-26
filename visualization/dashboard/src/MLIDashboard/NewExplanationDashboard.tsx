import React from "react";
import { IExplanationDashboardProps, IMultiClassLocalFeatureImportance, ISingleClassLocalFeatureImportance } from "./Interfaces";
import { IFilter, IFilterContext } from "./Interfaces/IFilter";
import { JointDataset } from "./JointDataset";
import { ModelMetadata } from "mlchartlib";
import { localization } from "../Localization/localization";
import { IExplanationModelMetadata, ModelTypes } from "./IExplanationContext";
import * as memoize from "memoize-one";
import { IPivot, IPivotItemProps, PivotItem, Pivot, PivotLinkSize } from "office-ui-fabric-react/lib/Pivot";
import _ from "lodash";
import { NewDataExploration } from "./Controls/Scatter/NewDataExploration";
import { GlobalExplanationTab, IGlobalBarSettings } from "./Controls/GlobalExplanationTab";
import { mergeStyleSets } from "office-ui-fabric-react/lib/Styling";
import { ModelExplanationUtils } from "./ModelExplanationUtils";
import { WhatIfTab } from "./Controls/WhatIfTab";
import { Cohort } from "./Cohort";
import { CohortControl } from "./Controls/CohortControl";
import { initializeIcons } from "@uifabric/icons";
import { ModelPerformanceTab } from "./Controls/ModelPerformaceTab";

export interface INewExplanationDashboardState {
    cohorts: Cohort[];
    activeGlobalTab: globalTabKeys;
    jointDataset: JointDataset;
    modelMetadata: IExplanationModelMetadata;
    modelChartConfig: IGenericChartProps;
    dataChartConfig: IGenericChartProps;
    whatIfChartConfig: IGenericChartProps;
    globalBarConfig: IGlobalBarSettings;
    dependenceProps: IGenericChartProps;
    globalImportanceIntercept: number;
    globalImportance: number[];
    isGlobalImportanceDerivedFromLocal: boolean;
    sortVector: number[];
}

interface IGlobalExplanationProps {
    globalImportanceIntercept: number;
    globalImportance: number[];
    isGlobalImportanceDerivedFromLocal: boolean;
}

export enum ChartTypes {
    Scatter = "scattergl",
    Bar = "histogram",
    Box = "box"
}

export interface IGenericChartProps {
    chartType: ChartTypes;
    xAxis?: ISelectorConfig;
    yAxis?: ISelectorConfig;
    colorAxis?: ISelectorConfig;
    selectedCohortIndex?: number;
}

export interface ISelectorConfig {
    property: string;
    index?: number;
    options: {
        dither?: boolean;
        // this is only used in the ambiguous case of numeric values on color axis for scatter chart, when binned or unbinned are valid
        bin?: boolean;
    };
}

enum globalTabKeys {
    modelPerformance = "modelPerformance",
    dataExploration = "dataExploration",
    explanationTab = "explanationTab",
    whatIfTab = "whatIfTab"
}

export class NewExplanationDashboard extends React.PureComponent<IExplanationDashboardProps, INewExplanationDashboardState> {
    private static iconsInitialized = false;
    
    private static initializeIcons(props: IExplanationDashboardProps): void {
        if (NewExplanationDashboard.iconsInitialized === false && props.shouldInitializeIcons !== false) {
            initializeIcons(props.iconUrl);
            NewExplanationDashboard.iconsInitialized = true;
        }
    }
    
    private static readonly classNames = mergeStyleSets({
        pivotWrapper: {
            display: "contents"
        }
    });

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
        let classNames = props.dataSummary.classNames;
        const classLength = NewExplanationDashboard.getClassLength(props);
        if (!classNames || classNames.length !== classLength) {
            classNames = NewExplanationDashboard.buildIndexedNames(classLength, localization.defaultClassNames);
        }
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
        if (props.probabilityY && Array.isArray(props.probabilityY) && Array.isArray(props.probabilityY[0]) && props.probabilityY[0].length > 0) {
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

    private static buildGlobalProperties(props: IExplanationDashboardProps): IGlobalExplanationProps {
        const result: IGlobalExplanationProps = {} as IGlobalExplanationProps;
        if (props.precomputedExplanations &&
            props.precomputedExplanations.globalFeatureImportance &&
            props.precomputedExplanations.globalFeatureImportance.scores) {
            result.isGlobalImportanceDerivedFromLocal = false;
            if ((props.precomputedExplanations.globalFeatureImportance.scores as number[][])
                .every(dim1 => Array.isArray(dim1))) {
                result.globalImportance = (props.precomputedExplanations.globalFeatureImportance.scores as number[][])
                    .map(classArray => classArray.reduce((a, b) => a + b), 0);
                result.globalImportanceIntercept = (props.precomputedExplanations.globalFeatureImportance.intercept as number[])
                    .reduce((a, b) => a + b, 0);
            } else {
                result.globalImportance = props.precomputedExplanations.globalFeatureImportance.scores as number[];
                result.globalImportanceIntercept = props.precomputedExplanations.globalFeatureImportance.intercept as number;
            }
        }
        return result;
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
        const globalProps = NewExplanationDashboard.buildGlobalProperties(props);
        // consider taking filters in as param arg for programatic users
        const cohorts = [new Cohort(localization.Cohort.cohort + " 0", jointDataset, [])];
        return {
            cohorts,
            activeGlobalTab: globalTabKeys.dataExploration,
            jointDataset,
            modelMetadata,
            modelChartConfig: undefined,
            dataChartConfig: undefined,
            whatIfChartConfig: undefined,
            dependenceProps: undefined,
            globalBarConfig: undefined,
            globalImportanceIntercept: globalProps.globalImportanceIntercept,
            globalImportance: globalProps.globalImportance,
            isGlobalImportanceDerivedFromLocal: globalProps.isGlobalImportanceDerivedFromLocal,
            sortVector: undefined
        };
    }

    private pivotItems: IPivotItemProps[] = [];
    private pivotRef: IPivot;
    constructor(props: IExplanationDashboardProps) {
        super(props);
        NewExplanationDashboard.initializeIcons(props);
        this.onModelConfigChanged = this.onModelConfigChanged.bind(this);
        this.onConfigChanged = this.onConfigChanged.bind(this);
        this.onWhatIfConfigChanged = this.onWhatIfConfigChanged.bind(this);
        this.onDependenceChange = this.onDependenceChange.bind(this);
        this.handleGlobalTabClick = this.handleGlobalTabClick.bind(this);
        this.setGlobalBarSettings = this.setGlobalBarSettings.bind(this);
        this.setSortVector = this.setSortVector.bind(this);
        this.onCohortChange = this.onCohortChange.bind(this);
        this.deleteCohort = this.deleteCohort.bind(this);
        if (this.props.locale) {
            localization.setLanguage(this.props.locale);
        }
        this.state = NewExplanationDashboard.buildInitialExplanationContext(props);

        if (this.state.jointDataset.hasDataset) {
            this.pivotItems.push({headerText: localization.dataExploration, itemKey: globalTabKeys.dataExploration});
            this.pivotItems.push({headerText: localization.modelPerformance, itemKey: globalTabKeys.modelPerformance});
        }
        if (this.state.jointDataset.localExplanationFeatureCount > 0) {
            this.pivotItems.push({headerText: localization.globalImportance, itemKey: globalTabKeys.explanationTab});
        }
        if (this.state.jointDataset.localExplanationFeatureCount > 0 && this.state.jointDataset.hasDataset && this.props.requestPredictions) {
            this.pivotItems.push({headerText: localization.explanationExploration, itemKey: globalTabKeys.whatIfTab});
        }
    }

    render(): React.ReactNode {
        const cohortIDs = this.state.cohorts.map(cohort => cohort.getCohortID().toString());
        // const filterContext: IFilterContext = {
        //     filters: this.state.filters,
        //     onAdd: this.addFilter,
        //     onDelete: this.deleteFilter,
        //     onUpdate: this.updateFilter
        // }
        // this.state.jointDataset.applyFilters(this.state.filters);
        return (
            <>
                <div className="explainerDashboard">
                    <CohortControl 
                        cohorts={this.state.cohorts}
                        jointDataset={this.state.jointDataset}
                        onChange={this.onCohortChange}
                        onDelete={this.deleteCohort}
                    />
                        <div className={NewExplanationDashboard.classNames.pivotWrapper}>
                            <Pivot
                                componentRef={ref => {this.pivotRef = ref;}}
                                selectedKey={this.state.activeGlobalTab}
                                onLinkClick={this.handleGlobalTabClick}
                                linkSize={PivotLinkSize.normal}
                                headersOnly={true}
                            >
                                {this.pivotItems.map(props => <PivotItem key={props.itemKey} {...props}/>)}
                            </Pivot>
                            {this.state.activeGlobalTab === globalTabKeys.modelPerformance && (
                                <ModelPerformanceTab
                                    jointDataset={this.state.jointDataset}
                                    theme={this.props.theme}
                                    metadata={this.state.modelMetadata}
                                    chartProps={this.state.modelChartConfig}
                                    onChange={this.onModelConfigChanged}
                                    cohorts={this.state.cohorts}
                                />
                            )}
                            {this.state.activeGlobalTab === globalTabKeys.dataExploration && (
                                <NewDataExploration
                                    jointDataset={this.state.jointDataset}
                                    theme={this.props.theme}
                                    metadata={this.state.modelMetadata}
                                    chartProps={this.state.dataChartConfig}
                                    onChange={this.onConfigChanged}
                                    cohorts={this.state.cohorts}
                                />
                            )}
                            {this.state.activeGlobalTab === globalTabKeys.explanationTab && (
                                <GlobalExplanationTab
                                    globalBarSettings={this.state.globalBarConfig}
                                    sortVector={this.state.sortVector}
                                    dependenceProps={this.state.dependenceProps}
                                    theme={this.props.theme}
                                    jointDataset={this.state.jointDataset}
                                    metadata={this.state.modelMetadata}
                                    globalImportance={this.state.globalImportance}
                                    isGlobalDerivedFromLocal={this.state.isGlobalImportanceDerivedFromLocal}
                                    onChange={this.setGlobalBarSettings}
                                    onDependenceChange={this.onDependenceChange}
                                    requestSortVector={this.setSortVector}
                                    cohorts={this.state.cohorts}
                                    cohortIDs={cohortIDs}
                                />
                            )}
                            {this.state.activeGlobalTab === globalTabKeys.whatIfTab && (
                                <WhatIfTab 
                                    theme={this.props.theme}
                                    jointDataset={this.state.jointDataset}
                                    metadata={this.state.modelMetadata}
                                    cohorts={this.state.cohorts}
                                    onChange={this.onWhatIfConfigChanged}
                                    chartProps={this.state.whatIfChartConfig}
                                    invokeModel={this.props.requestPredictions}
                                />
                            )}
                        </div>
                    </div>
            </>
        );
    }

    private onConfigChanged(newConfig: IGenericChartProps): void {
        this.setState({dataChartConfig: newConfig});
    }

    private onModelConfigChanged(newConfig: IGenericChartProps): void {
        this.setState({modelChartConfig: newConfig});
    }

    private onWhatIfConfigChanged(newConfig: IGenericChartProps): void {
        this.setState({whatIfChartConfig: newConfig});
    }

    private onDependenceChange(newConfig: IGenericChartProps): void {
        this.setState({dependenceProps: newConfig});
    }

    private handleGlobalTabClick(item: PivotItem): void {
        let index: globalTabKeys = globalTabKeys[item.props.itemKey];
        this.setState({activeGlobalTab: index});
    }

    private setGlobalBarSettings(settings: IGlobalBarSettings): void {
        this.setState({globalBarConfig: settings});
    }

    private setSortVector(): void {
        this.setState({sortVector: ModelExplanationUtils.getSortIndices(this.state.cohorts[0].calculateAverageImportance()).reverse()});
    }

    private onCohortChange(newCohort: Cohort, index: number): void {
        const prevCohorts = [...this.state.cohorts];
        prevCohorts[index] = newCohort;
        this.setState({cohorts: prevCohorts});
    }

    private deleteCohort(index: number): void {
        const prevCohorts = [...this.state.cohorts];
        prevCohorts.splice(index, 1);
        this.setState({cohorts: prevCohorts});
    }
}