import { initializeIcons } from "@uifabric/icons";
import * as _ from "lodash";
import * as memoize from "memoize-one";
import { PrimaryButton } from "office-ui-fabric-react/lib/Button";
import { IComboBox, IComboBoxOption } from "office-ui-fabric-react/lib/components/ComboBox";
import { IDropdownOption } from "office-ui-fabric-react/lib/Dropdown";
import { Pivot, PivotItem, PivotLinkFormat, PivotLinkSize, IPivotItemProps } from "office-ui-fabric-react/lib/Pivot";
import * as React from "react";
import { localization } from "../Localization/localization";
import { IPlotlyProperty, SelectionContext } from "../Shared";
import { FabricStyles } from "./FabricStyles";
import {
    FeatureImportanceWrapper,
    GlobalFeatureImportanceId,
    IFeatureImportanceConfig,
    FeatureImportanceModes,
    BarId,
    DataExploration,
    DataScatterId,
    ExplanationScatterId,
    ExplanationExploration,
    ICEPlot,
    PerturbationExploration,
    SinglePointFeatureImportance,
    LocalBarId,
    FeatureImportanceBar
} from "./Controls";

import { ICategoricalRange } from "../Shared/ICategoricalRange";
import { IExplanationContext, IExplanationGenerators, IGlobalExplanation, ILocalExplanation, IExplanationModelMetadata, ITestDataset, ModelTypes, IFeatureValueExplanation, IMultiClassBoundedCoordinates } from "./IExplanationContext";
import { IExplanationDashboardProps } from "./Interfaces/IExplanationDashboardProps";
import { INumericRange } from "../Shared/INumericRange";
import { IWeightedDropdownContext, WeightVectorOption, WeightVectors } from "./IWeightedDropdownContext";
import { ModelExplanationUtils } from "./ModelExplanationUtils";
import { RangeTypes } from "../Shared/RangeTypes";
import { IBarChartConfig } from "./SharedComponents/IBarChartConfig";
import { HelpMessageDict } from "./Interfaces/IStringsParam";
import { ModelMetadata } from "../Shared/ModelMetadata";
import { EbmExplanation } from "./Controls/EbmExplanation";

initializeIcons();

const s = require("./ExplanationDashboard.css");
const RowIndex: string = "rowIndex";

export interface IDashboardContext {
    explanationContext: IExplanationContext;
    weightContext: IWeightedDropdownContext;
}

export interface IDashboardState {
    dashboardContext: IDashboardContext;
    activeGlobalTab: number;
    activeLocalTab: number;
    configs: {[key: string]: IPlotlyProperty | IFeatureImportanceConfig | IBarChartConfig};
    selectedRow: number | undefined;
}

export class ExplanationDashboard extends React.Component<IExplanationDashboardProps, IDashboardState> {
    private readonly selectionContext = new SelectionContext(RowIndex, 1);
    private selectionSubscription: string;

    private static globalTabKeys: string[] = [
        "dataExploration",
        "globalImportance",
        "explanationExploration",
        "summaryImportance",
        "modelExplanation",
        "customVisualization"
    ];

    private static localTabKeys: string[] = [
        "featureImportance",
        "perturbationExploration",
        "ICE"
    ];

    private static transposeLocalImportanceMatrix: (input: number[][][]) =>  number[][][]
        = (memoize as any).default(
        (input: number[][][]): number[][][] => {
            const numClasses =input.length;
            const numRows = input[0].length;
            const numFeatures = input[0][0].length;
            const result: number[][][] = Array(numRows).fill(0)
                .map(r => Array(numFeatures).fill(0)
                .map(f => Array(numClasses).fill(0)));
            input.forEach((rowByFeature, classIndex) => {
                rowByFeature.forEach((featureArray, rowIndex) => {
                    featureArray.forEach((value, featureIndex) => {
                        result[rowIndex][featureIndex][classIndex] = value;
                    });
                });
            });
            return result;
        }
    );

    public static buildInitialExplanationContext(props: IExplanationDashboardProps): IExplanationContext {
        const explanationGenerators: IExplanationGenerators = {
            requestPredictions: props.requestPredictions,
            requestLocalFeatureExplanations: props.requestLocalFeatureExplanations
        };
        const modelMetadata = ExplanationDashboard.buildModelMetadata(props);
        const testDataset: ITestDataset =  {
                dataset: props.testData,
                predictedY: props.predictedY,
                probabilityY: props.probabilityY,
                trueY: props.trueY
            };
        let localExplanation: ILocalExplanation;
        if (props.precomputedExplanations && props.precomputedExplanations.localFeatureImportance !== undefined && testDataset) {
            let weighting = props.predictedY ? WeightVectors.predicted : WeightVectors.absAvg;
            let localFeatureMatrix = ExplanationDashboard.buildLocalFeatureMatrix(props.precomputedExplanations.localFeatureImportance.scores, modelMetadata.modelType);
            let flattenedFeatureMatrix = ExplanationDashboard.buildLocalFlattenMatrix(localFeatureMatrix, modelMetadata.modelType, testDataset, weighting);
            let intercepts = undefined;
            // if (props.precomputedExplanations.localFeatureImportance.intercept) {
            //     intercepts = (modelMetadata.modelType === ModelTypes.regression ?
            //         [props.precomputedExplanations.localFeatureImportance.intercept] :
            //         props.precomputedExplanations.localFeatureImportance.intercept) as number[];
            // }
            localExplanation = {
                values: localFeatureMatrix,
                flattenedValues: flattenedFeatureMatrix,
                intercepts
            };
        }

        let globalExplanation: IGlobalExplanation;
        let isGlobalDerived: boolean = false;
        if (props.precomputedExplanations && props.precomputedExplanations.globalFeatureImportance !== undefined) {
            let intercepts = undefined;
            // if (props.precomputedExplanations.globalFeatureImportance.intercept) {
            //     intercepts = props.precomputedExplanations.globalFeatureImportance.intercept;
            // }
            // determine if passed in vaules is 1D or 2D
            // Use the global explanation if its been computed and is 2D
            if ((props.precomputedExplanations.globalFeatureImportance.scores as number[][])
                .every(dim1 => Array.isArray(dim1))) {
                globalExplanation = {};
                globalExplanation.perClassFeatureImportances = props.precomputedExplanations.globalFeatureImportance.scores as number[][];
                globalExplanation.flattenedFeatureImportances = globalExplanation.perClassFeatureImportances
                    .map(classArray => classArray.reduce((a, b) => a + b), 0);
                globalExplanation.intercepts = intercepts as number[];
            } else if(localExplanation === undefined) { // Take the global if we can't build better from local
                globalExplanation = {};
                globalExplanation.flattenedFeatureImportances = props.precomputedExplanations.globalFeatureImportance.scores as number[];
                globalExplanation.intercepts = [intercepts] as number[];
            }
        }
        if (globalExplanation === undefined && localExplanation !== undefined) {
            globalExplanation = ExplanationDashboard.buildGlobalExplanationFromLocal(localExplanation);
            isGlobalDerived = true;
        }

        let ebmExplanation: IFeatureValueExplanation;
        if (props.precomputedExplanations && props.precomputedExplanations.ebmGlobalExplanation !== undefined) {
            ebmExplanation = {
                featureList: props.precomputedExplanations.ebmGlobalExplanation.feature_list.map(featureExplanation => {
                    if (featureExplanation.type !== "univariate") {
                        return undefined;
                    }
                    if (featureExplanation.scores.every(dim1 => Array.isArray(dim1))) {
                        return {
                            type: "univariate",
                            scores: featureExplanation.scores,
                            names: featureExplanation.names,
                            lowerBounds: featureExplanation.lower_bounds ? featureExplanation.lower_bounds : undefined,
                            upperBounds: featureExplanation.upper_bounds ? featureExplanation.upper_bounds : undefined
                        } as IMultiClassBoundedCoordinates;
                    }
                    return {
                        type: "univariate",
                        scores: [featureExplanation.scores],
                        names: featureExplanation.names,
                        lowerBounds: featureExplanation.lower_bounds ? [featureExplanation.lower_bounds] : undefined,
                        upperBounds: featureExplanation.upper_bounds ? [featureExplanation.upper_bounds] : undefined
                    } as IMultiClassBoundedCoordinates;
                }).filter(featureExplanation => featureExplanation !== undefined),
                displayParameters: {
                    interpolation: 'vh'
                }
            };
        }

        return {
            modelMetadata,
            explanationGenerators,
            localExplanation,
            testDataset,
            globalExplanation,
            isGlobalDerived,
            ebmExplanation: ebmExplanation,
            customVis: props.precomputedExplanations.customVis
        };
    }

    private static buildLocalFeatureMatrix(localExplanationRaw: number[][] | number[][][], modelType: ModelTypes): number[][][] {
        switch(modelType) {
            case ModelTypes.regression: {
                return (localExplanationRaw as number[][])
                        .map(featureArray => featureArray.map(val => [val]));
            }
            case ModelTypes.binary: {
                return ExplanationDashboard.transposeLocalImportanceMatrix(localExplanationRaw as number[][][])
                        .map(featuresByClasses => featuresByClasses.map(classArray => classArray.slice(0, 1)));
            }
            case ModelTypes.multiclass: {
                return ExplanationDashboard.transposeLocalImportanceMatrix(localExplanationRaw as number[][][]);
            }
        }
    }

    private static buildLocalFlattenMatrix(localExplanations: number[][][], modelType: ModelTypes, testData: ITestDataset, weightVector: WeightVectorOption): number[][] {
        switch(modelType) {
            case ModelTypes.regression:
            case ModelTypes.binary: {
                // no need to flatten what is already flat
                return localExplanations.map((featuresByClasses, rowIndex) => {
                    return featuresByClasses.map((classArray) => {
                        return classArray[0];
                    });
                });
            }
            case ModelTypes.multiclass: {
                return localExplanations.map((featuresByClasses, rowIndex) => {
                    return featuresByClasses.map((classArray) => {
                        switch (weightVector) {
                            case WeightVectors.equal: {
                                return classArray.reduce((a, b) => a + b) / classArray.length;
                            }
                            case WeightVectors.predicted: {
                                return classArray[testData.predictedY[rowIndex]];
                            }
                            case WeightVectors.absAvg: {
                                return classArray.reduce((a, b) => a + Math.abs(b), 0) / classArray.length;
                            }
                            default: {
                                return classArray[weightVector];
                            }
                        }
                    });
                });
            }
        }
    }

    private static buildGlobalExplanationFromLocal(localExplanation: ILocalExplanation): IGlobalExplanation {
        return {
           perClassFeatureImportances: ModelExplanationUtils.absoluteAverageTensor(localExplanation.values)
           // intercepts: localExplanation.intercepts ? localExplanation.intercepts.map(val => Math.abs(val)) : undefined
        };
    }

    private static buildModelMetadata(props: IExplanationDashboardProps): IExplanationModelMetadata {
        const modelType = ExplanationDashboard.getModelType(props);
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
            featureNames = ModelMetadata.buildIndexedNames(featureLength, localization.defaultFeatureNames);
            featureNamesAbridged = featureNames;
        }
        const classNames = props.dataSummary.classNames || ModelMetadata.buildIndexedNames(ExplanationDashboard.getClassLength(props), localization.defaultClassNames);
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

    private static buildWeightDropdownOptions: (explanationContext: IExplanationContext) => IDropdownOption[]
        = (memoize as any).default(
        (explanationContext: IExplanationContext): IDropdownOption[] => {
            const result: IDropdownOption[] = [
                {key: WeightVectors.absAvg, text: localization.absoluteAverage}
            ];
            if (explanationContext.testDataset.predictedY) {
                result.push({key: WeightVectors.predicted, text: localization.predictedClass});
            }
            explanationContext.modelMetadata.classNames.forEach((name, index) => {
                result.push({key: index, text: name});
            });
            return result;
        }
    );

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

    private static getModelType(props: IExplanationDashboardProps): ModelTypes {
        // If python gave us a hint, use it
        if (props.modelInformation.method === "regressor") {
            return ModelTypes.regression;
        }
        switch(ExplanationDashboard.getClassLength(props)) {
            case 1:
                return ModelTypes.regression;
            case 2:
                return ModelTypes.binary;
            default:
                return ModelTypes.multiclass;
        }
    }

    private pivotItems: IPivotItemProps[];

    constructor(props: IExplanationDashboardProps) {
        super(props);
        const explanationContext: IExplanationContext = ExplanationDashboard.buildInitialExplanationContext(props);
        const defaultTopK = Math.min(8, explanationContext.modelMetadata.featureNames.length);
        this.onClassSelect = this.onClassSelect.bind(this);
        this.onConfigChanged = this.onConfigChanged.bind(this);
        this.onClearSelection = this.onClearSelection.bind(this);
        this.handleGlobalTabClick = this.handleGlobalTabClick.bind(this);
        this.handleLocalTabClick = this.handleLocalTabClick.bind(this);
        this.pivotItems = [];
        if (explanationContext.testDataset.dataset !== undefined) {
            this.pivotItems.push({headerText: localization.dataExploration, itemKey: ExplanationDashboard.globalTabKeys[0]})
        }
        if (explanationContext.globalExplanation !== undefined) {
            this.pivotItems.push({headerText: localization.globalImportance, itemKey: ExplanationDashboard.globalTabKeys[1]})
        }
        if (explanationContext.localExplanation !== undefined && explanationContext.testDataset.dataset !== undefined) {
            this.pivotItems.push({headerText: localization.explanationExploration, itemKey: ExplanationDashboard.globalTabKeys[2]})
        }
        if (explanationContext.localExplanation !== undefined) {
            this.pivotItems.push({headerText: localization.summaryImportance, itemKey: ExplanationDashboard.globalTabKeys[3]})
        }
        if (explanationContext.ebmExplanation !== undefined) {
            this.pivotItems.push({headerText: localization.summaryImportance, itemKey: ExplanationDashboard.globalTabKeys[4]})
        }
        if (explanationContext.customVis !== undefined) {
            this.pivotItems.push({headerText: localization.summaryImportance, itemKey: ExplanationDashboard.globalTabKeys[5]})
        }

        this.state = {
            dashboardContext: {
                weightContext: {
                    selectedKey: props.predictedY ? WeightVectors.predicted : WeightVectors.absAvg,
                    onSelection: this.onClassSelect,
                    options: ExplanationDashboard.buildWeightDropdownOptions(explanationContext)
                },
                explanationContext
            },
            activeGlobalTab: this.pivotItems.length > 0 ? ( ExplanationDashboard.globalTabKeys.indexOf(this.pivotItems[0].itemKey)) : 0,
            activeLocalTab: (explanationContext.localExplanation === undefined && this.props.requestPredictions) ? 1 : 0,
            configs: {
                [BarId]: {displayMode: FeatureImportanceModes.bar, topK: defaultTopK, id: BarId},
                [GlobalFeatureImportanceId]: {displayMode: FeatureImportanceModes.beehive, topK: defaultTopK, id: GlobalFeatureImportanceId},
                [LocalBarId]: {topK: defaultTopK}
            },
            selectedRow: undefined
        };
    }

    public componentDidMount(): void {
        this.selectionSubscription = this.selectionContext.subscribe({
            selectionCallback: selections => {
                let selectedRow: number | undefined;
                if (selections && selections.length > 0) {
                    let numericValue = Number.parseInt(selections[0]);
                    if (!isNaN(numericValue)) {
                        selectedRow = numericValue;
                    }
                }
                this.setState({ selectedRow });
            }
        });
        this.fetchExplanations();
    }

    public componentDidUpdate(prevProps: IExplanationDashboardProps): void {
        if (_.isEqual(prevProps, this.props)) {
            return;
        }
        const newState = _.cloneDeep(this.state);
        newState.dashboardContext.explanationContext = ExplanationDashboard.buildInitialExplanationContext(this.props);
        if (newState.dashboardContext.explanationContext.localExplanation) {
            (newState.configs[GlobalFeatureImportanceId] as IFeatureImportanceConfig).displayMode = FeatureImportanceModes.box;
        }
        this.setState(newState);
        this.fetchExplanations();
    }

    public componentWillUnmount(): void {
        if (this.selectionSubscription) {
            this.selectionContext.unsubscribe(this.selectionSubscription);
        }
    }

    public render(): React.ReactNode {
        return (
            <>
                <div className="explainerDashboard">
                    <div className="charts-wrapper">
                        <div className="global-charts-wrapper">
                            <Pivot
                                selectedKey={ExplanationDashboard.globalTabKeys[this.state.activeGlobalTab]}
                                onLinkClick={this.handleGlobalTabClick}
                                linkFormat={PivotLinkFormat.tabs}
                                linkSize={PivotLinkSize.normal}
                                headersOnly={true}
                                styles={FabricStyles.verticalTabsStyle}
                            >
                                {this.pivotItems.map(props => <PivotItem key={props.itemKey} {...props}/>)}
                            </Pivot>
                            {this.state.activeGlobalTab === 0 && (
                                <DataExploration
                                    dashboardContext={this.state.dashboardContext}
                                    selectionContext={this.selectionContext}
                                    plotlyProps={this.state.configs[DataScatterId] as IPlotlyProperty}
                                    onChange={this.onConfigChanged}
                                    messages={this.props.stringParams ? this.props.stringParams.contextualHelp : undefined}
                                />
                            )}
                            {this.state.activeGlobalTab === 1 && (
                                <FeatureImportanceBar
                                    dashboardContext={this.state.dashboardContext}
                                    selectionContext={this.selectionContext}
                                    config={this.state.configs[BarId] as IFeatureImportanceConfig}
                                    onChange={this.onConfigChanged}
                                    messages={this.props.stringParams ? this.props.stringParams.contextualHelp : undefined}
                                />
                            )}
                            {this.state.activeGlobalTab === 2 && (
                                <ExplanationExploration
                                    dashboardContext={this.state.dashboardContext}
                                    selectionContext={this.selectionContext}
                                    plotlyProps={this.state.configs[ExplanationScatterId] as IPlotlyProperty}
                                    onChange={this.onConfigChanged}
                                    messages={this.props.stringParams ? this.props.stringParams.contextualHelp : undefined}
                                />
                            )}
                            {this.state.activeGlobalTab === 3 && (
                                <FeatureImportanceWrapper
                                    dashboardContext={this.state.dashboardContext}
                                    selectionContext={this.selectionContext}
                                    config={this.state.configs[GlobalFeatureImportanceId] as IFeatureImportanceConfig}
                                    onChange={this.onConfigChanged}
                                    messages={this.props.stringParams ? this.props.stringParams.contextualHelp : undefined}
                                />
                            )}
                            {this.state.activeGlobalTab === 4 && (
                                <EbmExplanation
                                    explanationContext={this.state.dashboardContext.explanationContext}
                                    theme={''}
                                />
                            )}
                            {this.state.activeGlobalTab === 5 && (
                                <iframe srcDoc={this.state.dashboardContext.explanationContext.customVis}/>
                            )}
                        </div>
                        {this.state.dashboardContext.explanationContext.localExplanation && (<div className="local-charts-wrapper">
                            {this.state.selectedRow === undefined && (
                                <div className="local-placeholder">
                                    <div className="placeholder-text">
                                        {localization.selectPoint}
                                    </div>
                                </div>
                            )}
                            {this.state.selectedRow !== undefined && (
                                <div className="tabbed-viewer">
                                    <Pivot
                                        selectedKey={ExplanationDashboard.localTabKeys[this.state.activeLocalTab]}
                                        onLinkClick={this.handleLocalTabClick}
                                        linkFormat={PivotLinkFormat.tabs}
                                        linkSize={PivotLinkSize.normal}
                                        headersOnly={true}
                                        styles={FabricStyles.verticalTabsStyle}
                                    >
                                        <PivotItem headerText={localization.localFeatureImportance} itemKey={ExplanationDashboard.localTabKeys[0]} />
                                        {this.props.requestPredictions !== undefined && this.state.dashboardContext.explanationContext.testDataset.dataset && (
                                            <PivotItem headerText={localization.perturbationExploration} itemKey={ExplanationDashboard.localTabKeys[1]} />
                                        )} {this.props.requestPredictions !== undefined && this.state.dashboardContext.explanationContext.testDataset.dataset && (
                                            <PivotItem headerText={localization.ice} itemKey={ExplanationDashboard.localTabKeys[2]} />
                                        )}
                                    </Pivot>
                                    <div className="view-panel">
                                        <div className="local-commands">
                                            <PrimaryButton
                                                ariaDescription="clear-button"
                                                className="clear-button"
                                                onClick={this.onClearSelection}
                                                text={localization.clearSelection}
                                            />
                                        </div>
                                        {this.state.activeLocalTab === 0 && (
                                            <SinglePointFeatureImportance
                                                explanationContext={this.state.dashboardContext.explanationContext}
                                                selectedRow={this.state.selectedRow}
                                                config={this.state.configs[LocalBarId] as IBarChartConfig}
                                                onChange={this.onConfigChanged}
                                                messages={this.props.stringParams ? this.props.stringParams.contextualHelp : undefined}
                                            />
                                        )}
                                        {this.state.activeLocalTab === 1 && (
                                            <PerturbationExploration
                                                explanationContext={this.state.dashboardContext.explanationContext}
                                                invokeModel={this.props.requestPredictions}
                                                datapointIndex={+this.selectionContext.selectedIds[0]}
                                                theme={this.props.theme}
                                                messages={this.props.stringParams ? this.props.stringParams.contextualHelp : undefined}
                                            />
                                        )}
                                        {this.state.activeLocalTab === 2 && (
                                            <ICEPlot
                                                explanationContext={this.state.dashboardContext.explanationContext}
                                                invokeModel={this.props.requestPredictions}
                                                datapointIndex={+this.selectionContext.selectedIds[0]}
                                                theme={this.props.theme}
                                                messages={this.props.stringParams ? this.props.stringParams.contextualHelp : undefined}
                                             />
                                        )}
                                    </div>
                                </div>
                            )}
                        </div>
                        )}
                    </div>
                </div>
            </>
        );
    }

    private fetchExplanations(): void {
        const expContext = this.state.dashboardContext.explanationContext;
        const dataset = expContext.testDataset;
        const modelMetadata =expContext.modelMetadata;
        if (expContext.explanationGenerators.requestLocalFeatureExplanations === undefined ||
            dataset === undefined || dataset.dataset === undefined ||
            (expContext.localExplanation !== undefined && expContext.localExplanation.values !== undefined)) {
            return;
        }

        this.setState(prevState => {
            const newState = _.cloneDeep(prevState);
            newState.dashboardContext.explanationContext.localExplanation = {
                // a mock number, we can impl a progress bar if desired.
                percentComplete: 10
            };
            return newState;
        }, () => {
        this.state.dashboardContext.explanationContext.explanationGenerators.requestLocalFeatureExplanations(dataset.dataset, new AbortController().signal)
            .then((result) => {
                if (!result) {
                    return;
                }
                this.setState(prevState => {
                    const weighting = prevState.dashboardContext.weightContext.selectedKey;
                    let localFeatureMatrix = ExplanationDashboard.buildLocalFeatureMatrix(result, modelMetadata.modelType);
                    let flattenedFeatureMatrix = ExplanationDashboard.buildLocalFlattenMatrix(localFeatureMatrix, modelMetadata.modelType, dataset, weighting);
                    const newState = _.cloneDeep(prevState);
                    newState.dashboardContext.explanationContext.localExplanation = {
                        values: localFeatureMatrix,
                        flattenedValues:flattenedFeatureMatrix,
                        percentComplete: undefined
                    };
                    if (prevState.dashboardContext.explanationContext.globalExplanation === undefined) {
                        newState.dashboardContext.explanationContext.globalExplanation =
                            ExplanationDashboard.buildGlobalExplanationFromLocal(newState.dashboardContext.explanationContext.localExplanation);
                        newState.dashboardContext.explanationContext.isGlobalDerived = true;
                    }
                    return newState;
                });
            });
        });
    }

    private onClassSelect(event: React.FormEvent<IComboBox>, item: IComboBoxOption): void {
        this.setState(prevState => {
            const newWeightContext = _.cloneDeep(prevState.dashboardContext.weightContext);
            newWeightContext.selectedKey = item.key as any;

            let flattenedFeatureMatrix = ExplanationDashboard.buildLocalFlattenMatrix(
                prevState.dashboardContext.explanationContext.localExplanation.values,
                prevState.dashboardContext.explanationContext.modelMetadata.modelType,
                prevState.dashboardContext.explanationContext.testDataset,
                item.key as any);
            return {
                dashboardContext: {
                    explanationContext: {
                        modelMetadata: prevState.dashboardContext.explanationContext.modelMetadata,
                        testDataset: prevState.dashboardContext.explanationContext.testDataset,
                        localExplanation: {
                            flattenedValues: flattenedFeatureMatrix,
                            intercepts:  prevState.dashboardContext.explanationContext.localExplanation.intercepts,
                            values: prevState.dashboardContext.explanationContext.localExplanation.values
                        },
                        globalExplanation: prevState.dashboardContext.explanationContext.globalExplanation,
                        explanationGenerators: prevState.dashboardContext.explanationContext.explanationGenerators,
                        isGlobalDerived: prevState.dashboardContext.explanationContext.isGlobalDerived
                    },
                    weightContext: newWeightContext
                }
            };
        });
    }

    private onConfigChanged(newConfig: IPlotlyProperty | IFeatureImportanceConfig | IBarChartConfig, configId: string): void {
        this.setState(prevState => {
            const newConfigs = _.cloneDeep(prevState.configs);
            newConfigs[configId] = newConfig;
            return {configs: newConfigs};
        });
    }

    private handleGlobalTabClick(item: PivotItem): void {
        let index = ExplanationDashboard.globalTabKeys.indexOf(item.props.itemKey);
        if (index === -1) {
            index = 0;
        }
        this.setState({activeGlobalTab: index});

    }

    private handleLocalTabClick(item: PivotItem): void {
        let index = ExplanationDashboard.localTabKeys.indexOf(item.props.itemKey);
        if (index === -1) {
            index = 0;
        }
        this.setState({activeLocalTab: index});

    }

    private onClearSelection(): void {
        this.selectionContext.onSelect([]);
        this.setState({activeLocalTab: 0});
    }
}
