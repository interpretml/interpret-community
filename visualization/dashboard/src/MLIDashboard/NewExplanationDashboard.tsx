import React from "react";
import { IExplanationDashboardProps, IMultiClassLocalFeatureImportance, ISingleClassLocalFeatureImportance } from "./Interfaces";
import { IFilter, IFilterContext } from "./Interfaces/IFilter";
import { JointDataset } from "./JointDataset";
import { IGenericChartProps } from "./Controls/ChartWithControls";
import { ModelMetadata } from "mlchartlib";
import { localization } from "../Localization/localization";
import { IExplanationModelMetadata, ModelTypes } from "./IExplanationContext";
import * as memoize from "memoize-one";
import { IPivot, IPivotItemProps, PivotItem, Pivot, PivotLinkSize } from "office-ui-fabric-react/lib/Pivot";
import _ from "lodash";
import { NewDataExploration, DataScatterId } from "./Controls/Scatter/NewDataExploration";

export interface INewExplanationDashboardState {
    filters: IFilter[];
    activeGlobalTab: globalTabKeys;
    jointDataset: JointDataset;
    modelMetadata: IExplanationModelMetadata;
    chartConfigs: {[key: string]: IGenericChartProps};
}

enum globalTabKeys {
    dataExploration ="dataExploration",
    explanationTab = "explanationTab",
    whatIfTab = "whatIfTab"
}

export class NewExplanationDashboard extends React.PureComponent<IExplanationDashboardProps, INewExplanationDashboardState> {
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
            activeGlobalTab: globalTabKeys.dataExploration,
            jointDataset,
            modelMetadata,
            chartConfigs: {}
        };
    }

    private pivotItems: IPivotItemProps[] = [];
    private pivotRef: IPivot;
    constructor(props: IExplanationDashboardProps) {
        super(props);
        this.onConfigChanged = this.onConfigChanged.bind(this);
        this.handleGlobalTabClick = this.handleGlobalTabClick.bind(this);
        this.addFilter = this.addFilter.bind(this);
        this.deleteFilter = this.deleteFilter.bind(this);
        if (this.props.locale) {
            localization.setLanguage(this.props.locale);
        }
        this.state = NewExplanationDashboard.buildInitialExplanationContext(props);

        if (this.state.jointDataset.hasDataset) {
            this.pivotItems.push({headerText: localization.dataExploration, itemKey: globalTabKeys.dataExploration});
        }
        if (this.state.jointDataset.localExplanationFeatureCount > 0) {
            this.pivotItems.push({headerText: localization.globalImportance, itemKey: globalTabKeys.explanationTab});
        }
        if (this.state.jointDataset.localExplanationFeatureCount > 0 && this.state.jointDataset.hasDataset && this.props.requestPredictions) {
            this.pivotItems.push({headerText: localization.explanationExploration, itemKey: globalTabKeys.whatIfTab});
        }
    }

    render(): React.ReactNode {
        const filterContext: IFilterContext = {
            filters: this.state.filters,
            onAdd: this.addFilter,
            onDelete: this.deleteFilter,
            onUpdate: this.updateFilter
        }
        this.state.jointDataset.applyFilters(this.state.filters);
        return (
            <>
                <div className="explainerDashboard">
                    <div className="charts-wrapper">
                        <div className="global-charts-wrapper">
                            <Pivot
                                componentRef={ref => {this.pivotRef = ref;}}
                                selectedKey={this.state.activeGlobalTab}
                                onLinkClick={this.handleGlobalTabClick}
                                linkSize={PivotLinkSize.normal}
                                headersOnly={true}
                            >
                                {this.pivotItems.map(props => <PivotItem key={props.itemKey} {...props}/>)}
                            </Pivot>
                            {this.state.activeGlobalTab === globalTabKeys.dataExploration && (
                                <NewDataExploration
                                    jointDataset={this.state.jointDataset}
                                    theme={this.props.theme}
                                    metadata={this.state.modelMetadata}
                                    chartProps={this.state.chartConfigs[DataScatterId] as IGenericChartProps}
                                    onChange={this.onConfigChanged}
                                    filterContext={filterContext}
                                />
                                // <DataExploration
                                //     dashboardContext={this.state.dashboardContext}
                                //     theme={this.props.theme}
                                //     selectionContext={this.selectionContext}
                                //     plotlyProps={this.state.configs[DataScatterId] as IPlotlyProperty}
                                //     onChange={this.onConfigChanged}
                                //     messages={this.props.stringParams ? this.props.stringParams.contextualHelp : undefined}
                                // />
                            )}
                            {this.state.activeGlobalTab === globalTabKeys.explanationTab && (
                                <div>TODO</div>
                            )}
                            {this.state.activeGlobalTab === globalTabKeys.whatIfTab && (
                                <div>TODO</div>
                            )}
                        </div>
                    </div>
                </div>
            </>
        );
    }

    private onConfigChanged(newConfig: IGenericChartProps, configId: string): void {
        this.setState(prevState => {
            const newConfigs = _.cloneDeep(prevState.chartConfigs);
            newConfigs[configId] = newConfig;
            return {chartConfigs: newConfigs};
        });
    }

    private handleGlobalTabClick(item: PivotItem): void {
        let index: globalTabKeys = globalTabKeys[item.props.itemKey];
        this.setState({activeGlobalTab: index});

    }

    private addFilter(newFilter: IFilter): void {
        this.setState(prevState => {
            const filters = [...prevState.filters];
            filters.push(newFilter);
            return {filters}
        });
    }

    private deleteFilter(index: number): void {
        this.setState(prevState => {
            if (prevState.filters.length < index || index < 0) {
                return;
            }
            prevState.filters.splice(index, 1);
            return prevState;
        });
    }

    private updateFilter(filter: IFilter, index: number): void {
        this.setState(prevState => {
            const filters = [...prevState.filters];
            filters[index] = filter;
            return {filters};
        });
    }
}