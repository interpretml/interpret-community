import React from "react";
import { JointDataset } from "../JointDataset";
import { IExplanationModelMetadata } from "../IExplanationContext";
import { IFilterContext } from "../Interfaces/IFilter";
import { BarChart, LoadingSpinner } from "../SharedComponents";
import { IPlotlyProperty, AccessibleChart } from "mlchartlib";
import { localization } from "../../Localization/localization";
import _ from "lodash";
import { DependencePlot } from "./DependencePlot";
import { IGenericChartProps } from "../NewExplanationDashboard";
import { mergeStyleSets } from "@uifabric/styling";
import { SpinButton } from "office-ui-fabric-react/lib/SpinButton";
import { Slider } from "office-ui-fabric-react/lib/Slider";
import { ModelExplanationUtils } from "../ModelExplanationUtils";
import { ComboBox, IComboBox, IComboBoxOption } from "office-ui-fabric-react/lib/ComboBox";
import { FabricStyles } from "../FabricStyles";
import { IDropdownOption } from "office-ui-fabric-react/lib/Dropdown";

export interface IGlobalBarSettings {
    topK: number;
    startingK: number;
    sortOption: string;
    includeOverallGlobal: boolean;
    sortIndexVector: number[];
}

export interface IGlobalExplanationTabProps {
    globalBarSettings: IGlobalBarSettings;
    // selectionContext: SelectionContext;
    theme?: string;
    // messages?: HelpMessageDict;
    jointDataset: JointDataset;
    dependenceProps: IGenericChartProps;
    metadata: IExplanationModelMetadata;
    globalImportance: number[];
    subsetAverageImportance: number[];
    isGlobalDerivedFromLocal: boolean;
    filterContext: IFilterContext;
    onChange: (props: IGlobalBarSettings) => void;
    onDependenceChange: (props: IGenericChartProps) => void;
}

export interface IGlobalExplanationtabState {
    secondChart: string;
    plotlyProps: IPlotlyProperty;
}

export class GlobalExplanationTab extends React.PureComponent<IGlobalExplanationTabProps, IGlobalExplanationtabState> {
    private static readonly classNames = mergeStyleSets({
        page: {
            display: "contents"
        },
        globalChartControls: {
            display: "flex",
            flexDirection: "row"
        },
        topK: {
            maxWidth: "200px"
        },
        startingK: {
            flex: 1
        },
        chartTypeDropdown: {
            margin: "0 5px 0 0"
        }
    });
    private chartOptions: IDropdownOption[] = [];
    constructor(props: IGlobalExplanationTabProps) {
        super(props);
        if (this.props.globalBarSettings === undefined) {
            this.setDefaultSettings(props);
        }
        if (this.props.jointDataset.localExplanationFeatureCount > 0) {
            // this.chartOptions.push({key: 'swarm', text: 'Swarm plot'});
            if (this.props.jointDataset.datasetFeatureCount > 0) {
                this.chartOptions.push({key: 'depPlot', text: 'Dependence plot'});
            }
        }
        this.state = {
            secondChart: this.chartOptions[0].key as string,
            plotlyProps: undefined
        };
        this.setStartingK = this.setStartingK.bind(this);
        this.onSecondaryChartChange = this.onSecondaryChartChange.bind(this);
    }
    public render(): React.ReactNode {
        if (this.props.globalBarSettings === undefined) {
            return (<div/>);
        }
        if (this.state.plotlyProps === undefined) {
            this.loadProps();
            return <LoadingSpinner/>;
        }
        const minK = Math.min(4, this.props.jointDataset.localExplanationFeatureCount);
        const maxK = Math.min(30, this.props.jointDataset.localExplanationFeatureCount);
        const maxStartingK = Math.max(0, this.props.jointDataset.localExplanationFeatureCount - this.props.globalBarSettings.topK);
        const relayoutArg = {'xaxis.range': [this.props.globalBarSettings.startingK - 0.5, this.props.globalBarSettings.startingK + this.props.globalBarSettings.topK - 0.5]};
        const plotlyProps = this.state.plotlyProps;
        _.set(plotlyProps, 'layout.xaxis.range', [this.props.globalBarSettings.startingK - 0.5, this.props.globalBarSettings.startingK + this.props.globalBarSettings.topK - 0.5]);
        return (
        <div className={GlobalExplanationTab.classNames.page}>
            <div className={GlobalExplanationTab.classNames.globalChartControls}>
                <SpinButton
                    className={GlobalExplanationTab.classNames.topK}
                    styles={{
                        spinButtonWrapper: {maxWidth: "150px"},
                        labelWrapper: { alignSelf: "center"},
                        root: {
                            display: "inline-flex",
                            float: "right",
                            selectors: {
                                "> div": {
                                    maxWidth: "160px"
                                }
                            }
                        }
                    }}
                    label={localization.AggregateImportance.topKFeatures}
                    min={minK}
                    max={maxK}
                    value={this.props.globalBarSettings.topK.toString()}
                    onIncrement={this.setNumericValue.bind(this, 1, maxK, minK)}
                    onDecrement={this.setNumericValue.bind(this, -1, maxK, minK)}
                    onValidate={this.setNumericValue.bind(this, 0, maxK, minK)}
                />
                <Slider
                    className={GlobalExplanationTab.classNames.startingK}
                    ariaLabel={localization.AggregateImportance.topKFeatures}
                    max={maxStartingK}
                    min={0}
                    step={1}
                    value={this.props.globalBarSettings.startingK}
                    onChange={this.setStartingK}
                    showValue={true}
                />
            </div>
            <AccessibleChart
                plotlyProps={plotlyProps}
                sharedSelectionContext={undefined}
                theme={this.props.theme}
                relayoutArg={relayoutArg as any}
                onSelection={undefined}
            />
            <ComboBox
                className={GlobalExplanationTab.classNames.chartTypeDropdown}
                label={localization.BarChart.sortBy}
                selectedKey={this.state.secondChart}
                onChange={this.onSecondaryChartChange}
                options={this.chartOptions}
                ariaLabel={"chart selector"}
                useComboBoxAsMenuWidth={true}
                styles={FabricStyles.smallDropdownStyle}
            />
            {this.state.secondChart === 'depPlot' && (<DependencePlot 
                chartProps={this.props.dependenceProps}
                jointDataset={this.props.jointDataset}
                metadata={this.props.metadata}
                onChange={this.props.onDependenceChange}
            />)}
        </div>);
    }

    private loadProps(): void {
        setTimeout(() => {
            const props = this.buildBarPlotlyProps();
            this.setState({plotlyProps: props});
            }, 1);
    }

    private buildBarPlotlyProps(): IPlotlyProperty {
        const sortedIndexVector = this.props.globalBarSettings.sortIndexVector;
        const baseSeries = {
            config: { displaylogo: false, responsive: true, displayModeBar: false } as Plotly.Config,
            data: [],
            layout: {
                autosize: true,
                dragmode: false,
                barmode: 'group',
                font: {
                    size: 10
                },
                margin: {t: 10, r: 10, b: 30},
                hovermode: 'closest',
                xaxis: {
                    automargin: true
                },
                yaxis: {
                    automargin: true,
                    title: localization.featureImportance
                },
                showlegend: this.props.globalBarSettings.includeOverallGlobal
            } as any
        };

        const x = sortedIndexVector.map((unused, index) => index);
        const y = sortedIndexVector.map(index => this.props.subsetAverageImportance[index]);

        baseSeries.data.push({
            orientation: 'v',
            type: 'bar',
            name: 'Absolute Average of Subset',
            x,
            y
        } as any);

        if (this.props.globalBarSettings.includeOverallGlobal) {
            baseSeries.data.push({
                orientation: 'v',
                type: 'bar',
                name: 'Global Importance',
                x,
                y: sortedIndexVector.map(index => this.props.globalImportance[index])
            } as any);
        }

        const ticktext = sortedIndexVector.map(i =>this.props.metadata.featureNamesAbridged[i]);
        const tickvals = sortedIndexVector.map((val, index) => index);

        _.set(baseSeries, 'layout.xaxis.ticktext', ticktext);
        _.set(baseSeries, 'layout.xaxis.tickvals', tickvals);
        return baseSeries;
    }

    private readonly setNumericValue = (delta: number, max: number, min: number, stringVal: string): string | void => {
        const newProps = this.props.globalBarSettings;
        if (delta === 0) {
            const number = +stringVal;
            if (!Number.isInteger(number)
                || number > max || number < min) {
                return this.props.globalBarSettings.topK.toString();
            }
            newProps.topK = number;
            this.props.onChange(newProps);
        } else {
            const prevVal = this.props.globalBarSettings.topK;
            const newVal = prevVal + delta;
            if (newVal > max || newVal < min) {
                return prevVal.toString();
            }
            newProps.topK = newVal
            this.props.onChange(newProps);
        }
        this.forceUpdate();
    }

    private setStartingK(newValue: number): void {
        const newConfig = this.props.globalBarSettings;
        newConfig.startingK = newValue;
        this.props.onChange(newConfig);
        this.forceUpdate();
    }

    private setDefaultSettings(props: IGlobalExplanationTabProps): void {
        const result: IGlobalBarSettings = {} as IGlobalBarSettings;
        result.topK = Math.min(this.props.jointDataset.localExplanationFeatureCount, 4);
        result.startingK = 0;
        result.sortOption = "global";
        result.includeOverallGlobal = this.props.filterContext.filters.length > 0 || !this.props.isGlobalDerivedFromLocal;
        result.sortIndexVector = ModelExplanationUtils.getSortIndices(this.props.globalImportance).reverse();
        this.props.onChange(result);
    }

    private onSecondaryChartChange(event: React.FormEvent<IComboBox>, item: IComboBoxOption): void {
        this.setState({secondChart: item.key as string});
    }
}