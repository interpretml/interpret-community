import React from "react";
import { JointDataset } from "../../JointDataset";
import { IExplanationModelMetadata } from "../../IExplanationContext";
import { BarChart, LoadingSpinner } from "../../SharedComponents";
import { IPlotlyProperty, AccessibleChart } from "mlchartlib";
import { localization } from "../../../Localization/localization";
import _ from "lodash";
import { DependencePlot } from "../DependencePlot/DependencePlot";
import { IGenericChartProps, ChartTypes } from "../../NewExplanationDashboard";
import { mergeStyleSets } from "@uifabric/styling";
import { SpinButton } from "office-ui-fabric-react/lib/SpinButton";
import { Slider } from "office-ui-fabric-react/lib/Slider";
import { ModelExplanationUtils } from "../../ModelExplanationUtils";
import { ComboBox, IComboBox, IComboBoxOption } from "office-ui-fabric-react/lib/ComboBox";
import { FabricStyles } from "../../FabricStyles";
import { IDropdownOption, Dropdown } from "office-ui-fabric-react/lib/Dropdown";
import { SwarmFeaturePlot } from "../SwarmFeaturePlot";
import { Cohort } from "../../Cohort";
import { FeatureImportanceBar } from "../FeatureImportanceBar/FeatureImportanceBar";
import { GlobalViolinPlot } from "../GlobalViolinPlot";
import { globalTabStyles } from "./GlobalExplanationTab.styles";
import { IGlobalSeries } from "./IGlobalSeries";
import { InteractiveLegend } from "../InteractiveLegend";
import { Icon, Text } from "office-ui-fabric-react";

export interface IGlobalBarSettings {
    topK: number;
    startingK: number;
    sortOption: string;
    includeOverallGlobal: boolean;
}

export interface IGlobalExplanationTabProps {
    globalBarSettings: IGlobalBarSettings;
    sortVector: number[];
    jointDataset: JointDataset;
    dependenceProps: IGenericChartProps;
    metadata: IExplanationModelMetadata;
    globalImportance?: number[];
    isGlobalDerivedFromLocal: boolean;
    cohorts: Cohort[];
    cohortIDs: string[];
    onChange: (props: IGlobalBarSettings) => void;
    onDependenceChange: (props: IGenericChartProps) => void;
}

export interface IGlobalExplanationtabState {
    startingK: number;
    topK: number;
    sortingSeriesIndex: number;
    sortArray: number[];
    seriesIsActive: boolean[];
    selectedCohortIndex: number;
    selectedFeatureIndex?: number;
}

export class GlobalExplanationTab extends React.PureComponent<IGlobalExplanationTabProps, IGlobalExplanationtabState> {
    private cohortSeries: IGlobalSeries[];
    private activeSeries: IGlobalSeries[];
    private readonly minK = Math.min(4, this.props.jointDataset.localExplanationFeatureCount);
    private readonly maxK = Math.min(30, this.props.jointDataset.localExplanationFeatureCount);



    constructor(props: IGlobalExplanationTabProps) {
        super(props);
        if (this.props.globalBarSettings === undefined) {
            this.setDefaultSettings(props);
        }
        
        this.state = {
            startingK: 0,
            topK: this.minK,
            selectedCohortIndex: 0,
            sortingSeriesIndex: 0,
            sortArray: ModelExplanationUtils.getSortIndices(
                this.props.cohorts[0].calculateAverageImportance()).reverse(),
            seriesIsActive: props.cohorts.map(unused => true)
        };
        this.buildGlobalSeries();
        this.buildActiveCohortSeries(this.state.sortArray);
        this.handleFeatureSelection = this.handleFeatureSelection.bind(this);
        this.setStartingK = this.setStartingK.bind(this);
        this.setSelectedCohort = this.setSelectedCohort.bind(this);
        this.setSortIndex = this.setSortIndex.bind(this);
        this.onXSet = this.onXSet.bind(this);
    }

    public componentDidUpdate(prevProps: IGlobalExplanationTabProps) {
        if (this.props.cohorts !== prevProps.cohorts) {
            this.updateIncludedCohortsOnCohortEdit();
        }
    }

    public render(): React.ReactNode {
        const classNames = globalTabStyles();
        
        const maxStartingK = Math.max(0, this.props.jointDataset.localExplanationFeatureCount - this.state.topK);
        if (this.props.globalBarSettings === undefined) {
            return (<div/>);
        }
        const cohortOptions: IDropdownOption[] = this.props.cohorts.map((cohort, index) => {return {key: index, text: cohort.name};});
        const featureOptions: IDropdownOption[] = [];
        for( let i =0; i < this.props.jointDataset.datasetFeatureCount; i++) {
            const key = JointDataset.DataLabelRoot + i.toString();
            featureOptions.push({key, text: this.props.jointDataset.metaDict[key].label});
        }
        return (
        <div className={classNames.page}>
            <div className={classNames.infoWithText}>
                <Icon iconName="Info" className={classNames.infoIcon}/>
                <Text variant="medium" className={classNames.helperText}>{localization.GlobalTab.helperText}</Text>
            </div>
            <div className={classNames.globalChartControls}>
                {/* <SpinButton
                    className={classNames.topK}
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
                    min={this.minK}
                    max={this.maxK}
                    value={this.state.topK.toString()}
                    onIncrement={this.setNumericValue.bind(this, 1, this.maxK, this.minK)}
                    onDecrement={this.setNumericValue.bind(this, -1, this.maxK, this.minK)}
                    onValidate={this.setNumericValue.bind(this, 0, this.maxK, this.minK)}
                /> */}
                <Text variant="medium" className={classNames.sliderLabel}>{localization.formatString(localization.GlobalTab.topAtoB, this.state.startingK + 1, this.state.startingK + this.state.topK)}</Text>
                <Slider
                    className={classNames.startingK}
                    ariaLabel={localization.AggregateImportance.topKFeatures}
                    max={maxStartingK}
                    min={0}
                    step={1}
                    value={this.state.startingK}
                    onChange={this.setStartingK}
                    showValue={false}
                />
            </div>
            <div className={classNames.globalChartWithLegend}>
                <FeatureImportanceBar
                    jointDataset={this.props.jointDataset}
                    yAxisLabels={[localization.GlobalTab.aggregateFeatureImportance]}
                    sortArray={this.state.sortArray}
                    startingK={this.state.startingK}
                    unsortedX={this.props.metadata.featureNamesAbridged}
                    unsortedSeries={this.activeSeries}
                    topK={this.props.globalBarSettings.topK}
                    onFeatureSelection={this.handleFeatureSelection}
                    selectedFeatureIndex={this.state.selectedFeatureIndex}
                />
                <div className={classNames.legendAndSort}>
                    <Text variant={"mediumPlus"} block className={classNames.cohortLegend}>{localization.GlobalTab.datasetCohorts}</Text>
                    <Text variant={"small"} className={classNames.legendHelpText}>{localization.GlobalTab.legendHelpText}</Text>
                    <InteractiveLegend
                        items={this.cohortSeries.map((row, rowIndex) => {
                            return {
                                name: row.name,
                                color: FabricStyles.fabricColorPalette[row.colorIndex],
                                activated: this.state.seriesIsActive[rowIndex],
                                onClick: this.toggleActivation.bind(this, rowIndex)
                            }
                        })}
                    />
                    <Text variant={"medium"} className={classNames.cohortLegend}>{localization.GlobalTab.sortBy}</Text>
                    <Dropdown 
                        styles={{ dropdown: { width: 150 } }}
                        options={cohortOptions}
                        selectedKey={this.state.sortingSeriesIndex}
                        onChange={this.setSortIndex}
                    />
                </div>
            </div>
            <div className={classNames.secondaryChartAndLegend}>
                <DependencePlot 
                    chartProps={this.props.dependenceProps}
                    cohortIndex={this.state.selectedCohortIndex}
                    cohort={this.props.cohorts[this.state.selectedCohortIndex]}
                    jointDataset={this.props.jointDataset}
                    metadata={this.props.metadata}
                    onChange={this.props.onDependenceChange}
                />
                <div className={classNames.legendAndSort}>
                    <Text variant={"mediumPlus"} block className={classNames.cohortLegend}>{localization.GlobalTab.viewDependencePlotFor}</Text>
                    {featureOptions && (<Dropdown 
                        styles={{ dropdown: { width: 150 } }}
                        options={featureOptions}
                        selectedKey={this.props.dependenceProps ? this.props.dependenceProps.xAxis.property : undefined}
                        onChange={this.onXSet}
                    />)}
                    {/* {(this.state.xDialogOpen) && (
                        <AxisConfigDialog 
                            jointDataset={this.props.jointDataset}
                            orderedGroupTitles={[ColumnCategories.dataset]}
                            selectedColumn={this.props.chartProps.xAxis}
                            canBin={false}
                            mustBin={false}
                            canDither={true}
                            onAccept={this.onXSet}
                            onCancel={this.setXOpen.bind(this, false)}
                            target={this._xButtonId}
                        />
                    )} */}
                    <Text variant={"mediumPlus"} block className={classNames.cohortLegend}>{localization.GlobalTab.datasetCohortSelector}</Text>
                    {cohortOptions && (<Dropdown 
                        styles={{ dropdown: { width: 150 } }}
                        options={cohortOptions}
                        selectedKey={this.state.selectedCohortIndex}
                        onChange={this.setSelectedCohort}
                    />)}
                </div>
            </div>
            {/* {this.state.secondChart === 'swarm' && (<GlobalViolinPlot
                jointDataset={this.props.jointDataset}
                metadata={this.props.metadata}
                cohort={this.props.cohorts[this.state.selectedCohortIndex]}
                topK={this.props.globalBarSettings.topK}
                startingK={this.state.startingK}
                sortVector={this.state.sortArray}
            />)} */}
        </div>);
    }

    private setSelectedCohort(event: React.FormEvent<HTMLDivElement>, item: IDropdownOption): void {
        this.setState({selectedCohortIndex: item.key as number});
    }

    private setStartingK(newValue: number): void {
        this.setState({startingK: newValue});
    }

    private readonly setNumericValue = (delta: number, max: number, min: number, stringVal: string): string | void => {
        if (delta === 0) {
            const number = +stringVal;
            if (!Number.isInteger(number)
                || number > max || number < min) {
                return this.state.startingK.toString();
            }
            this.setStartingK(number);
        } else {
            const prevVal = this.state.startingK;
            const newVal = prevVal + delta;
            if (newVal > max || newVal < min) {
                return prevVal.toString();
            }
            this.setStartingK(newVal);
        }
    }

    private toggleActivation(index: number): void {
        const seriesIsActive = [...this.state.seriesIsActive];
        seriesIsActive[index] = !seriesIsActive[index];
        this.buildActiveCohortSeries(seriesIsActive);
        this.setState({seriesIsActive});
    }

    private buildGlobalSeries(): void {
        this.cohortSeries = this.props.cohorts.map((cohort, i) => {
            return {
                name: cohort.name,
                unsortedIndividualY: cohort.transposedLocalFeatureImportances(),
                unsortedAggregateY: cohort.calculateAverageImportance(),
                colorIndex: i
            }
        });
    }

    // This can probably be done cheaper by passing the active array to the charts, and zeroing
    // the series in the plotlyProps. Later optimization.
    private buildActiveCohortSeries(activeArray): void {
        this.activeSeries = activeArray.map((isActive, index) => {
            if (isActive) {
                return this.cohortSeries[index];
            }
        }).filter(series => !!series);
    }

    private updateIncludedCohortsOnCohortEdit(): void {
        let selectedCohortIndex = this.state.selectedCohortIndex;
        if (selectedCohortIndex >= this.props.cohorts.length) {
            selectedCohortIndex = 0;
        }
        const seriesIsActive: boolean[] = this.props.cohorts.map(unused => true);
        this.buildGlobalSeries();
        this.buildActiveCohortSeries(seriesIsActive);
        this.setState({selectedCohortIndex, seriesIsActive});
    }

    private setDefaultSettings(props: IGlobalExplanationTabProps): void {
        const result: IGlobalBarSettings = {} as IGlobalBarSettings;
        result.topK = Math.min(this.props.jointDataset.localExplanationFeatureCount, 4);
        result.startingK = 0;
        result.sortOption = "global";
        result.includeOverallGlobal = !this.props.isGlobalDerivedFromLocal;
        this.props.onChange(result);
    }

    private setSortIndex(event: React.FormEvent<HTMLDivElement>, item: IDropdownOption): void {
        const newIndex = item.key as number;
        const sortArray = ModelExplanationUtils.getSortIndices(this.cohortSeries[newIndex].unsortedAggregateY).reverse()
        this.setState({sortingSeriesIndex: newIndex, sortArray});
    }

    private onXSet(event: React.FormEvent<HTMLDivElement>, item: IDropdownOption): void {
        const key = item.key as string;
        const index = this.props.jointDataset.metaDict[key].index;
        this.handleFeatureSelection(this.state.selectedCohortIndex, index);
    }

    private handleFeatureSelection(cohortIndex: number, featureIndex: number): void {
        // set to dependence plot initially, can be changed if other feature importances available
        const xKey = JointDataset.DataLabelRoot + featureIndex.toString();
        const xIsDithered = this.props.jointDataset.metaDict[xKey].treatAsCategorical;
        const yKey = JointDataset.ReducedLocalImportanceRoot + featureIndex.toString();
        const chartProps: IGenericChartProps = {
            chartType: ChartTypes.Scatter,
            xAxis: {
                property: xKey,
                options: {
                    dither: xIsDithered,
                    bin: false
                }
            },
            yAxis: {
                property: yKey,
                options: {}
            }
        };
        this.props.onDependenceChange(chartProps);
        this.setState({selectedCohortIndex: cohortIndex});
    }
}