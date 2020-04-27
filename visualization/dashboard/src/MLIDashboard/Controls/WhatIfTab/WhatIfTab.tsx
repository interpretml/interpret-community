import { IProcessedStyleSet } from "@uifabric/styling";
import _ from "lodash";
import { AccessibleChart, IPlotlyProperty, PlotlyMode } from "mlchartlib";
import { ChoiceGroup, IChoiceGroupOption, Icon, Slider, Text } from "office-ui-fabric-react";
import { DefaultButton, IconButton, PrimaryButton } from "office-ui-fabric-react/lib/Button";
import { Dropdown, IDropdownOption } from "office-ui-fabric-react/lib/Dropdown";
import { SearchBox } from "office-ui-fabric-react/lib/SearchBox";
import { TextField } from "office-ui-fabric-react/lib/TextField";
import React from "react";
import { localization } from "../../../Localization/localization";
import { Cohort } from "../../Cohort";
import { FabricStyles } from "../../FabricStyles";
import { IExplanationModelMetadata, ModelTypes } from "../../IExplanationContext";
import { ColumnCategories, JointDataset } from "../../JointDataset";
import { ModelExplanationUtils } from "../../ModelExplanationUtils";
import { ChartTypes, IGenericChartProps, ISelectorConfig } from "../../NewExplanationDashboard";
import { AxisConfigDialog } from "../AxisConfigurationDialog/AxisConfigDialog";
import { FeatureImportanceBar } from "../FeatureImportanceBar/FeatureImportanceBar";
import { IGlobalSeries } from "../GlobalExplanationTab/IGlobalSeries";
import { InteractiveLegend } from "../InteractiveLegend";
import { MultiICEPlot } from "../MultiICEPlot";
import { IWhatIfTabStyles, whatIfTabStyles } from "./WhatIfTab.styles";

export interface IWhatIfTabProps {
    theme: any;
    jointDataset: JointDataset;
    metadata: IExplanationModelMetadata;
    cohorts: Cohort[];
    chartProps: IGenericChartProps;
    onChange: (config: IGenericChartProps) => void;
    invokeModel: (data: any[], abortSignal: AbortSignal) => Promise<any[]>;
}

export interface IWhatIfTabState {
    isPanelOpen: boolean;
    xDialogOpen: boolean;
    yDialogOpen: boolean;
    selectedWhatIfRootIndex: number;
    editingDataCustomIndex?: number;
    customPoints: Array<{ [key: string]: any }>;
    selectedCohortIndex: number;
    filteredFeatureList: Array<{ key: string, label: string }>;
    request?: AbortController;
    selectedPointsIndexes: number[];
    pointIsActive: boolean[];
    customPointIsActive: boolean[];
    startingK: number;
    topK: number;
    sortArray: number[];
    sortingSeriesIndex: number;
    secondaryChartChoice: string;
}

interface ISelectedRowInfo {
    name: string;
    color: string;
    rowData: any[];
    rowImportances?: number[];
    isCustom: boolean;
    index?: number;
}

export class WhatIfTab extends React.PureComponent<IWhatIfTabProps, IWhatIfTabState> {
    private static readonly MAX_SELECTION = 2;
    private static readonly colorPath = "Color";
    private static readonly namePath = "Name";
    private static readonly IceKey = "ice";
    private static readonly featureImportanceKey = "feature-importance";
    private static readonly secondaryPlotChoices: IChoiceGroupOption[] = [
        {key: WhatIfTab.featureImportanceKey, text: localization.WhatIfTab.featureImportancePlot},
        {key: WhatIfTab.IceKey, text: localization.WhatIfTab.icePlot}
    ];

    public static basePlotlyProperties: IPlotlyProperty = {
        config: { displaylogo: false, responsive: true, displayModeBar: false },
        data: [{}],
        layout: {
            dragmode: false,
            autosize: true,
            font: {
                size: 10
            },
            margin: {
                t: 10,
                l: 10,
                b: 20,
                r:0
            },
            hovermode: "closest",
            showlegend: false,
            yaxis: {
                automargin: true
            },
        } as any
    };

    private buildCustomRowSeries(customRows: Array<{ [key: string]: any }>): any[] {
        return customRows.map((row, i) => {
            return {
                name: row[WhatIfTab.namePath],
                unsortedFeatureValues: JointDataset.datasetSlice(row, this.props.jointDataset.metaDict, this.props.jointDataset.localExplanationFeatureCount),
                unsortedY: undefined,
                color: row[WhatIfTab.colorPath],
                onEdit: this.setTemporaryPointToCustomPoint.bind(this, i),
                onDelete: this.removeCustomPoint.bind(this, i)
            };
        });
    }

    private readonly _xButtonId = "x-button-id";
    private readonly _yButtonId = "y-button-id";

    private readonly featureList: Array<{ key: string, label: string }> = new Array(this.props.jointDataset.datasetFeatureCount)
        .fill(0).map((unused, colIndex) => {
            const key = JointDataset.DataLabelRoot + colIndex.toString();
            const meta = this.props.jointDataset.metaDict[key];
            return { key, label: meta.label.toLowerCase() };
        });

    private includedFeatureImportance: IGlobalSeries[] = [];
    private selectedFeatureImportance: IGlobalSeries[] = [];
    private selectedDatapoints: any[][] = [];
    private customDatapoints: any[][] = [];
    private testableDatapoints: any[][] = [];
    private temporaryPoint: { [key: string]: any };
    private testableDatapointColors: string[] = FabricStyles.fabricColorPalette;

    constructor(props: IWhatIfTabProps) {
        super(props);
        if (props.chartProps === undefined) {
            this.generateDefaultChartAxes();
        }
        this.state = {
            isPanelOpen: true,
            xDialogOpen: false,
            yDialogOpen: false,
            selectedWhatIfRootIndex: 0,
            editingDataCustomIndex: undefined,
            customPoints: [],
            selectedCohortIndex: 0,
            request: undefined,
            filteredFeatureList: this.featureList,
            selectedPointsIndexes: [],
            pointIsActive: [],
            customPointIsActive: [],
            startingK: 0,
            topK: 4,
            sortArray: [],
            sortingSeriesIndex: undefined,
            secondaryChartChoice: WhatIfTab.featureImportanceKey
        };
        this.temporaryPoint = this.createCopyOfFirstRow();
        //this.seriesOfRows = this.buildJoinedSelectedRows(this.state.selectedPointsIndexes, this.state.customPoints);
        this.dismissPanel = this.dismissPanel.bind(this);
        this.openPanel = this.openPanel.bind(this);
        this.onXSet = this.onXSet.bind(this);
        this.onYSet = this.onYSet.bind(this);
        this.setCustomRowProperty = this.setCustomRowProperty.bind(this);
        this.setCustomRowPropertyDropdown = this.setCustomRowPropertyDropdown.bind(this);
        this.savePoint = this.savePoint.bind(this);
        this.saveAsPoint = this.saveAsPoint.bind(this);
        this.selectPointFromChart = this.selectPointFromChart.bind(this);
        this.filterFeatures = this.filterFeatures.bind(this);
        this.setSelectedCohort = this.setSelectedCohort.bind(this);
        this.setStartingK = this.setStartingK.bind(this);
        this.setSecondaryChart = this.setSecondaryChart.bind(this);
        this.setSelectedIndex = this.setSelectedIndex.bind(this);
        this.fetchData = _.debounce(this.fetchData.bind(this), 400);
    }

    public componentDidUpdate(prevProps: IWhatIfTabProps, prevState: IWhatIfTabState): void {
        let sortingSeriesIndex = this.state.sortingSeriesIndex;
        let sortArray = this.state.sortArray;
        const selectionsAreEqual = _.isEqual(this.state.selectedPointsIndexes, prevState.selectedPointsIndexes);
        const activePointsAreEqual = _.isEqual(this.state.pointIsActive, prevState.pointIsActive);
        const customPointsAreEqual = this.state.customPoints === prevState.customPoints;
        const customActivePointsAreEqual = _.isEqual(this.state.customPointIsActive, prevState.customPointIsActive);
        if (!selectionsAreEqual) {
            this.selectedFeatureImportance = this.state.selectedPointsIndexes.map((rowIndex, colorIndex) => {
                const row = this.props.jointDataset.getRow(rowIndex);
                return {
                    colorIndex: colorIndex,
                    id: rowIndex,
                    name: localization.formatString(localization.WhatIfTab.rowLabel, rowIndex.toString()) as string,
                    unsortedFeatureValues: JointDataset.datasetSlice(row, this.props.jointDataset.metaDict, this.props.jointDataset.localExplanationFeatureCount),
                    unsortedAggregateY: JointDataset.localExplanationSlice(row, this.props.jointDataset.localExplanationFeatureCount) as number[],
                }
            });
            this.selectedDatapoints = this.state.selectedPointsIndexes.map(rowIndex => {
                const row = this.props.jointDataset.getRow(rowIndex);
                return JointDataset.datasetSlice(row, this.props.jointDataset.metaDict, this.props.jointDataset.datasetFeatureCount);
            });
            if (!this.state.selectedPointsIndexes.includes(this.state.sortingSeriesIndex)) {
                if (this.state.selectedPointsIndexes.length !== 0) {
                    sortingSeriesIndex = 0;
                    sortArray = ModelExplanationUtils.getSortIndices(this.selectedFeatureImportance[0].unsortedAggregateY).reverse();
                } else {
                    sortingSeriesIndex = undefined;
                }
            }
        }
        if (!customPointsAreEqual) {
            this.customDatapoints = this.state.customPoints.map(row => {
                return JointDataset.datasetSlice(row, this.props.jointDataset.metaDict, this.props.jointDataset.datasetFeatureCount);
            });
        }
        if (!selectionsAreEqual || !activePointsAreEqual || !customPointsAreEqual || !customActivePointsAreEqual) {
            this.includedFeatureImportance = this.state.pointIsActive.map((isActive, i) => {
                if (isActive) {
                    return this.selectedFeatureImportance[i];
                }
            }).filter(item => !!item);
            const includedColors = this.includedFeatureImportance.map(item => FabricStyles.fabricColorPalette[item.colorIndex]);
            const includedRows = this.state.pointIsActive.map((isActive, i) => {
                if (isActive) {
                    return this.selectedDatapoints[i];
                }
            }).filter(item => !!item);
            const includedCustomRows = this.state.customPointIsActive.map((isActive, i) => {
                if (isActive) {
                    includedColors.push(FabricStyles.fabricColorPalette[WhatIfTab.MAX_SELECTION + i + 1])
                    return this.customDatapoints[i];
                }
            }).filter(item => !!item);
            this.testableDatapoints = [...includedRows, ...includedCustomRows];
            this.testableDatapointColors = includedColors;
            this.forceUpdate();
        }
        this.setState({ sortingSeriesIndex, sortArray })
    }

    public render(): React.ReactNode {
        if (this.props.chartProps === undefined) {
            return (<div />);
        }
        const plotlyProps = this.generatePlotlyProps(
            this.props.jointDataset,
            this.props.chartProps,
            this.props.cohorts[this.state.selectedCohortIndex]
        );
        const rowOptions: IDropdownOption[ ]= this.props.cohorts[this.state.selectedCohortIndex].unwrap(JointDataset.IndexLabel).map(index => {
            return {key: index, text: localization.formatString(localization.WhatIfTab.rowLabel, index.toString()) as string};
        });
        const classNames = whatIfTabStyles();
        const cohortOptions: IDropdownOption[] = this.props.cohorts.map((cohort, index) => { return { key: index, text: cohort.name }; });
        return (<div className={classNames.page}>
            <div className={classNames.infoWithText}>
                <Icon iconName="Info" className={classNames.infoIcon} />
                <Text variant="medium" className={classNames.helperText}>{localization.WhatIfTab.helperText}</Text>
            </div>
            <div className={classNames.mainArea}>
            <div className={this.state.isPanelOpen ?
                classNames.expandedPanel :
                classNames.collapsedPanel}>
                {this.state.isPanelOpen && this.props.invokeModel === undefined && (
                    <Text>{localization.WhatIfTab.panelPlaceholder}</Text>
                )}
                {this.state.isPanelOpen && this.props.invokeModel !== undefined && (<div>
                    <div className={classNames.panelIconAndLabel}>
                        <IconButton
                            iconProps={{ iconName: "ChevronRight" }}
                            onClick={this.dismissPanel}
                            className={classNames.blackIcon}
                        />
                        <Text variant={"medium"} className={classNames.boldText}>{localization.WhatIfTab.whatIfDatapoint}</Text>
                    </div>
                    <div className={classNames.upperWhatIfPanel}>
                        <Text variant={"small"} className={classNames.legendHelpText}>{localization.WhatIfTab.whatIfHelpText}</Text>
                        <Dropdown 
                            label={localization.WhatIfTab.indexLabel}
                            options={rowOptions}
                            selectedKey={this.state.selectedWhatIfRootIndex}
                            onChange={this.setSelectedIndex}
                        />
                        {this.buildExistingPredictionLabels(classNames)}
                        <TextField
                            label={localization.WhatIfTab.whatIfNameLabel}
                            value={this.temporaryPoint[WhatIfTab.namePath]}
                            onChange={this.setCustomRowProperty.bind(this, WhatIfTab.namePath, true)}
                            styles={{ fieldGroup: { width: 200 } }}
                        />
                    </div>
                    {this.buildCustomPredictionLabels(classNames)}
                    <div className={classNames.parameterList}>
                        <Text variant="medium" className={classNames.boldText}>{localization.WhatIfTab.featureValues}</Text>
                        <SearchBox
                            placeholder={localization.WhatIf.filterFeaturePlaceholder}
                            onChange={this.filterFeatures}
                        />
                        <div className={classNames.featureList}>
                            {this.state.filteredFeatureList.map(item => {
                                const metaInfo = this.props.jointDataset.metaDict[item.key];
                                if (metaInfo.isCategorical) {
                                    const options: IDropdownOption[] = metaInfo.sortedCategoricalValues.map((text, key) => {
                                        return {key, text};
                                    })
                                    return <Dropdown 
                                        label={metaInfo.abbridgedLabel}
                                        selectedKey={this.temporaryPoint[item.key]}
                                        options={options}
                                        onChange={this.setCustomRowPropertyDropdown.bind(this, item.key)}
                                    />
                                }
                                return <TextField
                                    label={metaInfo.abbridgedLabel}
                                    value={this.temporaryPoint[item.key].toString()}
                                    onChange={this.setCustomRowProperty.bind(this, item.key, this.props.jointDataset.metaDict[item.key].treatAsCategorical)}
                                    styles={{ fieldGroup: { width: 100 } }}
                                />
                            })}
                        </div>
                    </div>
                    {this.state.editingDataCustomIndex !== undefined && (
                        <PrimaryButton
                            className={classNames.saveButton}
                            disabled={this.temporaryPoint[JointDataset.PredictedYLabel] === undefined}
                            text={localization.WhatIfTab.saveChanges}
                            onClick={this.savePoint}
                        />
                    )}
                    <PrimaryButton
                        className={classNames.saveButton}
                        disabled={this.temporaryPoint[JointDataset.PredictedYLabel] === undefined}
                        text={localization.WhatIfTab.saveAsNewPoint}
                        onClick={this.saveAsPoint}
                    />
                </div>)}
                {!this.state.isPanelOpen && (<IconButton
                    iconProps={{ iconName: "ChevronLeft" }}
                    onClick={this.openPanel}
                />)}
            </div>
            <div className={classNames.chartsArea}>
                {cohortOptions && (<div className={classNames.cohortPickerWrapper}>
                    <Text variant="mediumPlus" className={classNames.cohortPickerLabel}>{localization.WhatIfTab.cohortPickerLabel}</Text>
                    <Dropdown
                        styles={{ dropdown: { width: 150 } }}
                        options={cohortOptions}
                        selectedKey={this.state.selectedCohortIndex}
                        onChange={this.setSelectedCohort}
                    />
                </div>)}
                <div className={classNames.topArea}>
                    <div className={classNames.chartWithAxes}>
                        <div className={classNames.chartWithVertical}>
                            <div className={classNames.verticalAxis}>
                                <div className={classNames.rotatedVerticalBox}>
                                    <Text block variant="mediumPlus" className={classNames.boldText}>{localization.Charts.yValue}</Text>
                                    <DefaultButton
                                        onClick={this.setYOpen.bind(this, true)}
                                        id={this._yButtonId}
                                        text={this.props.jointDataset.metaDict[this.props.chartProps.yAxis.property].abbridgedLabel}
                                        title={this.props.jointDataset.metaDict[this.props.chartProps.yAxis.property].label}
                                    />
                                    {(this.state.yDialogOpen) && (
                                        <AxisConfigDialog
                                            jointDataset={this.props.jointDataset}
                                            orderedGroupTitles={[ColumnCategories.index, ColumnCategories.dataset, ColumnCategories.outcome]}
                                            selectedColumn={this.props.chartProps.yAxis}
                                            canBin={false}
                                            mustBin={false}
                                            canDither={this.props.chartProps.chartType === ChartTypes.Scatter}
                                            onAccept={this.onYSet}
                                            onCancel={this.setYOpen.bind(this, false)}
                                            target={this._yButtonId}
                                        />
                                    )}
                                </div>
                            </div>
                            <AccessibleChart
                                plotlyProps={plotlyProps}
                                theme={undefined}
                                onClickHandler={this.selectPointFromChart}
                            />
                        </div>
                        <div className={classNames.horizontalAxisWithPadding}>
                            <div className={classNames.paddingDiv}></div>
                            <div className={classNames.horizontalAxis}>
                                <div>
                                    <Text block variant="mediumPlus" className={classNames.boldText}>{localization.Charts.xValue}</Text>
                                    <DefaultButton
                                        onClick={this.setXOpen.bind(this, true)}
                                        id={this._xButtonId}
                                        text={this.props.jointDataset.metaDict[this.props.chartProps.xAxis.property].abbridgedLabel}
                                        title={this.props.jointDataset.metaDict[this.props.chartProps.xAxis.property].label}
                                    />
                                </div>
                                {(this.state.xDialogOpen) && (
                                    <AxisConfigDialog
                                        jointDataset={this.props.jointDataset}
                                        orderedGroupTitles={[ColumnCategories.index, ColumnCategories.dataset, ColumnCategories.outcome]}
                                        selectedColumn={this.props.chartProps.xAxis}
                                        canBin={this.props.chartProps.chartType === ChartTypes.Bar || this.props.chartProps.chartType === ChartTypes.Box}
                                        mustBin={this.props.chartProps.chartType === ChartTypes.Bar || this.props.chartProps.chartType === ChartTypes.Box}
                                        canDither={this.props.chartProps.chartType === ChartTypes.Scatter}
                                        onAccept={this.onXSet}
                                        onCancel={this.setXOpen.bind(this, false)}
                                        target={this._xButtonId}
                                    />
                                )}
                            </div>
                        </div>
                    </div >
                    <div className={classNames.legendAndText}>
                        <div className={classNames.legendHlepWrapper}>
                            <Text variant={"small"} className={classNames.legendHelpText}>{localization.WhatIfTab.scatterLegendText}</Text>
                        </div>
                        <Text variant={"small"} block className={classNames.legendLabel}>{localization.WhatIfTab.realPoint}</Text>
                        {this.selectedFeatureImportance.length > 0 &&
                        <InteractiveLegend
                            items={this.selectedFeatureImportance.map((row, rowIndex) => {
                                return {
                                    name: row.name,
                                    color: FabricStyles.fabricColorPalette[rowIndex],
                                    activated: this.state.pointIsActive[rowIndex],
                                    onClick: this.toggleActivation.bind(this, rowIndex),
                                    onDelete: this.toggleSelectionOfPoint.bind(this, row.id)
                                }
                            })}
                        />}
                        {this.selectedFeatureImportance.length === 0 && 
                        <Text variant={"xSmall"} className={classNames.smallItalic}>{localization.WhatIfTab.noneSelectedYet}</Text>}
                        <Text variant={"small"} block className={classNames.legendLabel}>{localization.WhatIfTab.whatIfDatapoints}</Text>
                        {this.state.customPoints.length > 0 &&
                        <InteractiveLegend
                            items={this.state.customPoints.map((row, rowIndex) => {
                                return {
                                    name: row[WhatIfTab.namePath],
                                    color: FabricStyles.fabricColorPalette[rowIndex + WhatIfTab.MAX_SELECTION + 1],
                                    activated: this.state.customPointIsActive[rowIndex],
                                    onClick: this.toggleCustomActivation.bind(this, rowIndex),
                                    onDelete: this.removeCustomPoint.bind(this, rowIndex),
                                    onEdit: this.setTemporaryPointToCustomPoint.bind(this, rowIndex)
                                }
                            })}
                        />}
                        {this.state.customPoints.length === 0 && 
                        <Text variant={"xSmall"} className={classNames.smallItalic}>{localization.WhatIfTab.noneCreatedYet}</Text>}

                    </div>
                </div>
                {this.buildSecondaryArea(classNames)}
            </div>
            </div>
        </div>);
    }

    private buildSecondaryArea(classNames: IProcessedStyleSet<IWhatIfTabStyles>): React.ReactNode {
        let secondaryPlot: React.ReactNode;
        if (this.state.secondaryChartChoice === WhatIfTab.featureImportanceKey) {
            if (this.includedFeatureImportance.length === 0){
                secondaryPlot = <div className={classNames.secondaryChartPlacolderBox}>
                    <div className={classNames.secondaryChartPlacolderSpacer}>
                        <Text variant="large" className={classNames.faintText}>{localization.WhatIfTab.featureImportanceGetStartedText}</Text>
                    </div>
                </div>
            } else {
                const maxStartingK = Math.max(0, this.props.jointDataset.localExplanationFeatureCount - this.state.topK);
                secondaryPlot = (<div className={classNames.featureImportanceArea}>
                    <div className={classNames.featureImportanceControls}>
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
                    <div className={classNames.featureImportanceChartAndLegend}>
                        <FeatureImportanceBar
                            jointDataset={this.props.jointDataset}
                            sortArray={this.state.sortArray}
                            startingK={this.state.startingK}
                            unsortedX={this.props.metadata.featureNamesAbridged}
                            unsortedSeries={this.includedFeatureImportance}
                            topK={this.state.topK}
                        />
                        <div className={classNames.featureImportanceLegend}> </div>
                    </div>
                </div>);
            }
        } else {
            if (this.testableDatapoints.length === 0){
                secondaryPlot = <div className={classNames.secondaryChartPlacolderBox}>
                    <div className={classNames.secondaryChartPlacolderSpacer}>
                        <Text variant="large" className={classNames.faintText}>{localization.WhatIfTab.IceGetStartedText}</Text>
                    </div>
            </div>;
            } else { 
                secondaryPlot = (<div className={classNames.featureImportanceArea}>
                    <div className={classNames.featureImportanceChartAndLegend}>
                        <MultiICEPlot 
                            invokeModel={this.props.invokeModel}
                            datapoints={this.testableDatapoints}
                            colors={this.testableDatapointColors}
                            jointDataset={this.props.jointDataset}
                            metadata={this.props.metadata}
                            theme={this.props.theme}
                        />
                        <div className={classNames.featureImportanceLegend}> </div>
                    </div>
                </div>);
            }
            
        }

        return( <div>
            <div className={classNames.choiceBoxArea}>
                <Text variant="medium" className={classNames.boldText}>{localization.WhatIfTab.showLabel}</Text>
                <ChoiceGroup
                    className={classNames.choiceGroup}
                    styles={{
                        flexContainer: classNames.choiceGroupFlexContainer
                    }}
                    options={WhatIfTab.secondaryPlotChoices}
                    selectedKey={this.state.secondaryChartChoice}
                    onChange={this.setSecondaryChart}/>
            </div>
            {secondaryPlot}
        </div>)
    }

    private buildExistingPredictionLabels(classNames: IProcessedStyleSet<IWhatIfTabStyles>): React.ReactNode {
        if (this.props.metadata.modelType === ModelTypes.binary) {
            const row = this.props.jointDataset.getRow(this.state.selectedWhatIfRootIndex);
            const predictedClass = this.props.jointDataset.hasPredictedY ?
                row[JointDataset.PredictedYLabel] : undefined;
            const predictedClassName = predictedClass !== undefined ?
                this.props.jointDataset.metaDict[JointDataset.PredictedYLabel].sortedCategoricalValues[predictedClass] :
                undefined;
            const predictedProb = this.props.jointDataset.hasPredictedProbabilities ?
                row[JointDataset.ProbabilityYRoot + predictedClass.toString()] :
                undefined;
            return (<div className={classNames.predictedBlock}>
                    {predictedClass !== undefined &&
                    (<Text block variant="small">{localization.formatString(localization.WhatIfTab.predictedClass, predictedClassName)}</Text>)}
                    {predictedProb !== undefined &&
                    (<Text block variant="small">{localization.formatString(localization.WhatIfTab.probability, predictedProb.toPrecision(5))}</Text>)}
                </div>);
        } else {
            return <div></div>
        }
    }

    private buildCustomPredictionLabels(classNames: IProcessedStyleSet<IWhatIfTabStyles>): React.ReactNode {
        if (this.props.metadata.modelType === ModelTypes.binary) {
            const predictedClass = this.props.jointDataset.hasPredictedY ?
                this.temporaryPoint[JointDataset.PredictedYLabel] : undefined;
            const predictedClassName = predictedClass !== undefined ?
                this.props.jointDataset.metaDict[JointDataset.PredictedYLabel].sortedCategoricalValues[predictedClass] :
                undefined;
            const predictedProb = this.props.jointDataset.hasPredictedProbabilities && predictedClass !== undefined ?
                this.temporaryPoint[JointDataset.ProbabilityYRoot + predictedClass.toString()] :
                undefined;
            return (<div className={classNames.customPredictBlock}>
                    {predictedClass !== undefined &&
                    (<Text block variant="small" className={classNames.boldText}>{localization.formatString(localization.WhatIfTab.newPredictedClass, predictedClassName)}</Text>)}
                    {predictedProb !== undefined &&
                    (<Text block variant="small" className={classNames.boldText}>{localization.formatString(localization.WhatIfTab.newProbability, predictedProb.toPrecision(5))}</Text>)}
                </div>);
        } else {
            return <div></div>
        }
    }

    private setStartingK(newValue: number): void {
        this.setState({ startingK: newValue });
    }

    private getDefaultSelectedPointIndexes(cohort: Cohort): number[] {
        const indexes = cohort.unwrap(JointDataset.IndexLabel);
        if (indexes.length > 0) {
            return [indexes[0]];
        }
        return [];
    }

    private setSelectedCohort(event: React.FormEvent<HTMLDivElement>, item: IDropdownOption): void {
        this.setState({ selectedCohortIndex: item.key as number, selectedPointsIndexes: [] });
    }

    private setSortIndex(event: React.FormEvent<HTMLDivElement>, item: IDropdownOption): void {
        const newIndex = item.key as number;
        const sortArray = ModelExplanationUtils.getSortIndices(this.selectedFeatureImportance[newIndex].unsortedAggregateY).reverse()
        this.setState({ sortingSeriesIndex: newIndex, sortArray });
    }

    private setSecondaryChart(event: React.SyntheticEvent<HTMLElement>, item: IChoiceGroupOption): void {
        this.setState({secondaryChartChoice: item.key});
    }

    private setSelectedIndex(event: React.FormEvent<HTMLDivElement>, item: IDropdownOption): void {
        this.setTemporaryPointToCopyOfDatasetPoint(item.key as number);
    }

    private setTemporaryPointToCopyOfDatasetPoint(index: number): void {
        this.temporaryPoint = this.props.jointDataset.getRow(index);
        this.temporaryPoint[WhatIfTab.namePath] = localization.formatString(localization.WhatIf.defaultCustomRootName, index) as string;
        this.temporaryPoint[WhatIfTab.colorPath] = FabricStyles.fabricColorPalette[WhatIfTab.MAX_SELECTION + this.state.customPoints.length];

        this.setState({
            selectedWhatIfRootIndex: index,
            editingDataCustomIndex: undefined
        });
    }

    private setTemporaryPointToCustomPoint(index: number): void {
        this.temporaryPoint = _.cloneDeep(this.state.customPoints[index]);
        this.setState({
            selectedWhatIfRootIndex: this.temporaryPoint[JointDataset.IndexLabel],
            editingDataCustomIndex: index
        });
        this.openPanel();
    }

    private removeCustomPoint(index: number): void {
        this.setState(prevState => {
            const customPoints = [...prevState.customPoints];
            customPoints.splice(index, 1);
            return { customPoints };
        });
    }

    private setCustomRowProperty(key: string, isString: boolean, event: React.FormEvent<HTMLInputElement | HTMLTextAreaElement>, newValue?: string): void {
        const editingData = this.temporaryPoint;
        if (isString) {
            editingData[key] = newValue;
        } else {
            const asNumber = +newValue;
            if (!Number.isFinite(asNumber)) {
                alert('thats no number')
            }
            editingData[key] = asNumber;
        }
        this.forceUpdate();
        this.fetchData(editingData);
    }

    private setCustomRowPropertyDropdown(key: string, event: React.FormEvent<HTMLDivElement>, item: IDropdownOption): void {
        const editingData = this.temporaryPoint;
        editingData[key] = item.key;
        this.forceUpdate();
        this.fetchData(editingData);
    }

    private savePoint(): void {
        const customPoints = [...this.state.customPoints];
        customPoints[this.state.editingDataCustomIndex] = this.temporaryPoint;
        this.temporaryPoint = _.cloneDeep(this.temporaryPoint);
        this.setState({ customPoints });
    }

    private saveAsPoint(): void {
        const editingDataCustomIndex = this.state.editingDataCustomIndex !== undefined ?
            this.state.editingDataCustomIndex : this.state.customPoints.length;
        const customPoints = [...this.state.customPoints];
        const customPointIsActive = [...this.state.customPointIsActive];
        customPoints.push(this.temporaryPoint);
        customPointIsActive.push(true);
        this.temporaryPoint = _.cloneDeep(this.temporaryPoint);
        this.setState({ editingDataCustomIndex, customPoints, customPointIsActive});
    }

    private createCopyOfFirstRow(): { [key: string]: any } {
        const indexes = this.getDefaultSelectedPointIndexes(this.props.cohorts[this.state.selectedCohortIndex]);
        if (indexes.length === 0) {
            return undefined;
        }
        const customData = this.props.jointDataset.getRow(indexes[0]) as any;
        customData[WhatIfTab.namePath] = localization.formatString(localization.WhatIf.defaultCustomRootName, indexes[0]) as string;
        customData[WhatIfTab.colorPath] = FabricStyles.fabricColorPalette[WhatIfTab.MAX_SELECTION + this.state.customPoints.length];
        return customData;
    }

    private toggleActivation(index: number): void {
        const pointIsActive = [...this.state.pointIsActive];
        pointIsActive[index] = !pointIsActive[index];
        this.setState({ pointIsActive });
    }

    private toggleCustomActivation(index: number): void {
        const customPointIsActive = [...this.state.customPointIsActive];
        customPointIsActive[index] = !customPointIsActive[index];
        this.setState({ customPointIsActive });
    }

    private dismissPanel(): void {
        this.setState({ isPanelOpen: false });
        window.dispatchEvent(new Event('resize'));
    }

    private openPanel(): void {
        this.setState({ isPanelOpen: true });
        window.dispatchEvent(new Event('resize'));
    }

    private onXSet(value: ISelectorConfig): void {
        const newProps = _.cloneDeep(this.props.chartProps);
        newProps.xAxis = value;
        this.props.onChange(newProps);
        this.setState({ xDialogOpen: false })
    }

    private onYSet(value: ISelectorConfig): void {
        const newProps = _.cloneDeep(this.props.chartProps);
        newProps.yAxis = value;
        this.props.onChange(newProps);
        this.setState({ yDialogOpen: false })
    }

    private filterFeatures(event?: React.ChangeEvent<HTMLInputElement>, newValue?: string): void {
        if (newValue === undefined || newValue === null || !/\S/.test(newValue)) {
            this.setState({ filteredFeatureList: this.featureList });
        }
        const filteredFeatureList = this.featureList.filter(item => {
            return item.label.includes(newValue.toLowerCase());
        });
        this.setState({ filteredFeatureList });
    }

    private readonly setXOpen = (val: boolean): void => {
        if (val && this.state.xDialogOpen === false) {
            this.setState({ xDialogOpen: true });
            return;
        }
        this.setState({ xDialogOpen: false });
    }

    private readonly setYOpen = (val: boolean): void => {
        if (val && this.state.yDialogOpen === false) {
            this.setState({ yDialogOpen: true });
            return;
        }
        this.setState({ yDialogOpen: false });
    }

    private selectPointFromChart(data: any): void {
        const trace = data.points[0];
        // custom point
        if (trace.curveNumber === 1) {
            this.setTemporaryPointToCustomPoint(trace.pointNumber);
        } else {
            const index = trace.customdata[JointDataset.IndexLabel];
            this.setTemporaryPointToCopyOfDatasetPoint(index);
            this.toggleSelectionOfPoint(index);
        }
    }

    private toggleSelectionOfPoint(index: number): void {
        const indexOf = this.state.selectedPointsIndexes.indexOf(index);
        let newSelections = [...this.state.selectedPointsIndexes];
        let pointIsActive = [...this.state.pointIsActive];
        if (indexOf === -1) {
            if (this.state.selectedPointsIndexes.length > WhatIfTab.MAX_SELECTION) {
                return;
            }
            // const startingIdex = this.state.selectedPointsIndexes.length > WhatIfTab.MAX_SELECTION ? 1 : 0;
            // newSelections = this.state.selectedPointsIndexes.slice(startingIdex);
            newSelections.push(index);
            pointIsActive.push(true);
        } else {
            newSelections.splice(indexOf, 1);
            pointIsActive.splice(indexOf, 1);
        }
        this.setState({ selectedPointsIndexes: newSelections, pointIsActive });
    }

    // fetch prediction for temporary point
    private fetchData(fetchingReference: { [key: string]: any }): void {
        if (this.state.request !== undefined) {
            this.state.request.abort();
        }
        const abortController = new AbortController();
        const rawData = JointDataset.datasetSlice(fetchingReference, this.props.jointDataset.metaDict, this.props.jointDataset.datasetFeatureCount);
        fetchingReference[JointDataset.PredictedYLabel] = undefined;
        const promise = this.props.invokeModel([rawData], abortController.signal);


        this.setState({ request: abortController }, async () => {
            try {
                const fetchedData = await promise;
                // returns predicted probabilities
                if (Array.isArray(fetchedData[0])) {
                    const predictionVector = fetchedData[0];
                    let predictedClass = 0;
                    let maxProb = Number.MIN_SAFE_INTEGER;
                    for (let i = 0; i < predictionVector.length; i++) {
                        fetchingReference[JointDataset.ProbabilityYRoot + i.toString()] = predictionVector[i];
                        if (predictionVector[i] > maxProb) {
                            predictedClass = i;
                            maxProb = predictionVector[i];
                        }
                    }
                    fetchingReference[JointDataset.PredictedYLabel] = predictedClass;
                    this.setState({request: undefined});
                } else {
                    // prediction is a scalar, no probabilities
                    fetchingReference[JointDataset.PredictedYLabel] = fetchedData[0]
                }
            } catch (err) {
                if (err.name === 'AbortError') {
                    return;
                }
                if (err.name === 'PythonError') {
                    alert("error");
                }
            }
        });
    }

    private generatePlotlyProps(jointData: JointDataset, chartProps: IGenericChartProps, cohort: Cohort): IPlotlyProperty {
        const plotlyProps = _.cloneDeep(WhatIfTab.basePlotlyProperties);
        plotlyProps.data[0].hoverinfo = "all";
        const indexes = cohort.unwrap(JointDataset.IndexLabel);
        plotlyProps.data[0].type = chartProps.chartType;
        plotlyProps.data[0].mode = PlotlyMode.markers;
        plotlyProps.data[0].marker = {
            symbol: indexes.map(i => this.state.selectedPointsIndexes.includes(i) ? "square" : "circle") as any,
            color: indexes.map((rowIndex) => {
                const selectionIndex = this.state.selectedPointsIndexes.indexOf(rowIndex);
                if (selectionIndex === -1) {
                    return FabricStyles.fabricColorInactiveSeries;
                }
                return FabricStyles.fabricColorPalette[selectionIndex];
            }) as any,
            size: 8
        }

        plotlyProps.data[1] = {
            type: "scatter",
            mode: PlotlyMode.markers,
            marker: {
                symbol: "star",
                size: 12,
                color: this.state.customPoints.map((unused, i) => FabricStyles.fabricColorPalette[WhatIfTab.MAX_SELECTION + 1 + i])
            }
        }

        plotlyProps.data[2] = {
            type: "scatter",
            mode: PlotlyMode.markers,
            text: "Editable What-If point",
            hoverinfo: "text",
            marker: {
                opacity: 0.5,
                symbol: "star",
                size: 12,
                color: 'rgba(0,0,0,0)',
                line: {
                    color: FabricStyles.fabricColorPalette[WhatIfTab.MAX_SELECTION + 1 + this.state.customPoints.length],
                    width: 2
                }
            }
        }

        if (chartProps.xAxis) {
            if (jointData.metaDict[chartProps.xAxis.property].isCategorical) {
                const xLabels = jointData.metaDict[chartProps.xAxis.property].sortedCategoricalValues;
                const xLabelIndexes = xLabels.map((unused, index) => index);
                _.set(plotlyProps, "layout.xaxis.ticktext", xLabels);
                _.set(plotlyProps, "layout.xaxis.tickvals", xLabelIndexes);
            }
            const rawX = cohort.unwrap(chartProps.xAxis.property);
            const customX = JointDataset.unwrap(this.state.customPoints, chartProps.xAxis.property);
            const tempX = JointDataset.unwrap([this.temporaryPoint], chartProps.xAxis.property);
            if (chartProps.xAxis.options.dither) {
                const dithered = cohort.unwrap(JointDataset.DitherLabel);
                const customDithered = JointDataset.unwrap(this.state.customPoints, JointDataset.DitherLabel);
                const tempDithered = JointDataset.unwrap([this.temporaryPoint], JointDataset.DitherLabel);
                plotlyProps.data[0].x = dithered.map((dither, index) => { return rawX[index] + dither; });
                plotlyProps.data[1].x = customDithered.map((dither, index) => { return customX[index] + dither; });
                plotlyProps.data[2].x = tempDithered.map((dither, index) => { return tempX[index] + dither; })
            } else {
                plotlyProps.data[0].x = rawX;
                plotlyProps.data[1].x = customX;
                plotlyProps.data[2].x = tempX;
            }
        }
        if (chartProps.yAxis) {
            if (jointData.metaDict[chartProps.yAxis.property].isCategorical) {
                const yLabels = jointData.metaDict[chartProps.yAxis.property].sortedCategoricalValues;
                const yLabelIndexes = yLabels.map((unused, index) => index);
                _.set(plotlyProps, "layout.yaxis.ticktext", yLabels);
                _.set(plotlyProps, "layout.yaxis.tickvals", yLabelIndexes);
            }
            const rawY = cohort.unwrap(chartProps.yAxis.property);
            const customY = JointDataset.unwrap(this.state.customPoints, chartProps.yAxis.property);
            const tempY = JointDataset.unwrap([this.temporaryPoint], chartProps.yAxis.property);
            if (chartProps.yAxis.options.dither) {
                const dithered = cohort.unwrap(JointDataset.DitherLabel);
                const customDithered = JointDataset.unwrap(this.state.customPoints, JointDataset.DitherLabel);
                const tempDithered = JointDataset.unwrap([this.temporaryPoint], JointDataset.DitherLabel);
                plotlyProps.data[0].y = dithered.map((dither, index) => { return rawY[index] + dither; });
                plotlyProps.data[1].y = customDithered.map((dither, index) => { return customY[index] + dither; });
                plotlyProps.data[2].y = tempDithered.map((dither, index) => { return tempY[index] + dither; })
            } else {
                plotlyProps.data[0].y = rawY;
                plotlyProps.data[1].y = customY;
                plotlyProps.data[2].y = tempY;
            }
        }


        plotlyProps.data[0].customdata = WhatIfTab.buildCustomData(jointData, chartProps, cohort);
        plotlyProps.data[0].hovertemplate = WhatIfTab.buildHoverTemplate(chartProps);
        return plotlyProps;
    }

    private static buildHoverTemplate(chartProps: IGenericChartProps): string {
        let hovertemplate = "";
        if (chartProps.xAxis) {
            if (chartProps.xAxis.options.dither) {
                hovertemplate += "x: %{customdata.X}<br>";
            } else {
                hovertemplate += "x: %{x}<br>";
            }
        }
        if (chartProps.yAxis) {
            if (chartProps.yAxis.options.dither) {
                hovertemplate += "y: %{customdata.Y}<br>";
            } else {
                hovertemplate += "y: %{y}<br>";
            }
        }
        hovertemplate += "<extra></extra>";
        return hovertemplate;
    }

    private static buildCustomData(jointData: JointDataset, chartProps: IGenericChartProps, cohort: Cohort): Array<any> {
        const customdata = cohort.unwrap(JointDataset.IndexLabel).map(val => {
            const dict = {};
            dict[JointDataset.IndexLabel] = val;
            return dict;
        });
        if (chartProps.chartType === ChartTypes.Scatter) {
            const xAxis = chartProps.xAxis;
            if (xAxis && xAxis.property && xAxis.options.dither) {
                const rawX = cohort.unwrap(chartProps.xAxis.property);
                rawX.forEach((val, index) => {
                    // If categorical, show string value in tooltip
                    if (jointData.metaDict[chartProps.xAxis.property].isCategorical) {
                        customdata[index]["X"] = jointData.metaDict[chartProps.xAxis.property]
                            .sortedCategoricalValues[val];
                    } else {
                        customdata[index]["X"] = val;
                    }
                });
            }
            const yAxis = chartProps.yAxis;
            if (yAxis && yAxis.property && yAxis.options.dither) {
                const rawY = cohort.unwrap(chartProps.yAxis.property);
                rawY.forEach((val, index) => {
                    // If categorical, show string value in tooltip
                    if (jointData.metaDict[chartProps.yAxis.property].isCategorical) {
                        customdata[index]["Y"] = jointData.metaDict[chartProps.yAxis.property]
                            .sortedCategoricalValues[val];
                    } else {
                        customdata[index]["Y"] = val;
                    }
                });
            }
        }
        return customdata;
    }

    private generateDefaultChartAxes(): void {
        const yKey = JointDataset.DataLabelRoot + "0";
        const yIsDithered = this.props.jointDataset.metaDict[yKey].isCategorical;
        const chartProps: IGenericChartProps = {
            chartType: ChartTypes.Scatter,
            xAxis: {
                property: JointDataset.IndexLabel,
                options: {}
            },
            yAxis: {
                property: yKey,
                options: {
                    dither: yIsDithered,
                    bin: false
                }
            },
            colorAxis: {
                property: this.props.jointDataset.hasPredictedY ?
                    JointDataset.PredictedYLabel : JointDataset.IndexLabel,
                options: {}
            }
        }
        this.props.onChange(chartProps);
    }
}