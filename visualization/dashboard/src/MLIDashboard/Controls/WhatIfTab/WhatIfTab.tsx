import { IProcessedStyleSet, getTheme } from "@uifabric/styling";
import _ from "lodash";
import { AccessibleChart, IPlotlyProperty, PlotlyMode, IData } from "mlchartlib";
import { ChoiceGroup, IChoiceGroupOption, Icon, Slider, Text, ComboBox, IComboBox  } from "office-ui-fabric-react";
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
import { MultiICEPlot } from "../MultiICEPlot/MultiICEPlot";
import { IGlobalSeries } from "../GlobalExplanationTab/IGlobalSeries";
import { ModelExplanationUtils } from "../../ModelExplanationUtils";
import { ChartTypes, IGenericChartProps, ISelectorConfig, NewExplanationDashboard } from "../../NewExplanationDashboard";
import { AxisConfigDialog } from "../AxisConfigurationDialog/AxisConfigDialog";
import { FeatureImportanceBar } from "../FeatureImportanceBar/FeatureImportanceBar";
import { InteractiveLegend } from "../InteractiveLegend";
import { IWhatIfTabStyles, whatIfTabStyles } from "./WhatIfTab.styles";

export interface IWhatIfTabProps {
    jointDataset: JointDataset;
    metadata: IExplanationModelMetadata;
    cohorts: Cohort[];
    chartProps: IGenericChartProps;
    onChange: (config: IGenericChartProps) => void;
    invokeModel: (data: any[], abortSignal: AbortSignal) => Promise<any[]>;
    editCohort: (index: number) => void;
}

export interface IWhatIfTabState {
    isPanelOpen: boolean;
    xDialogOpen: boolean;
    yDialogOpen: boolean;
    selectedWhatIfRootIndex: number;
    editingDataCustomIndex?: number;
    showSelectionWarning: boolean;
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
    selectedFeatureKey: string;
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
    private testableDatapointNames: string[] = [];
    private featuresOption: IDropdownOption[] = new Array(this.props.jointDataset.datasetFeatureCount).fill(0)
        .map((unused, index) => {
        const key = JointDataset.DataLabelRoot + index.toString();
        return {key, text: this.props.jointDataset.metaDict[key].abbridgedLabel};
    });

    constructor(props: IWhatIfTabProps) {
        super(props);
        
        if (!this.props.jointDataset.hasDataset) {
            return;
        }
        this.state = {
            isPanelOpen: this.props.invokeModel !== undefined,
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
            secondaryChartChoice: WhatIfTab.featureImportanceKey,
            selectedFeatureKey: JointDataset.DataLabelRoot + "0",
            showSelectionWarning: false
        };

        if (props.chartProps === undefined) {
            this.generateDefaultChartAxes();
        }
        this.temporaryPoint = this.createCopyOfFirstRow();
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
        this.onFeatureSelected = this.onFeatureSelected.bind(this);
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
            const includedNames = this.includedFeatureImportance.map(item => item.name);
            const includedRows = this.state.pointIsActive.map((isActive, i) => {
                if (isActive) {
                    return this.selectedDatapoints[i];
                }
            }).filter(item => !!item);
            const includedCustomRows = this.state.customPointIsActive.map((isActive, i) => {
                if (isActive) {
                    includedColors.push(FabricStyles.fabricColorPalette[WhatIfTab.MAX_SELECTION + i + 1])
                    includedColors.push(FabricStyles.fabricColorPalette[WhatIfTab.MAX_SELECTION + i + 1]);
                    includedNames.push(this.state.customPoints[i][WhatIfTab.namePath]);
                    return this.customDatapoints[i];
                }
            }).filter(item => !!item);
            this.testableDatapoints = [...includedRows, ...includedCustomRows];
            this.testableDatapointColors = includedColors;
            this.testableDatapointNames = includedNames;
            this.forceUpdate();
        }
        this.setState({ sortingSeriesIndex, sortArray })
    }

    public render(): React.ReactNode {
        const classNames = whatIfTabStyles();
        if (!this.props.jointDataset.hasDataset) {
            return (
                <div className={classNames.missingParametersPlaceholder}>
                    <div className={classNames.missingParametersPlaceholderSpacer}>
                        <Text variant="large" className={classNames.faintText}>{localization.WhatIfTab.missingParameters}</Text>
                    </div>
                </div>
            );
        }
        if (this.props.chartProps === undefined) {
            return (<div />);
        }
        const plotlyProps = this.generatePlotlyProps(
            this.props.jointDataset,
            this.props.chartProps,
            this.props.cohorts[this.state.selectedCohortIndex]
        );
        const cohortLength = this.props.cohorts[this.state.selectedCohortIndex].rowCount;
        const canRenderChart = cohortLength < NewExplanationDashboard.ROW_ERROR_SIZE || this.props.chartProps.chartType !== ChartTypes.Scatter;
        const rowOptions: IDropdownOption[ ]= this.props.cohorts[this.state.selectedCohortIndex].unwrap(JointDataset.IndexLabel).map(index => {
            return {key: index, text: localization.formatString(localization.WhatIfTab.rowLabel, index.toString()) as string};
        });
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
                    <div>
                        <div className={classNames.panelIconAndLabel}>
                        <IconButton
                            iconProps={{ iconName: "ChevronRight" }}
                            onClick={this.dismissPanel}
                            className={classNames.blackIcon}
                        />
                        <Text variant={"medium"} className={classNames.boldText}>{localization.WhatIfTab.whatIfDatapoint}</Text>
                        </div>
                        <div className={classNames.panelPlaceholderWrapper}>
                            <div className={classNames.missingParametersPlaceholderSpacer}>
                                <Text>{localization.WhatIfTab.panelPlaceholder}</Text>
                            </div>
                        </div>
                    </div>
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
                        {this.buildCustomPredictionLabels(classNames)}
                    </div>
                    <div className={classNames.parameterList}>
                        <Text variant="medium" className={classNames.boldText}>{localization.WhatIfTab.featureValues}</Text>
                        <SearchBox
                            className={classNames.featureSearch}
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
                    <div className={classNames.disclaimerWrapper}>
                        <Text variant={"xSmall"}>{localization.WhatIfTab.disclaimer}</Text>
                    </div>
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
                            {!canRenderChart && (
                                <div className={classNames.missingParametersPlaceholder}>
                                    <div className={classNames.missingParametersPlaceholderSpacer}>
                                        <Text block variant="large" className={classNames.faintText}>{localization.ValidationErrors.datasizeError}</Text>
                                        <PrimaryButton onClick={this.editCohort}>{localization.ValidationErrors.addFilters}</PrimaryButton>
                                    </div>
                                </div>
                            )}
                            {canRenderChart && (
                            <AccessibleChart
                                plotlyProps={plotlyProps}
                                theme={getTheme() as any}
                                onClickHandler={this.selectPointFromChart}
                            />)}
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
                                        canBin={this.props.chartProps.chartType === ChartTypes.Histogram || this.props.chartProps.chartType === ChartTypes.Box}
                                        mustBin={this.props.chartProps.chartType === ChartTypes.Histogram || this.props.chartProps.chartType === ChartTypes.Box}
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
                        {this.state.showSelectionWarning &&
                        <Text variant={"xSmall"} className={classNames.errorText}>{localization.WhatIfTab.selectionLimit}</Text>}
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

    private editCohort(): void {
        this.props.editCohort(this.state.selectedCohortIndex);
    }

    private buildSecondaryArea(classNames: IProcessedStyleSet<IWhatIfTabStyles>): React.ReactNode {
        let secondaryPlot: React.ReactNode;
        if (this.state.secondaryChartChoice === WhatIfTab.featureImportanceKey) {
            if (!this.props.jointDataset.hasLocalExplanations) {
                secondaryPlot = <div className={classNames.missingParametersPlaceholder}>
                    <div className={classNames.missingParametersPlaceholderSpacer}>
                        <Text variant="large" className={classNames.faintText}>{localization.WhatIfTab.featureImportanceLackingParameters}</Text>
                    </div>
                </div>
            }
            else if (this.includedFeatureImportance.length === 0){
                secondaryPlot = <div className={classNames.missingParametersPlaceholder}>
                    <div className={classNames.missingParametersPlaceholderSpacer}>
                        <Text variant="large" className={classNames.faintText}>{localization.WhatIfTab.featureImportanceGetStartedText}</Text>
                    </div>
                </div>
            } else {
                const yAxisLabels: string[] = [localization.featureImportance];
                if (this.props.metadata.modelType !== ModelTypes.regression) {
                    yAxisLabels.push(localization.formatString(localization.WhatIfTab.classLabel, this.props.metadata.classNames[0]) as string);
                }
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
                            yAxisLabels={yAxisLabels}
                            chartType={ChartTypes.Bar}
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
            if (!this.props.invokeModel) {
                secondaryPlot = <div className={classNames.missingParametersPlaceholder}>
                    <div className={classNames.missingParametersPlaceholderSpacer}>
                        <Text variant="large" className={classNames.faintText}>{localization.WhatIfTab.iceLackingParameters}</Text>
                    </div>
                </div>
            }
            else if (this.testableDatapoints.length === 0){
                secondaryPlot = <div className={classNames.missingParametersPlaceholder}>
                    <div className={classNames.missingParametersPlaceholderSpacer}>
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
                            rowNames={this.testableDatapointNames}
                            jointDataset={this.props.jointDataset}
                            metadata={this.props.metadata}
                            feature={this.state.selectedFeatureKey}
                        />
                        <div className={classNames.featureImportanceLegend}>
                            <ComboBox
                                autoComplete={"on"}
                                className={classNames.iceFeatureSelection}
                                options={this.featuresOption}
                                onChange={this.onFeatureSelected}
                                label={localization.IcePlot.featurePickerLabel}
                                ariaLabel="feature picker"
                                selectedKey={this.state.selectedFeatureKey }
                                useComboBoxAsMenuWidth={true}
                            />
                        </div>
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
        if (this.props.metadata.modelType !== ModelTypes.regression) {
            const row = this.props.jointDataset.getRow(this.state.selectedWhatIfRootIndex);
            const trueClass = this.props.jointDataset.hasTrueY ?
                row[JointDataset.TrueYLabel] : undefined;
            const predictedClass = this.props.jointDataset.hasPredictedY ?
                row[JointDataset.PredictedYLabel] : undefined;
            const predictedClassName = predictedClass !== undefined ?
                this.props.jointDataset.metaDict[JointDataset.PredictedYLabel].sortedCategoricalValues[predictedClass] :
                undefined;
            const predictedProb = this.props.jointDataset.hasPredictedProbabilities ?
                row[JointDataset.ProbabilityYRoot + predictedClass.toString()] :
                undefined;
            return (<div className={classNames.predictedBlock}>
                    {trueClass !== undefined && 
                    (<Text block variant="small">{localization.formatString(localization.WhatIfTab.trueClass, 
                        this.props.jointDataset.metaDict[JointDataset.PredictedYLabel].sortedCategoricalValues[trueClass])}</Text>)}
                    {predictedClass !== undefined &&
                    (<Text block variant="small">{localization.formatString(localization.WhatIfTab.predictedClass, predictedClassName)}</Text>)}
                    {predictedProb !== undefined &&
                    (<Text block variant="small">{localization.formatString(localization.WhatIfTab.probability, predictedProb.toLocaleString(undefined, {maximumFractionDigits: 3}))}</Text>)}
                </div>);
        } else {
            const row = this.props.jointDataset.getRow(this.state.selectedWhatIfRootIndex);
            const trueValue = this.props.jointDataset.hasTrueY ?
                row[JointDataset.TrueYLabel] : undefined;
            const predictedValue = this.props.jointDataset.hasPredictedY ?
                row[JointDataset.PredictedYLabel] : undefined;
            return (<div className={classNames.predictedBlock}>
                {trueValue !== undefined && 
                    (<Text block variant="small">{localization.formatString(localization.WhatIfTab.trueValue, trueValue)}</Text>)}
                {predictedValue !== undefined &&
                    (<Text block variant="small">{localization.formatString(localization.WhatIfTab.predictedValue, predictedValue.toLocaleString(undefined, {maximumFractionDigits: 3}))}</Text>)}
            </div>);
    }
    }

    private buildCustomPredictionLabels(classNames: IProcessedStyleSet<IWhatIfTabStyles>): React.ReactNode {
        if (this.props.metadata.modelType !== ModelTypes.regression) {
            const predictedClass = this.props.jointDataset.hasPredictedY ?
                this.temporaryPoint[JointDataset.PredictedYLabel] : undefined;
            const predictedClassName = predictedClass !== undefined ?
                this.props.jointDataset.metaDict[JointDataset.PredictedYLabel].sortedCategoricalValues[predictedClass] :
                undefined;
            const predictedProb = this.props.jointDataset.hasPredictedProbabilities && predictedClass !== undefined ?
                this.temporaryPoint[JointDataset.ProbabilityYRoot + predictedClass.toString()] :
                undefined;
            return (<div className={classNames.customPredictBlock}>
                    {this.props.jointDataset.hasPredictedY && predictedClass !== undefined &&
                    (<Text block variant="small" className={classNames.boldText}>{localization.formatString(localization.WhatIfTab.newPredictedClass, predictedClassName)}</Text>)}
                    {this.props.jointDataset.hasPredictedY && predictedClass === undefined &&
                    (<Text block variant="small" className={classNames.boldText}>{localization.formatString(localization.WhatIfTab.newPredictedClass, localization.WhatIfTab.loading)}</Text>)}
                    {this.props.jointDataset.hasPredictedProbabilities && predictedProb !== undefined &&
                    (<Text block variant="small" className={classNames.boldText}>{localization.formatString(localization.WhatIfTab.newProbability, predictedProb.toLocaleString(undefined, {maximumFractionDigits: 3}))}</Text>)}
                    {this.props.jointDataset.hasPredictedProbabilities && predictedProb === undefined &&
                    (<Text block variant="small" className={classNames.boldText}>{localization.formatString(localization.WhatIfTab.newProbability, localization.WhatIfTab.loading)}</Text>)}
                </div>);
        } else {
            const predictedValue = this.props.jointDataset.hasPredictedY ?
                this.temporaryPoint[JointDataset.PredictedYLabel] : undefined;
            return (<div className={classNames.customPredictBlock}>
                {this.props.jointDataset.hasPredictedY && predictedValue !== undefined &&
                (<Text block variant="small" className={classNames.boldText}>{localization.formatString(localization.WhatIfTab.newPredictedValue, predictedValue.toLocaleString(undefined, {maximumFractionDigits: 3}))}</Text>)}
                {this.props.jointDataset.hasPredictedY && predictedValue === undefined &&
                (<Text block variant="small" className={classNames.boldText}>{localization.formatString(localization.WhatIfTab.newPredictedValue, localization.WhatIfTab.loading)}</Text>)}
            </div>);
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
        this.setState({ selectedCohortIndex: item.key as number, selectedPointsIndexes: [], showSelectionWarning: false });
    }

    private onFeatureSelected(event: React.FormEvent<IComboBox>, item: IDropdownOption): void {
        this.setState({ selectedFeatureKey: item.key as string});
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
            const customPointIsActive = [...prevState.customPointIsActive];
            customPointIsActive.splice(index, 1);
            return { customPoints, customPointIsActive };
        });
    }

    private setCustomRowProperty(key: string, isString: boolean, event: React.FormEvent<HTMLInputElement | HTMLTextAreaElement>, newValue?: string): void {
        const editingData = this.temporaryPoint;
        if (isString) {
            editingData[key] = newValue;
        } else {
            const asNumber = +newValue;
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
                this.setState({showSelectionWarning: true});
                return;
            }
            newSelections.push(index);
            pointIsActive.push(true);
        } else {
            newSelections.splice(indexOf, 1);
            pointIsActive.splice(indexOf, 1);
        }
        this.setState({ selectedPointsIndexes: newSelections, pointIsActive, showSelectionWarning: false });
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
                } else {
                    // prediction is a scalar, no probabilities
                    fetchingReference[JointDataset.PredictedYLabel] = fetchedData[0];
                }
                if (this.props.jointDataset.hasTrueY) {
                    JointDataset.setErrorMetrics(fetchingReference, this.props.metadata.modelType);
                }
                this.setState({request: undefined});
            } catch (err) {
                if (err.name === 'AbortError') {
                    return;
                }
                if (err.name === 'PythonError') {
                    alert(localization.formatString(localization.IcePlot.errorPrefix, err.message) as string);
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
        }
        if (chartProps.yAxis) {
            if (jointData.metaDict[chartProps.yAxis.property].isCategorical) {
                const yLabels = jointData.metaDict[chartProps.yAxis.property].sortedCategoricalValues;
                const yLabelIndexes = yLabels.map((unused, index) => index);
                _.set(plotlyProps, "layout.yaxis.ticktext", yLabels);
                _.set(plotlyProps, "layout.yaxis.tickvals", yLabelIndexes);
            }
        }


        this.generateDataTrace(cohort.filteredData, chartProps, plotlyProps.data[0]);
        this.generateDataTrace(this.state.customPoints, chartProps, plotlyProps.data[1]);
        this.generateDataTrace([this.temporaryPoint], chartProps, plotlyProps.data[2]);
        return plotlyProps;
    }

    private generateDataTrace(dictionary: Array<{[key: string]: number}>, chartProps: IGenericChartProps, trace: IData): void {
        const customdata = JointDataset.unwrap(dictionary, JointDataset.IndexLabel).map(val => {
            const dict = {};
            dict[JointDataset.IndexLabel] = val;
            return dict;
        });
        let hovertemplate = "";
        if (chartProps.xAxis) {
            const metaX = this.props.jointDataset.metaDict[chartProps.xAxis.property];
            const rawX = JointDataset.unwrap(dictionary, chartProps.xAxis.property);
            if (metaX.isCategorical) {
                hovertemplate += metaX.abbridgedLabel + ": %{customdata.X}<br>";
                rawX.map((val, index) => {
                    customdata[index]["X"] = metaX.sortedCategoricalValues[val]
                });
                if (chartProps.xAxis.options.dither) {
                    const dither = JointDataset.unwrap(dictionary, JointDataset.DitherLabel);
                    trace.x = dither.map((ditherVal, index) => { return rawX[index] + ditherVal; });
                } else {
                    trace.x = rawX;
                }
            } else {
                hovertemplate += metaX.abbridgedLabel + ": %{x}<br>";
                trace.x = rawX;
            }
        }
        if (chartProps.yAxis) {
            const metaY = this.props.jointDataset.metaDict[chartProps.yAxis.property];
            const rawY = JointDataset.unwrap(dictionary, chartProps.yAxis.property);
            if (metaY.isCategorical) {
                hovertemplate += metaY.abbridgedLabel + ": %{customdata.Y}<br>";
                rawY.map((val, index) => {
                    customdata[index]["Y"] = metaY.sortedCategoricalValues[val]
                });
                if (chartProps.yAxis.options.dither) {
                    const dither = JointDataset.unwrap(dictionary, JointDataset.DitherLabel);
                    trace.y = dither.map((ditherVal, index) => { return rawY[index] + ditherVal; });
                } else {
                    trace.y = rawY;
                }
            } else {
                hovertemplate += metaY.abbridgedLabel + ": %{y}<br>";
                trace.y = rawY;
            }
        }
        hovertemplate += localization.Charts.rowIndex + ": %{customdata.Index}<br>";
        hovertemplate += "<extra></extra>";
        trace.customdata = customdata as any;
        trace.hovertemplate = hovertemplate;
    }

    private generateDefaultChartAxes(): void {
        const yKey = JointDataset.DataLabelRoot + "0";
        const yIsDithered = this.props.jointDataset.metaDict[yKey].isCategorical;
        const chartProps: IGenericChartProps = {
            chartType: ChartTypes.Scatter,
            xAxis: {
                property: this.props.jointDataset.hasPredictedProbabilities ?
                    JointDataset.ProbabilityYRoot + "0" :
                    JointDataset.IndexLabel,
                options: {}
            },
            yAxis: {
                property: yKey,
                options: {
                    dither: yIsDithered,
                    bin: false
                }
            }
        }
        this.props.onChange(chartProps);
    }
}