import { ComboBox, IComboBox, IComboBoxOption } from "office-ui-fabric-react/lib/ComboBox";
import * as Plotly from "plotly.js-dist";
import React from "react";
import { AccessibleChart, IPlotlyProperty, DefaultSelectionFunctions, PlotlyMode, IPlotlyAnimateProps } from "mlchartlib";
import { localization } from "../../../Localization/localization";
import { FabricStyles } from "../../FabricStyles";
import _ from "lodash";
import { NoDataMessage, LoadingSpinner } from "../../SharedComponents";
import { mergeStyleSets } from "@uifabric/styling";
import { JointDataset, ColumnCategories } from "../../JointDataset";
import { IDropdownOption, Dropdown } from "office-ui-fabric-react/lib/Dropdown";
import { IconButton, Button, DefaultButton } from "office-ui-fabric-react/lib/Button";
import { IFilter, IFilterContext } from "../../Interfaces/IFilter";
import { FilterControl } from "../FilterControl";
import { IExplanationModelMetadata } from "../../IExplanationContext";
import { Transform } from "plotly.js-dist";
import { ISelectorConfig, IGenericChartProps, ChartTypes } from "../../NewExplanationDashboard";
import { AxisConfigDialog } from "../AxisConfigDialog";
import { Cohort } from "../../Cohort";
import { dastasetExplorerTabStyles as datasetExplorerTabStyles } from "./DatasetExplorerTab.styles";
import { Icon, Text } from "office-ui-fabric-react";

export interface IDatasetExplorerTabProps {
    chartProps: IGenericChartProps;
    // selectionContext: SelectionContext;
    theme?: string;
    // messages?: HelpMessageDict;
    jointDataset: JointDataset;
    metadata: IExplanationModelMetadata;
    cohorts: Cohort[];
    onChange: (props: IGenericChartProps) => void;
}

export interface IDatasetExplorerTabState {
    xDialogOpen: boolean;
    yDialogOpen: boolean;
    colorDialogOpen: boolean;
    selectedCohortIndex: number;
}

export class DatasetExplorerTab extends React.PureComponent<IDatasetExplorerTabProps, IDatasetExplorerTabState> {
    public static basePlotlyProperties: IPlotlyProperty = {
        config: { displaylogo: false, responsive: true, displayModeBar: false},
        data: [{}],
        layout: {
            dragmode: false,
            autosize: true,
            font: {
                size: 10
            },
            margin: {
                t: 10,
                l: 0,
                b: 20,
            },
            hovermode: "closest",
            showlegend: false,
            yaxis: {
                automargin: true,
                color: FabricStyles.chartAxisColor,
                tickfont: {
                    family: "Roboto, Helvetica Neue, sans-serif",
                    size: 11
                },
                zeroline: true,
                showgrid: true,
                gridcolor: "#e5e5e5"
            },
            xaxis: {
                mirror: true,
                color: FabricStyles.chartAxisColor,
                tickfont: {
                    family: FabricStyles.fontFamilies,
                    size: 11
                },
                zeroline: true
            }
        } as any
    };

    private chartOptions: IComboBoxOption[] = [
        {
            key: ChartTypes.Scatter,
            text: "Scatter"
        },
        {
            key: ChartTypes.Bar,
            text: "Histogram"
        }
    ];

    private readonly _xButtonId = "x-button-id";
    private readonly _colorButtonId = "color-button-id";
    private readonly _yButtonId = "y-button-id";

    constructor(props: IDatasetExplorerTabProps) {
        super(props);
        if (props.chartProps === undefined) {
            this.generateDefaultChartAxes();
        }
        this.onXSet = this.onXSet.bind(this);
        this.onYSet = this.onYSet.bind(this);
        this.onColorSet = this.onColorSet.bind(this);
        this.scatterSelection = this.scatterSelection.bind(this);
        this.onChartTypeChange = this.onChartTypeChange.bind(this);
        this.setSelectedCohort = this.setSelectedCohort.bind(this);

        this.state = {
            xDialogOpen: false,
            yDialogOpen: false,
            colorDialogOpen: false,
            selectedCohortIndex: 0
        };
    }

    public render(): React.ReactNode {
        const classNames = datasetExplorerTabStyles();
        if (this.props.chartProps === undefined) {
            return (<div/>);
        }
        const plotlyProps = DatasetExplorerTab.generatePlotlyProps(
            this.props.jointDataset,
            this.props.chartProps,
            this.props.cohorts[this.state.selectedCohortIndex]
        );
        const cohortOptions: IDropdownOption[] = this.props.chartProps.xAxis.property !== Cohort.CohortKey ?
            this.props.cohorts.map((cohort, index) => {return {key: index, text: cohort.name};}) : undefined;
        return (
            <div className={classNames.page}>
                <div className={classNames.infoWithText}>
                    <Icon iconName="Info" className={classNames.infoIcon}/>
                    <Text variant="medium" className={classNames.helperText}>{localization.DatasetExplorer.helperText}</Text>
                </div>
                <div className={classNames.cohortPickerWrapper}>
                    <Text variant="mediumPlus" className={classNames.cohortPickerLabel}>{localization.ModelPerformance.cohortPickerLabel}</Text>
                    <Dropdown 
                        styles={{ dropdown: { width: 150 } }}
                        options={cohortOptions}
                        selectedKey={this.state.selectedCohortIndex}
                        onChange={this.setSelectedCohort}
                    />
                </div>
                {/* <div>
                    <DefaultButton 
                        onClick={this.setColorOpen.bind(this, true)}
                        id={this._colorButtonId}
                        text={localization.ExplanationScatter.colorValue + this.props.jointDataset.metaDict[this.props.chartProps.colorAxis.property].abbridgedLabel}
                        title={localization.ExplanationScatter.colorValue + this.props.jointDataset.metaDict[this.props.chartProps.colorAxis.property].label}
                    />
                    {(this.state.colorDialogOpen) && (
                        <AxisConfigDialog 
                            jointDataset={this.props.jointDataset}
                            orderedGroupTitles={[ColumnCategories.index, ColumnCategories.dataset, ColumnCategories.outcome]}
                            selectedColumn={this.props.chartProps.colorAxis}
                            canBin={true}
                            mustBin={false}
                            canDither={false}
                            onAccept={this.onColorSet}
                            onCancel={this.setColorOpen.bind(this, false)}
                            target={this._colorButtonId}
                        />
                    )}
                </div> */}
                <div className={classNames.chartWithAxes}>
                    <div className={classNames.chartWithVertical}>
                        <div className={classNames.verticalAxis}>
                            <div className={classNames.rotatedVerticalBox}>
                                <div>
                                    <Text block variant="mediumPlus" className={classNames.boldText}>{localization.Charts.yValue}</Text>
                                    <DefaultButton 
                                        onClick={this.setYOpen.bind(this, true)}
                                        id={this._yButtonId}
                                        text={this.props.jointDataset.metaDict[this.props.chartProps.yAxis.property].abbridgedLabel}
                                        title={this.props.jointDataset.metaDict[this.props.chartProps.yAxis.property].label}
                                    />
                                </div>
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
                </div>
        </div>);
    }

    private setSelectedCohort(event: React.FormEvent<HTMLDivElement>, item: IDropdownOption): void {
        this.setState({selectedCohortIndex: item.key as number});
    }

    private onChartTypeChange(event: React.FormEvent<IComboBox>, item: IComboBoxOption): void {
        const newProps = _.cloneDeep(this.props.chartProps);
        newProps.chartType = item.key as ChartTypes;
        this.props.onChange(newProps);
    }

    private readonly setXOpen = (val: boolean): void => {
        if (val && this.state.xDialogOpen === false) {
            this.setState({xDialogOpen: true});
            return;
        }
        this.setState({xDialogOpen: false});
    }

    private readonly setColorOpen = (val: boolean): void => {
        if (val && this.state.colorDialogOpen === false) {
            this.setState({colorDialogOpen: true});
            return;
        }
        this.setState({colorDialogOpen: false});
    }

    private readonly setYOpen = (val: boolean): void => {
        if (val && this.state.yDialogOpen === false) {
            this.setState({yDialogOpen: true});
            return;
        }
        this.setState({yDialogOpen: false});
    }

    private onXSet(value: ISelectorConfig): void {
        const newProps = _.cloneDeep(this.props.chartProps);
        newProps.xAxis = value;
        this.props.onChange(newProps);
        this.setState({xDialogOpen: false})
    }

    private onYSet(value: ISelectorConfig): void {
        const newProps = _.cloneDeep(this.props.chartProps);
        newProps.yAxis = value;
        this.props.onChange(newProps);
        this.setState({yDialogOpen: false})
    }

    private onColorSet(value: ISelectorConfig): void {
        const newProps = _.cloneDeep(this.props.chartProps);
        newProps.colorAxis = value;
        this.props.onChange(newProps);
        this.setState({colorDialogOpen: false})
    }

    private static generatePlotlyProps(jointData: JointDataset, chartProps: IGenericChartProps, cohort: Cohort): IPlotlyProperty {
        const plotlyProps = _.cloneDeep(DatasetExplorerTab.basePlotlyProperties);
        plotlyProps.data[0].hoverinfo = "all";
        if (chartProps.colorAxis && (chartProps.colorAxis.options.bin ||
            jointData.metaDict[chartProps.colorAxis.property].isCategorical)) {
                cohort.sort(chartProps.colorAxis.property);
        }
        switch(chartProps.chartType) {
            case ChartTypes.Scatter: {
                plotlyProps.data[0].type = chartProps.chartType;
                plotlyProps.data[0].mode = PlotlyMode.markers;
                if (chartProps.xAxis) {
                    if (jointData.metaDict[chartProps.xAxis.property].isCategorical) {
                        const xLabels = jointData.metaDict[chartProps.xAxis.property].sortedCategoricalValues;
                        const xLabelIndexes = xLabels.map((unused, index) => index);
                        _.set(plotlyProps, "layout.xaxis.ticktext", xLabels);
                        _.set(plotlyProps, "layout.xaxis.tickvals", xLabelIndexes);
                    }
                    const rawX = cohort.unwrap(chartProps.xAxis.property);
                    if (chartProps.xAxis.options.dither) {
                        const dithered = cohort.unwrap(JointDataset.DitherLabel);
                        plotlyProps.data[0].x = dithered.map((dither, index) => { return rawX[index] + dither;});
                    } else {
                        plotlyProps.data[0].x = rawX;
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
                    if (chartProps.yAxis.options.dither) {
                        const dithered = cohort.unwrap(JointDataset.DitherLabel);
                        plotlyProps.data[0].y = dithered.map((dither, index) => { return rawY[index] + dither;});
                    } else {
                        plotlyProps.data[0].y = rawY;
                    }
                }
                if (chartProps.colorAxis) {
                    const isBinned = chartProps.colorAxis.options && chartProps.colorAxis.options.bin;
                    const rawColor = cohort.unwrap(chartProps.colorAxis.property, isBinned);
                    // handle binning to categories later
                    if (jointData.metaDict[chartProps.colorAxis.property].isCategorical || isBinned) {
                        const styles = jointData.metaDict[chartProps.colorAxis.property].sortedCategoricalValues.map((label, index) => {
                            return {
                                target: index,
                                value: { 
                                    name: label,
                                    marker: {
                                        color: FabricStyles.fabricColorPalette[index]
                                    }
                                }
                            };
                        });
                        plotlyProps.data[0].transforms = [{
                            type: "groupby",
                            groups: rawColor,
                            styles
                        }];
                        plotlyProps.layout.showlegend = true;
                    } else {
                        plotlyProps.data[0].marker = {
                            color: rawColor,
                            colorbar: {
                                title: {
                                    side: "right",
                                    text: "placeholder"
                                } as any
                            },
                            colorscale: "Bluered"
                        };
                    }
                }
                break;
            }
            case ChartTypes.Bar: {
                // for now, treat all bar charts as histograms, the issue with plotly implemented histogram is
                // it tries to bin the data passed to it(we'd like to apply the user specified bins.)
                plotlyProps.data[0].type = "bar";
                const rawX = cohort.unwrap(chartProps.xAxis.property, true);
                
                const xLabels = jointData.metaDict[chartProps.xAxis.property].sortedCategoricalValues;
                const y = new Array(rawX.length).fill(1);
                const xLabelIndexes = xLabels.map((unused, index) => index);
                plotlyProps.data[0].text = rawX.map(index => xLabels[index]);
                plotlyProps.data[0].x = rawX;
                plotlyProps.data[0].y = y;
                _.set(plotlyProps, "layout.xaxis.ticktext", xLabels);
                _.set(plotlyProps, "layout.xaxis.tickvals", xLabelIndexes);
                const transforms: Partial<Transform>[] = [
                    {
                        type: "aggregate",
                        groups: rawX,
                        aggregations: [
                          {target: "y", func: "sum"},
                        ]
                    }
                ];
                if (chartProps.colorAxis) {
                    const rawColor = cohort.unwrap(chartProps.colorAxis.property, true);
                    const styles = jointData.metaDict[chartProps.colorAxis.property].sortedCategoricalValues.map((label, index) => {
                        return {
                            target: index,
                            value: { name: label}
                        };
                    });
                    transforms.push({
                        type: "groupby",
                        groups: rawColor,
                        styles
                    });
                    plotlyProps.layout.showlegend = true;
                }
                plotlyProps.data[0].transforms = transforms;
                break;
            }
        }
        plotlyProps.data[0].customdata = this.buildCustomData(jointData, chartProps, cohort);
        plotlyProps.data[0].hovertemplate = this.buildHoverTemplate(chartProps);
        return plotlyProps;
    }

    private static buildHoverTemplate(chartProps: IGenericChartProps): string {
        let hovertemplate = "";
        switch(chartProps.chartType) {
            case ChartTypes.Scatter: {
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
                break;
            }
            case ChartTypes.Bar: {
                hovertemplate += "x: %{text}<br>";
                hovertemplate += "count: %{y}<br>";
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
        let maxIndex: number = 0;
        let maxVal: number = Number.MIN_SAFE_INTEGER;
        // const exp = this.props.dashboardContext.explanationContext;

        // if (exp.globalExplanation && exp.globalExplanation.perClassFeatureImportances) {
        //     // Find the top metric
        //     exp.globalExplanation.perClassFeatureImportances
        //         .map(classArray => classArray.reduce((a, b) => a + b), 0)
        //         .forEach((val, index) => {
        //             if (val >= maxVal) {
        //                 maxIndex = index;
        //                 maxVal = val;
        //             }
        //         });
        // } else if (exp.globalExplanation && exp.globalExplanation.flattenedFeatureImportances) {
        //     exp.globalExplanation.flattenedFeatureImportances
        //         .forEach((val, index) => {
        //             if (val >= maxVal) {
        //                 maxIndex = index;
        //                 maxVal = val;
        //             }
        //         });
        // }
        const yKey = JointDataset.DataLabelRoot + maxIndex.toString();
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

    private scatterSelection(guid: string, selections: string[], plotlyProps: IPlotlyProperty): void {
        const selectedPoints =
            selections.length === 0
                ? null
                : plotlyProps.data.map(trace => {
                      const selectedIndexes: number[] = [];
                      if ((trace as any).customdata) {
                          ((trace as any).customdata as any[]).forEach((dict, index) => {
                              if (selections.indexOf(dict[JointDataset.IndexLabel]) !== -1) {
                                  selectedIndexes.push(index);
                              }
                          });
                      }
                      return selectedIndexes;
                  });
        Plotly.restyle(guid, "selectedpoints" as any, selectedPoints as any);
        const newLineWidths =
            selections.length === 0
                ? [0]
                : plotlyProps.data.map(trace => {
                    if ((trace as any).customdata) {
                        const customData = ((trace as any).customdata as string[]);
                        const newWidths: number[] = new Array(customData.length).fill(0);
                        customData.forEach((id, index) => {
                            if (selections.indexOf(id) !== -1) {
                                newWidths[index] = 2;
                            }
                        });
                        return newWidths;
                    }
                    return [0];
                  });
        Plotly.restyle(guid, "marker.line.width" as any, newLineWidths as any);
    }
}