import { ComboBox, IComboBox, IComboBoxOption } from "office-ui-fabric-react/lib/ComboBox";
import * as Plotly from "plotly.js-dist";
import React from "react";
import { AccessibleChart, IPlotlyProperty, DefaultSelectionFunctions, PlotlyMode, IPlotlyAnimateProps } from "mlchartlib";
import { localization } from "../../../Localization/localization";
import { FabricStyles } from "../../FabricStyles";
import { ScatterUtils } from "./ScatterUtils";
import _ from "lodash";
import { NoDataMessage, LoadingSpinner } from "../../SharedComponents";
import { mergeStyleSets } from "@uifabric/styling";
import { JointDataset, ColumnCategories } from "../../JointDataset";
import { IDropdownOption } from "office-ui-fabric-react/lib/Dropdown";
import { IconButton, Button, DefaultButton } from "office-ui-fabric-react/lib/Button";
import { IFilter, IFilterContext } from "../../Interfaces/IFilter";
import { FilterControl } from "../FilterControl";
import { IExplanationModelMetadata } from "../../IExplanationContext";
import { Transform } from "plotly.js-dist";
import { ISelectorConfig, IGenericChartProps, ChartTypes } from "../../NewExplanationDashboard";
import { AxisConfigDialog } from "../AxisConfigDialog";

export interface INewDataTabProps {
    chartProps: IGenericChartProps;
    // selectionContext: SelectionContext;
    theme?: string;
    // messages?: HelpMessageDict;
    jointDataset: JointDataset;
    metadata: IExplanationModelMetadata;
    filterContext: IFilterContext;
    onChange: (props: IGenericChartProps) => void;
}

export interface INewDataTabState {
    xDialogOpen: boolean;
    yDialogOpen: boolean;
    colorDialogOpen: boolean;
}

export class NewDataExploration extends React.PureComponent<INewDataTabProps, INewDataTabState> {
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
                b: 0,
            },
            hovermode: "closest",
            showlegend: false,
            yaxis: {
                automargin: true
            },
        } as any
    };

    private static readonly classNames = mergeStyleSets({
        dataTab: {
            display: "contents"
        },
        topConfigArea: {
            display: "flex",
            padding: "3px 15px",
            justifyContent: "space-between"
        },
        chartWithAxes: {
            display: "flex",
            padding: "5px 20px 0 20px",
            flexDirection: "column"
        },
        chartWithVertical: {
            display: "flex",
            flexDirection: "row"
        },
        verticalAxis: {
            position: "relative",
            top: "0px",
            height: "auto",
            width: "50px"
        },
        rotatedVerticalBox: {
            transform: "translateX(-50%) translateY(-50%) rotate(270deg)",
            marginLeft: "15px",
            position: "absolute",
            top: "50%",
            textAlign: "center",
            width: "max-content"
        },
        horizontalAxisWithPadding: {
            display: "flex",
            flexDirection: "row"
        },
        paddingDiv: {
            width: "50px"
        },
        horizontalAxis: {
            flex: 1,
            textAlign:"center"
        }
    });

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

    constructor(props: INewDataTabProps) {
        super(props);
        if (props.chartProps === undefined) {
            this.generateDefaultChartAxes();
        }
        this.onXSet = this.onXSet.bind(this);
        this.onYSet = this.onYSet.bind(this);
        this.onColorSet = this.onColorSet.bind(this);
        this.scatterSelection = this.scatterSelection.bind(this);
        this.onChartTypeChange = this.onChartTypeChange.bind(this);

        this.state = {
            xDialogOpen: false,
            yDialogOpen: false,
            colorDialogOpen: false
        };
    }

    public render(): React.ReactNode {
        if (this.props.chartProps === undefined) {
            return (<div/>);
        }
        const plotlyProps = NewDataExploration.generatePlotlyProps(
            this.props.jointDataset,
            this.props.chartProps
        );
        const jointData = this.props.jointDataset;
        return (
            <div className={NewDataExploration.classNames.dataTab}>
                <FilterControl 
                    jointDataset={jointData}
                    filterContext={this.props.filterContext}
                />
                <div className={NewDataExploration.classNames.topConfigArea}>
                    <ComboBox
                        options={this.chartOptions}
                        onChange={this.onChartTypeChange}
                        selectedKey={this.props.chartProps.chartType}
                        useComboBoxAsMenuWidth={true}
                        styles={FabricStyles.defaultDropdownStyle}
                    />
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
                </div>
                <div className={NewDataExploration.classNames.chartWithAxes}>
                    <div className={NewDataExploration.classNames.chartWithVertical}>
                        <div className={NewDataExploration.classNames.verticalAxis}>
                            <div className={NewDataExploration.classNames.rotatedVerticalBox}>
                                {(this.props.chartProps.chartType === ChartTypes.Scatter) && (
                                    <DefaultButton 
                                        onClick={this.setYOpen.bind(this, true)}
                                        id={this._yButtonId}
                                        text={localization.ExplanationScatter.yValue + this.props.jointDataset.metaDict[this.props.chartProps.yAxis.property].abbridgedLabel}
                                        title={localization.ExplanationScatter.yValue + this.props.jointDataset.metaDict[this.props.chartProps.yAxis.property].label}
                                    />
                                )}
                                {(this.props.chartProps.chartType !== ChartTypes.Scatter) && (
                                    <div>{localization.ExplanationScatter.count}</div>
                                )}
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
                    <div className={NewDataExploration.classNames.horizontalAxisWithPadding}>
                        <div className={NewDataExploration.classNames.paddingDiv}></div>
                        <div className={NewDataExploration.classNames.horizontalAxis}>
                            <DefaultButton 
                                onClick={this.setXOpen.bind(this, true)}
                                id={this._xButtonId}
                                text={localization.ExplanationScatter.xValue + this.props.jointDataset.metaDict[this.props.chartProps.xAxis.property].abbridgedLabel}
                                title={localization.ExplanationScatter.xValue + this.props.jointDataset.metaDict[this.props.chartProps.xAxis.property].label}
                            />
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

    private static generatePlotlyProps(jointData: JointDataset, chartProps: IGenericChartProps): IPlotlyProperty {
        const plotlyProps = _.cloneDeep(NewDataExploration.basePlotlyProperties);
        plotlyProps.data[0].hoverinfo = "all";
        if (chartProps.colorAxis && (chartProps.colorAxis.options.bin ||
            jointData.metaDict[chartProps.colorAxis.property].isCategorical)) {
                jointData.sort(chartProps.colorAxis.property);
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
                    const rawX = jointData.unwrap(chartProps.xAxis.property);
                    if (chartProps.xAxis.options.dither) {
                        const dithered = jointData.unwrap(JointDataset.DitherLabel);
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
                    const rawY = jointData.unwrap(chartProps.yAxis.property);
                    if (chartProps.yAxis.options.dither) {
                        const dithered = jointData.unwrap(JointDataset.DitherLabel);
                        plotlyProps.data[0].y = dithered.map((dither, index) => { return rawY[index] + dither;});
                    } else {
                        plotlyProps.data[0].y = rawY;
                    }
                }
                if (chartProps.colorAxis) {
                    const isBinned = chartProps.colorAxis.options && chartProps.colorAxis.options.bin;
                    const rawColor = jointData.unwrap(chartProps.colorAxis.property, isBinned);
                    // handle binning to categories later
                    if (jointData.metaDict[chartProps.colorAxis.property].isCategorical || isBinned) {
                        const styles = jointData.metaDict[chartProps.colorAxis.property].sortedCategoricalValues.map((label, index) => {
                            return {
                                target: index,
                                value: { name: label}
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
                const rawX = jointData.unwrap(chartProps.xAxis.property, true);
                
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
                    const rawColor = jointData.unwrap(chartProps.colorAxis.property, true);
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
        plotlyProps.data[0].customdata = this.buildCustomData(jointData, chartProps);
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

    private static buildCustomData(jointData: JointDataset, chartProps: IGenericChartProps): Array<any> {
        const customdata = jointData.unwrap(JointDataset.IndexLabel).map(val => {
            const dict = {};
            dict[JointDataset.IndexLabel] = val;
            return dict;
        });
        if (chartProps.chartType === ChartTypes.Scatter) {
            const xAxis = chartProps.xAxis;
            if (xAxis && xAxis.property && xAxis.options.dither) {
                const rawX = jointData.unwrap(chartProps.xAxis.property);
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
                const rawY = jointData.unwrap(chartProps.yAxis.property);
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