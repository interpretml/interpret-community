import { ComboBox, IComboBox, IComboBoxOption } from "office-ui-fabric-react/lib/ComboBox";
import * as Plotly from "plotly.js-dist";
import React from "react";
import { AccessibleChart, IPlotlyProperty, DefaultSelectionFunctions, PlotlyMode } from "mlchartlib";
import { localization } from "../../../Localization/localization";
import { FabricStyles } from "../../FabricStyles";
import { ScatterUtils } from "./ScatterUtils";
import _ from "lodash";
import { NoDataMessage, LoadingSpinner } from "../../SharedComponents";
import { mergeStyleSets } from "@uifabric/styling";
import { JointDataset } from "../../JointDataset";
import { IDropdownOption } from "office-ui-fabric-react/lib/Dropdown";
import { IconButton, Button } from "office-ui-fabric-react/lib/Button";
import { IFilter, IFilterContext } from "../../Interfaces/IFilter";
import { FilterControl } from "../FilterControl";
import { IExplanationModelMetadata } from "../../IExplanationContext";
import { Transform } from "plotly.js-dist";
import { ISelectorConfig, IGenericChartProps } from "../../NewExplanationDashboard";

export enum ChartTypes {
    Scatter = "scattergl",
    Bar = "histogram",
    Box = "box"
}

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
                t: 10
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
        }
    });

    private axisOptions: IDropdownOption[];
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
    constructor(props: INewDataTabProps) {
        super(props);
        if (props.chartProps === undefined) {
            this.generateDefaultChartAxes();
        }
        this.axisOptions = this.generateDropdownOptions();
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
        const plotlyProps = this.generatePlotlyProps();
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
                </div>
                <ChartWithControls
                    filters={this.props.filterContext}
                    axisOptions={this.axisOptions}
                    jointDataset={jointData}
                    chartProps={this.props.chartProps}
                    onChange={this.props.onChange}
                />
        </div>);
    }

    private onChartTypeChange(event: React.FormEvent<IComboBox>, item: IComboBoxOption): void {
        const newProps = _.cloneDeep(this.props.chartProps);
        newProps.chartType = item.key as ChartTypes;
        this.props.onChange(newProps);
    }

    private generateDropdownOptions(): IDropdownOption[] {
        const jointData = this.props.jointDataset;
        return Object.keys(jointData.metaDict).map((key, index) => {
            return {
                key: key,
                text: jointData.metaDict[key].label
            };
        });
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

    private generatePlotlyProps(): IPlotlyProperty {
        const plotlyProps = _.cloneDeep(NewDataExploration.basePlotlyProperties);
        const jointData = this.props.jointDataset;
        plotlyProps.data[0].hoverinfo = "all";
        let hovertemplate = "";
        if (this.props.chartProps.colorAxis && (this.props.chartProps.colorAxis.options.bin ||
            jointData.metaDict[this.props.chartProps.colorAxis.property].isCategorical)) {
                jointData.sort(this.props.chartProps.colorAxis.property);
        }
        const customdata = jointData.unwrap(JointDataset.IndexLabel).map(val => {
            const dict = {};
            dict[JointDataset.IndexLabel] = val;
            return dict;
        });
        switch(this.props.chartProps.chartType) {
            case ChartTypes.Scatter: {
                plotlyProps.data[0].type = this.props.chartProps.chartType;
                plotlyProps.data[0].mode = PlotlyMode.markers;
                if (this.props.chartProps.xAxis) {
                    if (jointData.metaDict[this.props.chartProps.xAxis.property].isCategorical) {
                        const xLabels = jointData.metaDict[this.props.chartProps.xAxis.property].sortedCategoricalValues;
                        const xLabelIndexes = xLabels.map((unused, index) => index);
                        _.set(plotlyProps, "layout.xaxis.ticktext", xLabels);
                        _.set(plotlyProps, "layout.xaxis.tickvals", xLabelIndexes);
                    }
                    const rawX = jointData.unwrap(this.props.chartProps.xAxis.property);
                    if (this.props.chartProps.xAxis.options.dither) {
                        const dithered = jointData.unwrap(JointDataset.DitherLabel);
                        plotlyProps.data[0].x = dithered.map((dither, index) => { return rawX[index] + dither;});
                        hovertemplate += "x: %{customdata.X}<br>";
                        rawX.forEach((val, index) => {
                            // If categorical, show string value in tooltip
                            if (jointData.metaDict[this.props.chartProps.xAxis.property].isCategorical) {
                                customdata[index]["X"] = jointData.metaDict[this.props.chartProps.xAxis.property]
                                    .sortedCategoricalValues[val];
                            } else {
                                customdata[index]["X"] = val;
                            }
                        });
                    } else {
                        plotlyProps.data[0].x = rawX;
                        hovertemplate += "x: %{x}<br>";
                    }
                }
                if (this.props.chartProps.yAxis) {
                    if (jointData.metaDict[this.props.chartProps.yAxis.property].isCategorical) {
                        const yLabels = jointData.metaDict[this.props.chartProps.yAxis.property].sortedCategoricalValues;
                        const yLabelIndexes = yLabels.map((unused, index) => index);
                        _.set(plotlyProps, "layout.yaxis.ticktext", yLabels);
                        _.set(plotlyProps, "layout.yaxis.tickvals", yLabelIndexes);
                    }
                    const rawY = jointData.unwrap(this.props.chartProps.yAxis.property);
                    if (this.props.chartProps.yAxis.options.dither) {
                        const dithered = jointData.unwrap(JointDataset.DitherLabel);
                        plotlyProps.data[0].y = dithered.map((dither, index) => { return rawY[index] + dither;});
                        hovertemplate += "y: %{customdata.Y}<br>";
                        rawY.forEach((val, index) => {
                            // If categorical, show string value in tooltip
                            if (jointData.metaDict[this.props.chartProps.yAxis.property].isCategorical) {
                                customdata[index]["Y"] = jointData.metaDict[this.props.chartProps.yAxis.property].sortedCategoricalValues[val];
                            } else {
                                customdata[index]["Y"] = val;
                            }
                        });
                    } else {
                        plotlyProps.data[0].y = rawY;
                        hovertemplate += "y: %{y}<br>";
                    }
                }
                if (this.props.chartProps.colorAxis) {
                    const isBinned = this.props.chartProps.colorAxis.options && this.props.chartProps.colorAxis.options.bin;
                    const rawColor = jointData.unwrap(this.props.chartProps.colorAxis.property, isBinned);
                    // handle binning to categories later
                    if (jointData.metaDict[this.props.chartProps.colorAxis.property].isCategorical || isBinned) {
                        const styles = jointData.metaDict[this.props.chartProps.colorAxis.property].sortedCategoricalValues.map((label, index) => {
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
                const rawX = jointData.unwrap(this.props.chartProps.xAxis.property, true);
                
                const xLabels = jointData.metaDict[this.props.chartProps.xAxis.property].sortedCategoricalValues;
                const y = new Array(rawX.length).fill(1);
                const xLabelIndexes = xLabels.map((unused, index) => index);
                plotlyProps.data[0].text = rawX.map(index => xLabels[index]);
                plotlyProps.data[0].x = rawX;
                plotlyProps.data[0].y = y;
                _.set(plotlyProps, "layout.xaxis.ticktext", xLabels);
                _.set(plotlyProps, "layout.xaxis.tickvals", xLabelIndexes);
                hovertemplate += "x: %{text}<br>";
                hovertemplate += "count: %{y}<br>";
                const transforms: Partial<Transform>[] = [
                    {
                        type: "aggregate",
                        groups: rawX,
                        aggregations: [
                          {target: "y", func: "sum"},
                        ]
                    }
                ];
                if (this.props.chartProps.colorAxis) {
                    const rawColor = jointData.unwrap(this.props.chartProps.colorAxis.property, true);
                    const styles = jointData.metaDict[this.props.chartProps.colorAxis.property].sortedCategoricalValues.map((label, index) => {
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
        hovertemplate += "<extra></extra>";
        plotlyProps.data[0].customdata = customdata as any;
        plotlyProps.data[0].hovertemplate = hovertemplate;
        return plotlyProps;
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