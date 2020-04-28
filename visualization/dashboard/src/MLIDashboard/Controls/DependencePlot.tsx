import { ComboBox, IComboBox, IComboBoxOption } from "office-ui-fabric-react/lib/ComboBox";
import * as Plotly from "plotly.js-dist";
import React from "react";
import { AccessibleChart, IPlotlyProperty, DefaultSelectionFunctions, PlotlyMode } from "mlchartlib";
import { mergeStyleSets } from "@uifabric/styling";
import { IDropdownOption } from "office-ui-fabric-react/lib/Dropdown";
import { IconButton, Button, DefaultButton } from "office-ui-fabric-react/lib/Button";
import { Transform } from "plotly.js-dist";
import { IGenericChartProps, ISelectorConfig, ChartTypes } from "../NewExplanationDashboard";
import { JointDataset, ColumnCategories } from "../JointDataset";
import { IExplanationModelMetadata } from "../IExplanationContext";
import { AxisConfigDialog } from "./AxisConfigurationDialog/AxisConfigDialog";
import { localization } from "../../Localization/localization";
import _ from "lodash";
import { Cohort } from "../Cohort";
import { Text } from "office-ui-fabric-react";
import { FabricStyles } from "../FabricStyles";

export interface INewDataTabProps {
    chartProps: IGenericChartProps;
    // selectionContext: SelectionContext;
    theme?: string;
    // messages?: HelpMessageDict;
    jointDataset: JointDataset;
    cohort: Cohort;
    cohortIndex: number;
    metadata: IExplanationModelMetadata;
    onChange: (props: IGenericChartProps) => void;
}

export class DependencePlot extends React.PureComponent<INewDataTabProps> {
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
                l: 10,
                b: 20,
                r: 10
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
                automargin: true,
                color: FabricStyles.chartAxisColor,
                tickfont: {
                    family: "Roboto, Helvetica Neue, sans-serif",
                    size: 11
                },
                zeroline: true,
                showgrid: true,
                gridcolor: "#e5e5e5"
            }
        } as any
    };

    private static readonly classNames = mergeStyleSets({
        DependencePlot: {
            display: "flex",
            flexGrow: "1",
            flexDirection: "row"
        },
        chartWithAxes: {
            display: "flex",
            flex: "1",
            padding: "5px 20px 0 20px",
            flexDirection: "column"
        },
        chart: {
            height: "100%",
            flex: 1
        },
        chartWithVertical: {
            height: "400px",
            width: "100%",
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
        },
        placeholderWrapper: {
            margin: "100px auto 0 auto"
        },
        placeholder: {
            maxWidth: "70%"
        },
        secondaryChartPlacolderBox: {
            height: "400px",
            width: "100%"
        },
        secondaryChartPlacolderSpacer: {
            margin: "25px auto 0 auto",
            padding: "23px",
            width:"fit-content",
            boxShadow: "0px 0px 6px rgba(0, 0, 0, 0.2)"
        },
        faintText: {
            fontWeight: "350" as any,
        }
    });

    public render(): React.ReactNode {
        if (this.props.chartProps === undefined) { 
            return (<div className={DependencePlot.classNames.secondaryChartPlacolderBox}>
                <div className={DependencePlot.classNames.secondaryChartPlacolderSpacer}>
                    <Text variant="large" className={DependencePlot.classNames.faintText}>{localization.DependencePlot.placeholder}</Text>
                </div>
            </div>);
        }
        const plotlyProps = this.generatePlotlyProps();
        return (
            <div className={DependencePlot.classNames.DependencePlot}>
                <div className={DependencePlot.classNames.chartWithAxes}>
                    <div className={DependencePlot.classNames.chartWithVertical}>
                        <div className={DependencePlot.classNames.verticalAxis}>
                            <div className={DependencePlot.classNames.rotatedVerticalBox}>
                                <Text variant={"medium"} block>{localization.DependencePlot.featureImportanceOf}</Text>
                                <Text variant={"medium"}>{this.props.jointDataset.metaDict[this.props.chartProps.xAxis.property].label + " : " + this.props.metadata.classNames[0]}</Text>
                            </div>
                        </div>
                        <div className={DependencePlot.classNames.chart}>
                            <AccessibleChart
                                plotlyProps={plotlyProps}
                                theme={undefined}
                            />
                        </div>
                    </div>
                    <div className={DependencePlot.classNames.horizontalAxisWithPadding}>
                        <div className={DependencePlot.classNames.paddingDiv}></div>
                        <div className={DependencePlot.classNames.horizontalAxis}>
                            <Text variant={"medium"}>{this.props.jointDataset.metaDict[this.props.chartProps.xAxis.property].label}</Text>
                        </div>
                    </div>
                </div >
        </div>);
    }

    private onXSet(value: ISelectorConfig): void {
        const newProps = _.cloneDeep(this.props.chartProps);
        newProps.xAxis = value;
        const propMeta = this.props.jointDataset.metaDict[value.property]
        newProps.yAxis = {
            property: JointDataset.ReducedLocalImportanceRoot + propMeta.index,
            options: {}
        };
        this.props.onChange(newProps);
        this.setState({xDialogOpen: false})
    }

    private onColorSet(value: ISelectorConfig): void {
        const newProps = _.cloneDeep(this.props.chartProps);
        newProps.colorAxis = value;
        this.props.onChange(newProps);
        this.setState({colorDialogOpen: false})
    }

    private generatePlotlyProps(): IPlotlyProperty {
        const plotlyProps = _.cloneDeep(DependencePlot.basePlotlyProperties);
        const jointData = this.props.jointDataset;
        const cohort = this.props.cohort;
        plotlyProps.data[0].hoverinfo = "all";
        let hovertemplate = "";
        if (this.props.chartProps.colorAxis && (this.props.chartProps.colorAxis.options.bin ||
            jointData.metaDict[this.props.chartProps.colorAxis.property].isCategorical)) {
                cohort.sort(this.props.chartProps.colorAxis.property);
        }
        const customdata = cohort.unwrap(JointDataset.IndexLabel).map(val => {
            const dict = {};
            dict[JointDataset.IndexLabel] = val;
            return dict;
        });
        plotlyProps.data[0].type = this.props.chartProps.chartType;
        plotlyProps.data[0].mode = PlotlyMode.markers;
        plotlyProps.data[0].marker = { color: FabricStyles.fabricColorPalette[this.props.cohortIndex]}
        if (this.props.chartProps.xAxis) {
            if (jointData.metaDict[this.props.chartProps.xAxis.property].isCategorical) {
                const xLabels = jointData.metaDict[this.props.chartProps.xAxis.property].sortedCategoricalValues;
                const xLabelIndexes = xLabels.map((unused, index) => index);
                _.set(plotlyProps, "layout.xaxis.ticktext", xLabels);
                _.set(plotlyProps, "layout.xaxis.tickvals", xLabelIndexes);
            }
            const rawX = cohort.unwrap(this.props.chartProps.xAxis.property);
            if (this.props.chartProps.xAxis.options.dither) {
                const dithered = cohort.unwrap(JointDataset.DitherLabel);
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
            const rawY = cohort.unwrap(this.props.chartProps.yAxis.property);
            if (this.props.chartProps.yAxis.options.dither) {
                const dithered = cohort.unwrap(JointDataset.DitherLabel);
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
            const rawColor = cohort.unwrap(this.props.chartProps.colorAxis.property, isBinned);
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
        hovertemplate += "<extra></extra>";
        plotlyProps.data[0].customdata = customdata as any;
        plotlyProps.data[0].hovertemplate = hovertemplate;
        return plotlyProps;
    }

    private generateDefaultChartAxes(): void {
        let maxIndex: number = 0;

        const xKey = JointDataset.DataLabelRoot + maxIndex.toString();
        const xIsDithered = this.props.jointDataset.metaDict[xKey].isCategorical;
        const yKey = JointDataset.ReducedLocalImportanceRoot + maxIndex.toString();
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