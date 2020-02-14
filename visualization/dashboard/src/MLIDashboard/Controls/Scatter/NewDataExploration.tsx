import { ComboBox, IComboBox, IComboBoxOption } from "office-ui-fabric-react/lib/ComboBox";
import * as Plotly from "plotly.js-dist";
import React from "react";
import { AccessibleChart, IPlotlyProperty, DefaultSelectionFunctions } from "mlchartlib";
import { localization } from "../../../Localization/localization";
import { FabricStyles } from "../../FabricStyles";
import {  ScatterUtils, INewScatterProps} from "./ScatterUtils";
import _ from "lodash";
import { NoDataMessage, LoadingSpinner } from "../../SharedComponents";
import { mergeStyleSets } from "@uifabric/styling";
import { JointDataset } from "../../JointDataset";
import { IDropdownOption } from "office-ui-fabric-react/lib/Dropdown";
import { IconButton, Button } from "office-ui-fabric-react/lib/Button";
import FilterEditor from "../FilterEditor";
import { IFilter } from "../../Interfaces/IFilter";
import FilterControl from "../FilterControl";
import ChartWithControls, { IGenericChartProps, ChartTypes } from "../ChartWithControls";

export const DataScatterId = "data_scatter_id";

export class NewDataExploration extends React.PureComponent<INewScatterProps> {

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
    constructor(props: INewScatterProps) {
        super(props);
        if (props.chartProps === undefined) {
            this.generateDefaultChartAxes();
        }
        this.axisOptions = this.generateDropdownOptions();
        this.state = {open: false};
        this.scatterSelection = this.scatterSelection.bind(this);
        this.onChartTypeChange = this.onChartTypeChange.bind(this);
    }

    public render(): React.ReactNode {
        if (this.props.chartProps === undefined) {
            // this.generateDefaultChartAxes();
            return (<div/>);
        }
        const plotlyProps = this.generatePlotlyProps();
        const jointData = this.props.dashboardContext.explanationContext.jointDataset;
        return (
            <div className="explanation-chart">
                <FilterControl 
                    jointDataset={this.props.dashboardContext.explanationContext.jointDataset}
                    filterContext={this.props.filterContext}
                />
                <div className="path-selector">
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
                    onChange={this.onChange}
                />
        </div>);
    }

    private readonly onChange = (newProps: IGenericChartProps): void => {
        this.props.onChange(newProps, DataScatterId);
    }

    private onChartTypeChange(event: React.FormEvent<IComboBox>, item: IComboBoxOption): void {
        const newProps = _.cloneDeep(this.props.chartProps);
        newProps.chartType = item.key as ChartTypes;
        this.props.onChange(newProps, DataScatterId);
    }

    private generateDropdownOptions(): IDropdownOption[] {
        const jointData = this.props.dashboardContext.explanationContext.jointDataset;
        return Object.keys(jointData.metaDict).map((key, index) => {
            return {
                key: key,
                text: jointData.metaDict[key].label
            };
        });
    }

    private generatePlotlyProps(): IPlotlyProperty {
        const plotlyProps: IPlotlyProperty = _.cloneDeep(ScatterUtils.baseScatterProperties);
        plotlyProps.data[0].type = this.props.chartProps.chartType;
        plotlyProps.data[0].datapointLevelAccessors = undefined;
        plotlyProps.data[0].hoverinfo = "all";
        let hovertemplate = "";
        const jointData = this.props.dashboardContext.explanationContext.jointDataset;
        if (this.props.chartProps.colorAxis) {
            jointData.sort(this.props.chartProps.colorAxis.property);
        }
        const customdata = jointData.unwrap(JointDataset.IndexLabel).map(val => {
            const dict = {};
            dict[JointDataset.IndexLabel] = val;
            return dict;
        });
        if (this.props.chartProps.xAxis) {
            const shouldBin = this.props.chartProps.chartType !== ChartTypes.Scatter;
            const rawX = jointData.unwrap(this.props.chartProps.xAxis.property, shouldBin);
            if (this.props.chartProps.xAxis.options && this.props.chartProps.xAxis.options.dither) {
                const dithered = jointData.unwrap(JointDataset.DitherLabel);
                plotlyProps.data[0].x = dithered.map((dither, index) => { return rawX[index] + dither;});
                hovertemplate += "x: %{customdata.X}<br>";
                rawX.forEach((val, index) => {
                    // If categorical, show string value in tooltip
                    if (jointData.metaDict[this.props.chartProps.xAxis.property].isCategorical) {
                        customdata[index]["X"] = jointData.metaDict[this.props.chartProps.xAxis.property].sortedCategoricalValues[val];
                    } else {
                        customdata[index]["X"] = val;
                    }
                });
            } else {
                plotlyProps.data[0].x = rawX;
                hovertemplate += "x: %{x}<br>";
            }
        }
        if (this.props.chartProps.yAxis &&  this.props.chartProps.chartType !== ChartTypes.Bar) {
            const rawY = jointData.unwrap(this.props.chartProps.yAxis.property);
            if (this.props.chartProps.yAxis.options && this.props.chartProps.yAxis.options.dither) {
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
        hovertemplate += "<extra></extra>";
        plotlyProps.data[0].customdata = customdata as any;
        plotlyProps.data[0].hovertemplate = hovertemplate;
        return plotlyProps;
    }

    private generateDefaultChartAxes(): void {
        let maxIndex: number = 0;
        let maxVal: number = Number.MIN_SAFE_INTEGER;
        const exp = this.props.dashboardContext.explanationContext;

        if (exp.globalExplanation && exp.globalExplanation.perClassFeatureImportances) {
            // Find the top metric
            exp.globalExplanation.perClassFeatureImportances
                .map(classArray => classArray.reduce((a, b) => a + b), 0)
                .forEach((val, index) => {
                    if (val >= maxVal) {
                        maxIndex = index;
                        maxVal = val;
                    }
                });
        } else if (exp.globalExplanation && exp.globalExplanation.flattenedFeatureImportances) {
            exp.globalExplanation.flattenedFeatureImportances
                .forEach((val, index) => {
                    if (val >= maxVal) {
                        maxIndex = index;
                        maxVal = val;
                    }
                });
        }
        const yKey = JointDataset.DataLabelRoot + maxIndex.toString();
        const yIsDithered = exp.jointDataset.metaDict[yKey].isCategorical;
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
                property: exp.jointDataset.hasPredictedY ?
                    JointDataset.PredictedYLabel : JointDataset.IndexLabel,
                options: {}
            }
        }
        this.props.onChange(chartProps, DataScatterId);
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