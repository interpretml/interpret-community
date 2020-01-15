import { ComboBox, IComboBox, IComboBoxOption } from "office-ui-fabric-react/lib/ComboBox";
import React from "react";
import { AccessibleChart, IPlotlyProperty, DefaultSelectionFunctions } from "mlchartlib";
import { localization } from "../../../Localization/localization";
import { FabricStyles } from "../../FabricStyles";
import {  ScatterUtils, INewScatterProps, IGenericChartProps } from "./ScatterUtils";
import _ from "lodash";
import { NoDataMessage, LoadingSpinner } from "../../SharedComponents";
import { mergeStyleSets } from "@uifabric/styling";
import { JointDataset } from "../../JointDataset";

export const DataScatterId = 'data_scatter_id';

export class NewDataExploration extends React.PureComponent<INewScatterProps> {

    constructor(props: INewScatterProps) {
        super(props);
        // if (props.chartProps === undefined) {
        //     this.generateDefaultChartAxes();
        // }
    }

    public render(): React.ReactNode {
        if (this.props.chartProps === undefined) {
            this.generateDefaultChartAxes();
            return (<div/>);
        }
        const plotlyProps = this.generatePlotlyProps();
        return (<AccessibleChart
                        plotlyProps={plotlyProps}
                        sharedSelectionContext={this.props.selectionContext}
                        theme={this.props.theme}
                        onSelection={DefaultSelectionFunctions.scatterSelection}
                    />);
    }

    private generatePlotlyProps(): IPlotlyProperty {
        const plotlyProps: IPlotlyProperty = _.cloneDeep(ScatterUtils.baseScatterProperties);
        plotlyProps.data[0].datapointLevelAccessors = undefined;
        plotlyProps.data[0].hoverinfo = undefined;
        let hovertemplate = "";
        const jointData = this.props.dashboardContext.explanationContext.jointDataset;
        const customdata = jointData.unwrap(JointDataset.IndexLabel).map(val => {
            const dict = {};
            dict[JointDataset.IndexLabel] = val;
            return dict;
        });
        if (this.props.chartProps.xAxis) {
            const rawX = jointData.unwrap(this.props.chartProps.xAxis.property);
            if (this.props.chartProps.xAxis.options && this.props.chartProps.xAxis.options.dither) {
                const dithered = jointData.unwrap(JointDataset.DitherLabel);
                plotlyProps.data[0].x = dithered.map((dither, index) => { return rawX[index] + dither;});
                if (jointData.metaDict[this.props.chartProps.xAxis.property].isCategorical) {
                    rawX.forEach((val, index) => {
                        customdata[index]["X"] = jointData.metaDict[this.props.chartProps.xAxis.property].sortedCategoricalValues[val];
                    });
                    hovertemplate += "x: %{customdata.X}";
                } else {
                    hovertemplate += "x: %{x}";
                }
            } else {
                plotlyProps.data[0].x = rawX;
                hovertemplate += "x: %{x}";
            }
        }
        if (this.props.chartProps.yAxis) {
            const rawY = jointData.unwrap(this.props.chartProps.yAxis.property);
            if (this.props.chartProps.yAxis.options && this.props.chartProps.yAxis.options.dither) {
                const dithered = jointData.unwrap(JointDataset.DitherLabel);
                plotlyProps.data[0].y = dithered.map((dither, index) => { return rawY[index] + dither;});
                if (jointData.metaDict[this.props.chartProps.yAxis.property].isCategorical) {
                    rawY.forEach((val, index) => {
                        customdata[index]["Y"] = jointData.metaDict[this.props.chartProps.yAxis.property].sortedCategoricalValues[val];
                    });
                    hovertemplate += "y: %{customdata.Y}";
                } else {
                    hovertemplate += "y: %{y}";
                }
            } else {
                plotlyProps.data[0].y = rawY;
                hovertemplate += "y: %{y}";
            }
        }
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
        const chartProps: IGenericChartProps = {
            chartType: 'scatter',
            xAxis: {
                property: JointDataset.IndexLabel,
            },
            yAxis: {
                property: JointDataset.DataLabelTemplate.replace("{0}", maxIndex.toString()),
                index: maxIndex
            },
            colorAxis: {
                property: exp.jointDataset.hasPredictedY ?
                    JointDataset.PredictedYLabel : JointDataset.IndexLabel
            }
        }
        this.props.onChange(chartProps, DataScatterId);
    }
}