import { ComboBox, IComboBox, IComboBoxOption } from "office-ui-fabric-react/lib/ComboBox";
import React from "react";
import { AccessibleChart, IPlotlyProperty, DefaultSelectionFunctions } from "mlchartlib";
import { localization } from "../../../Localization/localization";
import { FabricStyles } from "../../FabricStyles";
import {  ScatterUtils, INewScatterProps, IGenericChartProps } from "./ScatterUtils";
import _ from "lodash";
import { NoDataMessage, LoadingSpinner } from "../../SharedComponents";
import { mergeStyleSets } from "@uifabric/styling";

export const DataScatterId = 'data_scatter_id';

export class DataExploration extends React.PureComponent<INewScatterProps> {

    constructor(props: INewScatterProps) {
        super(props);
        this.onXSelected = this.onXSelected.bind(this);
        this.onYSelected = this.onYSelected.bind(this);
        this.onColorSelected = this.onColorSelected.bind(this);
        this.onDismiss = this.onDismiss.bind(this);
        if (props.chartProps === undefined) {
            this.generateDefaultChartAxes();
        }
    }

    public render(): React.ReactNode {
        const plotlyProps = this.generatePlotlyProps();
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
                property: 'Index',
            },
            yAxis: {
                property: 'Data',
                index: maxIndex
            },
            colorAxis: {
                property: exp.testDataset.predictedY !== undefined ?
                    'PredictedY' : 'Index'
            }
        }
        this.props.onChange(chartProps, DataScatterId);
    }

    private generatePlotlyProps(): IPlotlyProperty {
        const plotlyProps: IPlotlyProperty = _.cloneDeep(ScatterUtils.baseScatterProperties);
        
    }
}