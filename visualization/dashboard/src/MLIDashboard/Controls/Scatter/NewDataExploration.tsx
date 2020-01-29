import { ComboBox, IComboBox, IComboBoxOption } from "office-ui-fabric-react/lib/ComboBox";
import * as Plotly from 'plotly.js-dist';
import React from "react";
import { AccessibleChart, IPlotlyProperty, DefaultSelectionFunctions } from "mlchartlib";
import { localization } from "../../../Localization/localization";
import { FabricStyles } from "../../FabricStyles";
import {  ScatterUtils, INewScatterProps, IGenericChartProps } from "./ScatterUtils";
import _ from "lodash";
import { NoDataMessage, LoadingSpinner } from "../../SharedComponents";
import { mergeStyleSets } from "@uifabric/styling";
import { JointDataset } from "../../JointDataset";
import { IDropdownOption } from "office-ui-fabric-react/lib/Dropdown";
import { IconButton } from "office-ui-fabric-react/lib/Button";

export const DataScatterId = 'data_scatter_id';

export class NewDataExploration extends React.PureComponent<INewScatterProps> {

    constructor(props: INewScatterProps) {
        super(props);
        // if (props.chartProps === undefined) {
        //     this.generateDefaultChartAxes();
        // }
        this.onXSelected = this.onXSelected.bind(this);
        this.onYSelected = this.onYSelected.bind(this);
        this.onColorSelected = this.onColorSelected.bind(this);
        this.onDitherXToggle = this.onDitherXToggle.bind(this);
        this.onDitherYToggle = this.onDitherYToggle.bind(this);
        this.scatterSelection = this.scatterSelection.bind(this);
    }

    public render(): React.ReactNode {
        if (this.props.chartProps === undefined) {
            this.generateDefaultChartAxes();
            return (<div/>);
        }
        const plotlyProps = this.generatePlotlyProps();
        const dropdownOptions = this.generateDropdownOptions();
        const jointData = this.props.dashboardContext.explanationContext.jointDataset;
        return (
            <div className="explanation-chart">
                <FilterControl
                    filterContext={this.props.filterContext}
                    dashboardContext={this.props.dashboardContext}
                />
                <div className="top-controls">
                    <div className="path-selector x-value">
                        <ComboBox
                            options={dropdownOptions}
                            onChange={this.onXSelected}
                            label={localization.ExplanationScatter.xValue}
                            ariaLabel="x picker"
                            selectedKey={this.props.chartProps.xAxis.property}
                            useComboBoxAsMenuWidth={true}
                            styles={ScatterUtils.xStyle}
                        />
                        {(jointData.metaDict[this.props.chartProps.xAxis.property].isCategorical ||
                            (jointData.metaDict[this.props.chartProps.xAxis.property].featureRange &&
                            jointData.metaDict[this.props.chartProps.xAxis.property].featureRange.rangeType)) && (
                            <IconButton
                                iconProps={{ iconName: 'Info' }}
                                title={localization.CrossClass.info}
                                ariaLabel="Info"
                                onClick={this.onDitherXToggle}
                                styles={{ root: { marginBottom: -3, color: 'rgb(0, 120, 212)' } }}
                            />
                        )}
                    </div>
                    <div className="path-selector">
                        <ComboBox
                            options={dropdownOptions}
                            onChange={this.onColorSelected}
                            label={localization.ExplanationScatter.colorValue}
                            ariaLabel="color picker"
                            selectedKey={this.props.chartProps.colorAxis.property}
                            useComboBoxAsMenuWidth={true}
                            styles={FabricStyles.defaultDropdownStyle}
                        />
                    </div>
                </div>
                <div className="top-controls">
                    <div className="path-selector y-value">
                        <ComboBox
                            options={dropdownOptions}
                            onChange={this.onYSelected}
                            label={localization.ExplanationScatter.yValue}
                            ariaLabel="y picker"
                            selectedKey={this.props.chartProps.yAxis.property}
                            useComboBoxAsMenuWidth={true}
                            styles={ScatterUtils.yStyle}
                        />
                        {(jointData.metaDict[this.props.chartProps.yAxis.property].isCategorical ||
                            (jointData.metaDict[this.props.chartProps.yAxis.property].featureRange &&
                            jointData.metaDict[this.props.chartProps.yAxis.property].featureRange.rangeType)) && (
                            <IconButton
                                iconProps={{ iconName: 'Info' }}
                                title={localization.CrossClass.info}
                                ariaLabel="Info"
                                onClick={this.onDitherYToggle}
                                styles={{ root: { marginBottom: -3, color: 'rgb(0, 120, 212)' } }}
                            />
                        )}
                    </div>
                </div>
                <AccessibleChart
                    plotlyProps={plotlyProps}
                    sharedSelectionContext={this.props.selectionContext}
                    theme={this.props.theme}
                    onSelection={this.scatterSelection}
                />
        </div>);
    }

    private onDitherXToggle(): void {
        const newProps = _.cloneDeep(this.props.chartProps);
        const initialValue = _.get(newProps.xAxis, 'options.dither', false);
        _.set(newProps.xAxis, 'options.dither', !initialValue);
        this.props.onChange(newProps, DataScatterId);
    }

    private onDitherYToggle(): void {
        const newProps = _.cloneDeep(this.props.chartProps);
        const initialValue = _.get(newProps.yAxis, 'options.dither', false);
        _.set(newProps.yAxis, 'options.dither', !initialValue);
        this.props.onChange(newProps, DataScatterId);
    }

    private onXSelected(event: React.FormEvent<IComboBox>, item: IComboBoxOption): void {
        const newProps = _.cloneDeep(this.props.chartProps);
        newProps.xAxis.property = item.key as string;
        if (this.props.dashboardContext.explanationContext.jointDataset.metaDict[item.key].isCategorical) {
            newProps.xAxis.options = {dither: true, binOptions: undefined};
        }
        this.props.onChange(newProps, DataScatterId);
    }

    private onYSelected(event: React.FormEvent<IComboBox>, item: IComboBoxOption): void {
        const newProps = _.cloneDeep(this.props.chartProps);
        newProps.yAxis.property = item.key as string;
        if (this.props.dashboardContext.explanationContext.jointDataset.metaDict[item.key].isCategorical) {
            newProps.yAxis.options = {dither: true, binOptions: undefined};
        }
        this.props.onChange(newProps, DataScatterId);
    }

    private onColorSelected(event: React.FormEvent<IComboBox>, item: IComboBoxOption): void {
        const newProps = _.cloneDeep(this.props.chartProps);
        newProps.colorAxis.property = item.key as string;
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
        plotlyProps.data[0].datapointLevelAccessors = undefined;
        plotlyProps.data[0].hoverinfo = 'all';
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
        if (this.props.chartProps.yAxis) {
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
            const rawColor = jointData.unwrap(this.props.chartProps.colorAxis.property);
            // handle binning to categories later
            if (jointData.metaDict[this.props.chartProps.colorAxis.property].isCategorical) {
                const styles = jointData.metaDict[this.props.chartProps.colorAxis.property].sortedCategoricalValues.map((label, index) => {
                    return {
                        target: index,
                        value: { name: label}
                    };
                });
                plotlyProps.data[0].transforms = [{
                    type: 'groupby',
                    groups: rawColor,
                    styles
                }];
                plotlyProps.layout.showlegend = true;
            } else {
                plotlyProps.data[0].marker = {
                    color: rawColor,
                    colorbar: {
                        title: {
                            side: 'right',
                            text: 'placeholder'
                        } as any
                    },
                    colorscale: 'Bluered'
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
        const yKey = JointDataset.DataLabelTemplate.replace("{0}", maxIndex.toString());
        const yIsDithered = exp.jointDataset.metaDict[yKey].isCategorical;
        const chartProps: IGenericChartProps = {
            chartType: 'scatter',
            xAxis: {
                property: JointDataset.IndexLabel,
            },
            yAxis: {
                property: yKey,
                options: {
                    dither: yIsDithered,
                    binOptions: undefined
                }
            },
            colorAxis: {
                property: exp.jointDataset.hasPredictedY ?
                    JointDataset.PredictedYLabel : JointDataset.IndexLabel
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
        Plotly.restyle(guid, 'selectedpoints' as any, selectedPoints as any);
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
        Plotly.restyle(guid, 'marker.line.width' as any, newLineWidths as any);
    }
}