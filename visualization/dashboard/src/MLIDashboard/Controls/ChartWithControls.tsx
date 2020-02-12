import React from "react";
import { IDropdownOption } from "office-ui-fabric-react/lib/Dropdown";
import { JointDataset, ColumnCategories } from "../JointDataset";
import { AccessibleChart, IPlotlyProperty, PlotlyMode } from "mlchartlib";
import { ComboBox, IComboBoxOption, IComboBox } from "office-ui-fabric-react/lib/ComboBox";
import { IconButton, DefaultButton } from "office-ui-fabric-react/lib/Button";
import { localization } from "../../Localization/localization";
import { FabricStyles } from "../FabricStyles";
import _ from "lodash";
import { Transform } from "plotly.js-dist";
import { AxisConfigDialog } from "./AxisConfigDialog";
import { Callout } from "office-ui-fabric-react/lib/Callout";

export enum ChartTypes {
    Scatter = 'scattergl',
    Bar = 'histogram',
    Box = 'box'
}

export interface IGenericChartProps {
    chartType: ChartTypes;
    xAxis?: ISelectorConfig;
    yAxis?: ISelectorConfig;
    colorAxis?: ISelectorConfig;
}

export interface ISelectorConfig {
    property: string;
    index?: number;
    options: {
        dither?: boolean;
        bin?: boolean;
    };
}

export interface IConfigurableChartProps {
    axisOptions: IDropdownOption[];
    jointDataset: JointDataset;
    chartProps: IGenericChartProps;
    onChange: (chartProps: IGenericChartProps) => void;
}

export interface IChartWithControlsState {
    xDialogOpen: boolean;
    yDialogOpen: boolean;
    colorDialogOpen: boolean;
}

export default class ChartWithControls extends React.PureComponent<IConfigurableChartProps, IChartWithControlsState> {
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

    private readonly _xButtonId = "x-button-id"

    constructor(props: IConfigurableChartProps) {
        super(props);
        this.onColorSelected = this.onColorSelected.bind(this);
        this.onDitherXToggle = this.onDitherXToggle.bind(this);
        this.onDitherYToggle = this.onDitherYToggle.bind(this);
        this.onXSelected = this.onXSelected.bind(this);
        this.onYSelected = this.onYSelected.bind(this);
        this.onXSet = this.onXSet.bind(this);

        this.state = {
            xDialogOpen: false,
            yDialogOpen: false,
            colorDialogOpen: false
        };
    }

    public render(): React.ReactNode {
        const plotlyProps = this.buildPlotlyProps();
        return (
        <div className="explanation-chart">
                <div className="top-controls">
                    <div className="path-selector x-value">
                        <DefaultButton 
                            onClick={this.setXOpen.bind(this, true)}
                            id={this._xButtonId}
                            text={localization.ExplanationScatter.xValue}
                        />
                        {(this.state.xDialogOpen) && (
                            <AxisConfigDialog 
                                jointDataset={this.props.jointDataset}
                                orderedGroupTitles={[ColumnCategories.outcome, ColumnCategories.dataset]}
                                selectedColumn={this.props.chartProps.xAxis}
                                canBin={this.props.chartProps.chartType === ChartTypes.Bar || this.props.chartProps.chartType === ChartTypes.Box}
                                canDither={this.props.chartProps.chartType === ChartTypes.Scatter}
                                onAccept={this.onXSet}
                                onCancel={this.setXOpen.bind(this, false)}
                                target={this._xButtonId}
                            />
                        )}
                    </div>
                    <div className="path-selector">
                        <ComboBox
                            options={this.props.axisOptions}
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
                            options={this.props.axisOptions}
                            onChange={this.onYSelected}
                            label={localization.ExplanationScatter.yValue}
                            ariaLabel="y picker"
                            selectedKey={this.props.chartProps.yAxis.property}
                            useComboBoxAsMenuWidth={true}
                            styles={FabricStyles.defaultDropdownStyle}
                        />
                        {(this.props.jointDataset.metaDict[this.props.chartProps.yAxis.property].isCategorical ||
                            (this.props.jointDataset.metaDict[this.props.chartProps.yAxis.property].featureRange &&
                            this.props.jointDataset.metaDict[this.props.chartProps.yAxis.property].featureRange.rangeType)) && (
                            <IconButton
                                iconProps={{ iconName: "Info" }}
                                title={localization.CrossClass.info}
                                ariaLabel="Info"
                                onClick={this.onDitherYToggle}
                                styles={{ root: { marginBottom: -3, color: "rgb(0, 120, 212)" } }}
                            />
                        )}
                    </div>
                </div>
                <AccessibleChart
                    plotlyProps={plotlyProps}
                    sharedSelectionContext={undefined}
                    theme={undefined}
                    onSelection={undefined}
                />
        </div>);
    }

    private readonly setXOpen = (val: boolean): void => {
        if (val && this.state.xDialogOpen === false) {
            this.setState({xDialogOpen: true});
            return;
        }
        this.setState({xDialogOpen: false});
    }

    private onDitherXToggle(): void {
        const newProps = _.cloneDeep(this.props.chartProps);
        const initialValue = _.get(newProps.xAxis, "options.dither", false);
        _.set(newProps.xAxis, "options.dither", !initialValue);
        this.props.onChange(newProps);
    }

    private onDitherYToggle(): void {
        const newProps = _.cloneDeep(this.props.chartProps);
        const initialValue = _.get(newProps.yAxis, "options.dither", false);
        _.set(newProps.yAxis, "options.dither", !initialValue);
        this.props.onChange(newProps);
    }

    private onXSelected(event: React.FormEvent<IComboBox>, item: IComboBoxOption): void {
        const newProps = _.cloneDeep(this.props.chartProps);
        newProps.xAxis.property = item.key as string;
        if (this.props.jointDataset.metaDict[item.key].isCategorical) {
            newProps.xAxis.options = {dither: true, bin: false};
        }
        this.props.onChange(newProps);
    }

    private onXSet(value: ISelectorConfig): void {
        const newProps = _.cloneDeep(this.props.chartProps);
        newProps.xAxis = value
        this.props.onChange(newProps);
        this.setState({xDialogOpen: false})
    }

    private onYSelected(event: React.FormEvent<IComboBox>, item: IComboBoxOption): void {
        const newProps = _.cloneDeep(this.props.chartProps);
        newProps.yAxis.property = item.key as string;
        if (this.props.jointDataset.metaDict[item.key].isCategorical) {
            newProps.yAxis.options = {dither: true, bin: false};
        }
        this.props.onChange(newProps);
    }

    private onColorSelected(event: React.FormEvent<IComboBox>, item: IComboBoxOption): void {
        const newProps = _.cloneDeep(this.props.chartProps);
        newProps.colorAxis.property = item.key as string;
        this.props.onChange(newProps);
    }

    private buildPlotlyProps(): IPlotlyProperty {
        const plotlyProps = _.cloneDeep(ChartWithControls.basePlotlyProperties);
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
                        _.set(plotlyProps, 'layout.xaxis.ticktext', xLabels);
                        _.set(plotlyProps, 'layout.xaxis.tickvals', xLabelIndexes);
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
                        _.set(plotlyProps, 'layout.yaxis.ticktext', yLabels);
                        _.set(plotlyProps, 'layout.yaxis.tickvals', yLabelIndexes);
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
                _.set(plotlyProps, 'layout.xaxis.ticktext', xLabels);
                _.set(plotlyProps, 'layout.xaxis.tickvals', xLabelIndexes);
                hovertemplate += "x: %{text}<br>";
                hovertemplate += "count: %{y}<br>";
                const transforms: Partial<Transform>[] = [
                    {
                        type: 'aggregate',
                        groups: rawX,
                        aggregations: [
                          {target: 'y', func: 'sum'},
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
}