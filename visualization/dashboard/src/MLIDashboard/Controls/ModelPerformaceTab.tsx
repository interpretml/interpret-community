import React from "react";
import { IGenericChartProps, ISelectorConfig, ChartTypes } from "../NewExplanationDashboard";
import { JointDataset, ColumnCategories } from "../JointDataset";
import { IExplanationModelMetadata } from "../IExplanationContext";
import { Cohort } from "../Cohort";
import { mergeStyleSets } from "@uifabric/styling";
import _ from "lodash";
import { DefaultButton } from "office-ui-fabric-react/lib/Button";
import { localization } from "../../Localization/localization";
import { AxisConfigDialog } from "./AxisConfigDialog";
import { AccessibleChart, IPlotlyProperty } from "mlchartlib";
import { Transform } from "plotly.js-dist";

export interface IModelPerformanceTabProps {
    chartProps: IGenericChartProps;
    theme?: string;
    jointDataset: JointDataset;
    metadata: IExplanationModelMetadata;
    cohorts: Cohort[];
    onChange: (props: IGenericChartProps) => void;
}

export interface IModelPerformanceTabState {
    xDialogOpen: boolean;
    yDialogOpen: boolean;
    selectedCohortIndex: number;
}

export class ModelPerformanceTab extends React.PureComponent<IModelPerformanceTabProps, IModelPerformanceTabState> {
    private static readonly classNames = mergeStyleSets({
        tab: {
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

    private readonly _xButtonId = "x-button-id";
    private readonly _yButtonId = "y-button-id";

    constructor(props: IModelPerformanceTabProps) {
        super(props);
        if (props.chartProps === undefined) {
            this.generateDefaultChartAxes();
        }
        this.onXSet = this.onXSet.bind(this);
        this.onYSet = this.onYSet.bind(this);
        this.setXOpen = this.setXOpen.bind(this);
        this.setYOpen = this.setYOpen.bind(this);

        this.state = {
            xDialogOpen: false,
            yDialogOpen: false,
            selectedCohortIndex: 0
        };
    }

    public render(): React.ReactNode {
        if (this.props.chartProps === undefined) {
            return (<div/>);
        }
        const plotlyProps = ModelPerformanceTab.generatePlotlyProps(
            this.props.jointDataset,
            this.props.chartProps,
            this.props.cohorts,
            this.state.selectedCohortIndex
        );
        return (
            <div className={ModelPerformanceTab.classNames.tab}>
                <div className={ModelPerformanceTab.classNames.chartWithAxes}>
                    <div className={ModelPerformanceTab.classNames.chartWithVertical}>
                        <div className={ModelPerformanceTab.classNames.verticalAxis}>
                            <div className={ModelPerformanceTab.classNames.rotatedVerticalBox}>
                                {(this.props.chartProps.chartType !== ChartTypes.Bar) && (
                                    <DefaultButton 
                                        onClick={this.setYOpen.bind(this, true)}
                                        id={this._yButtonId}
                                        text={localization.ExplanationScatter.yValue + this.props.jointDataset.metaDict[this.props.chartProps.yAxis.property].abbridgedLabel}
                                        title={localization.ExplanationScatter.yValue + this.props.jointDataset.metaDict[this.props.chartProps.yAxis.property].label}
                                    />
                                )}
                                {(this.props.chartProps.chartType === ChartTypes.Bar) && (
                                    <div>{localization.ExplanationScatter.count}</div>
                                )}
                                {(this.state.yDialogOpen) && (
                                    <AxisConfigDialog 
                                        jointDataset={this.props.jointDataset}
                                        orderedGroupTitles={[ColumnCategories.outcome]}
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
                    <div className={ModelPerformanceTab.classNames.horizontalAxisWithPadding}>
                        <div className={ModelPerformanceTab.classNames.paddingDiv}></div>
                        <div className={ModelPerformanceTab.classNames.horizontalAxis}>
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
            </div>
        );
    } 

    private readonly setXOpen = (val: boolean): void => {
        if (val && this.state.xDialogOpen === false) {
            this.setState({xDialogOpen: true});
            return;
        }
        this.setState({xDialogOpen: false});
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

    private generateDefaultChartAxes(): void {
        const chartProps: IGenericChartProps = {
            chartType: ChartTypes.Box,
            xAxis: {
                property: Cohort.CohortKey,
                options: {}
            },
            yAxis: {
                property: JointDataset.PredictedYLabel,
                options: {
                    bin: false
                }
            }
        }
        this.props.onChange(chartProps);
    }

    private static generatePlotlyProps(jointData: JointDataset, chartProps: IGenericChartProps, cohorts: Cohort[], selectedCohortIndex: number): IPlotlyProperty {
        // In this view, x will always be categorical (including a binned numberic variable), and could be
        // iteratinos over the cohorts. We can set x and the x labels before the rest of the char properties.
        const plotlyProps: IPlotlyProperty = {
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
        let rawX: number[];
        let rawY: number[];
        let xLabels: string[];
        let xLabelIndexes: number[];
        if (chartProps.xAxis.property === Cohort.CohortKey) {
            rawX = [];
            rawY = [];
            xLabels = [];
            xLabelIndexes = [];
            cohorts.forEach((cohort, cohortIndex) => {
                const cohortYs = cohort.unwrap(chartProps.yAxis.property, chartProps.chartType === ChartTypes.Bar);
                const cohortX = new Array(cohortYs.length).fill(cohortIndex);
                rawY.push(...cohortYs);
                rawX.push(...cohortX);
                xLabels.push(cohort.name);
                xLabelIndexes.push(cohortIndex);
            });
        } else {
            const cohort = cohorts[selectedCohortIndex];
            rawX = cohort.unwrap(chartProps.xAxis.property, true);
            rawY = cohort.unwrap(chartProps.yAxis.property, chartProps.chartType === ChartTypes.Bar);
            xLabels = jointData.metaDict[chartProps.xAxis.property].sortedCategoricalValues;
            const xLabelIndexes = xLabels.map((unused, index) => index);
        }
        plotlyProps.data[0].hoverinfo = "all";
        switch (chartProps.chartType) {
            case ChartTypes.Box: {
                plotlyProps.data[0].type = "box" as any;
                plotlyProps.data[0].x = rawX;
                plotlyProps.data[0].y = rawY;
                _.set(plotlyProps, "layout.xaxis.ticktext", xLabels);
                _.set(plotlyProps, "layout.xaxis.tickvals", xLabelIndexes);
                break;
            }
            case ChartTypes.Bar: {
                // for now, treat all bar charts as histograms, the issue with plotly implemented histogram is
                // it tries to bin the data passed to it(we'd like to apply the user specified bins.)
                // We also use the selected Y property as the series prop, since all histograms will just be a count.
                plotlyProps.data[0].type = "bar";
                const y = new Array(rawX.length).fill(1);
                plotlyProps.data[0].text = rawX.map(index => xLabels[index]);
                plotlyProps.data[0].x = rawX;
                plotlyProps.data[0].y = y;
                _.set(plotlyProps, "layout.xaxis.ticktext", xLabels);
                _.set(plotlyProps, "layout.xaxis.tickvals", xLabelIndexes);
                const styles = jointData.metaDict[chartProps.colorAxis.property].sortedCategoricalValues.map((label, index) => {
                    return {
                        target: index,
                        value: { name: label}
                    };
                });
                const transforms: Partial<Transform>[] = [
                    {
                        type: "aggregate",
                        groups: rawX,
                        aggregations: [
                          {target: "y", func: "sum"},
                        ]
                    },
                    {
                        type: "groupby",
                        groups: rawY,
                        styles
                    }
                ];
                plotlyProps.layout.showlegend = true;
                plotlyProps.data[0].transforms = transforms;
                break;
            }
        }
        return plotlyProps;
    };
}