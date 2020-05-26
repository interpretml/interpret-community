import React from "react";
import _ from "lodash";
import { localization } from "../../../Localization/localization";
import { mergeStyleSets, getTheme } from "@uifabric/styling";
import { IPlotlyProperty, AccessibleChart } from "mlchartlib";
import { LoadingSpinner } from "../../SharedComponents";
import { FabricStyles } from "../../FabricStyles";
import { JointDataset } from "../../JointDataset";
import { IGlobalSeries } from "../GlobalExplanationTab/IGlobalSeries";
import { featureImportanceBarStyles } from "../FeatureImportanceBar/FeatureImportanceBar.styles";
import { Text } from "office-ui-fabric-react";

export interface IFeatureBoxProps {
    jointDataset: JointDataset;
    yAxisLabels: string[];
    sortArray: number[];
    selectedFeatureIndex?: number;
    selectedSeriesIndex?: number;
    topK: number;
    startingK: number;
    unsortedX: string[];
    unsortedSeries: IGlobalSeries[];
    onFeatureSelection?: (seriesIndex: number, featureIndex: number) => void;
}

export interface IFeatureBarState {
    plotlyProps: IPlotlyProperty;
}

export class FeatureImportanceBox extends React.PureComponent<IFeatureBoxProps, IFeatureBarState> {

    constructor(props: IFeatureBoxProps) {
        super(props);
        this.state = {
            plotlyProps: undefined
        };
        this.selectPointFromChart = this.selectPointFromChart.bind(this )
    }

    public componentDidUpdate(prevProps: IFeatureBoxProps) {
        if (this.props.unsortedSeries !== prevProps.unsortedSeries || this.props.sortArray !== prevProps.sortArray) {
            this.setState({plotlyProps: undefined});
        }
    }

    public render(): React.ReactNode {
        const classNames = featureImportanceBarStyles();
        const relayoutArg = {'xaxis.range': [this.props.startingK - 0.5, this.props.startingK + this.props.topK - 0.5]};
        const plotlyProps = this.state.plotlyProps;
        _.set(plotlyProps, 'layout.xaxis.range', [this.props.startingK - 0.5, this.props.startingK + this.props.topK - 0.5]);

        if (!this.props.unsortedSeries ||this.props.unsortedSeries.length === 0 || !this.props.sortArray || this.props.sortArray.length === 0) {
            return (<div className={classNames.noData}>
                <Text variant={"xxLarge"}>No data</Text>
            </div>)
        }
        if (this.state.plotlyProps === undefined) {
            this.loadProps();
            return <LoadingSpinner/>;
        };
        return (<div className={classNames.chartWithVertical}>
                        <div className={classNames.verticalAxis}>
                            <div className={classNames.rotatedVerticalBox}>
                                <div>{this.props.yAxisLabels.map(label => 
                                    <Text block variant="medium" className={classNames.boldText}>{label}</Text>
                                    )}
                                </div>
                            </div>
                        </div>
            <AccessibleChart
                plotlyProps={plotlyProps}
                theme={getTheme() as any}
                relayoutArg={relayoutArg as any}
                onClickHandler={this.selectPointFromChart}
            />
        </div>);
    }

    private loadProps(): void {
        setTimeout(() => {
            const props = this.buildBoxPlotlyProps();
            this.setState({plotlyProps: props});
            }, 1);
    }

    private buildBoxPlotlyProps(): IPlotlyProperty {
        const sortedIndexVector = this.props.sortArray;
        const baseSeries = {
            config: { displaylogo: false, responsive: true, displayModeBar: false } as Plotly.Config,
            data: [],
            layout: {
                autosize: true,
                dragmode: false,
                margin: {t: 10, r: 10, b: 30, l: 0},
                hovermode: false,
                boxmode: 'group',
                xaxis: {
                    automargin: true,
                    color: FabricStyles.chartAxisColor,
                    tickfont: {
                        family: "Roboto, Helvetica Neue, sans-serif",
                        size: 11,
                        color: FabricStyles.chartAxisColor,
                    },
                    showgrid: false
                },
                yaxis: {
                    automargin: true,
                    color: FabricStyles.chartAxisColor,
                    tickfont: {
                        family: "Roboto, Helvetica Neue, sans-serif",
                        size: 11,
                        color: FabricStyles.chartAxisColor,
                    },
                    zeroline: true,
                    showgrid: true,
                    gridcolor: "#e5e5e5"
                },
                showlegend: false
            } as any
        };

        const x = sortedIndexVector.map((unused, index) => index);

        this.props.unsortedSeries.forEach((series, seriesIndex) => {
            baseSeries.data.push({
                type: 'box',
                name: series.name,
                x: sortedIndexVector.map((sortIndex, xIndex) => series.unsortedIndividualY[sortIndex].map(unused => xIndex)).reduce((prev, curr) => {
                    prev.push(...curr);
                    return prev;
                }, []),
                y: sortedIndexVector.map(index => series.unsortedIndividualY[index]).reduce((prev, curr) => {
                    prev.push(...curr);
                    return prev;
                }, [])
                // marker: {
                //     color: sortedIndexVector.map(index => (index === this.props.selectedFeatureIndex && seriesIndex === this.props.selectedSeriesIndex) ?
                //         FabricStyles.fabricColorPalette[series.colorIndex] : FabricStyles.fabricColorPalette[series.colorIndex])
                // }
            } as any);
            
        });

        const ticktext = sortedIndexVector.map(i =>this.props.unsortedX[i]);
        const tickvals = sortedIndexVector.map((val, index) => index);

        _.set(baseSeries, 'layout.xaxis.ticktext', ticktext);
        _.set(baseSeries, 'layout.xaxis.tickvals', tickvals);
        return baseSeries;
    }

    private selectPointFromChart(data: any): void {
        const trace = data.points[0];
        const featureNumber = this.props.sortArray[trace.x];
        this.props.onFeatureSelection(trace.curveNumber, featureNumber)
    }  
}