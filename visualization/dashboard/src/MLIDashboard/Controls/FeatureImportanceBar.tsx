import React from "react";
import _ from "lodash";
import { localization } from "../../Localization/localization";
import { mergeStyleSets } from "@uifabric/styling";
import { ModelExplanationUtils } from "../ModelExplanationUtils";
import { IPlotlyProperty, AccessibleChart } from "mlchartlib";
import { SpinButton } from "office-ui-fabric-react/lib/SpinButton";
import { Slider } from "office-ui-fabric-react/lib/Slider";
import { LoadingSpinner } from "../SharedComponents";
import { isThisExpression } from "@babel/types";
import { Dropdown, IDropdownOption } from "office-ui-fabric-react/lib/Dropdown";
import { InteractiveLegend, ILegendItem, SortingState } from "./InteractiveLegend";
import { FabricStyles } from "../FabricStyles";
import { JointDataset } from "../JointDataset";
import { IGlobalSeries } from "./GlobalExplanationTab/IGlobalSeries";

export interface IFeatureBarProps {
    theme: any;
    jointDataset: JointDataset;
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

export class FeatureImportanceBar extends React.PureComponent<IFeatureBarProps, IFeatureBarState> {
    private static readonly classNames = mergeStyleSets({
        wrapper: {
            display:"flex",
            flexDirection: "row",
            width: "100%"
        },
        chartAndControls: {
            flex: "1"
        },
        globalChartControls: {
            display: "flex",
            flexDirection: "row"
        },
        topK: {
            maxWidth: "200px"
        },
        startingK: {
            flex: 1
        },
        globalChart: {
            height: "400px",
            width: "100%"
        }
    });

    constructor(props: IFeatureBarProps) {
        super(props);
        this.state = {
            plotlyProps: undefined
        };
        this.selectPointFromChart = this.selectPointFromChart.bind(this )
    }

    public componentDidUpdate(prevProps: IFeatureBarProps) {
        if (this.props.unsortedSeries !== prevProps.unsortedSeries || this.props.sortArray !== prevProps.sortArray) {
            this.setState({plotlyProps: undefined});
        }
    }

    public render(): React.ReactNode {
        const relayoutArg = {'xaxis.range': [this.props.startingK - 0.5, this.props.startingK + this.props.topK - 0.5]};
        const plotlyProps = this.state.plotlyProps;
        _.set(plotlyProps, 'layout.xaxis.range', [this.props.startingK - 0.5, this.props.startingK + this.props.topK - 0.5]);

        if (!this.props.unsortedSeries ||this.props.unsortedSeries.length === 0 || !this.props.sortArray || this.props.sortArray.length === 0) {
            return (<div>No Data</div>)
        }
        if (this.state.plotlyProps === undefined) {
            this.loadProps();
            return <LoadingSpinner/>;
        };
        return (<div className={FeatureImportanceBar.classNames.globalChart}>
            <AccessibleChart
                plotlyProps={plotlyProps}
                theme={this.props.theme}
                relayoutArg={relayoutArg as any}
                onClickHandler={this.selectPointFromChart}
            />
        </div>);
    }

    private loadProps(): void {
        setTimeout(() => {
            const props = this.buildBarPlotlyProps();
            this.setState({plotlyProps: props});
            }, 1);
    }

    private buildBarPlotlyProps(): IPlotlyProperty {
        const sortedIndexVector = this.props.sortArray;
        const baseSeries = {
            config: { displaylogo: false, responsive: true, displayModeBar: false } as Plotly.Config,
            data: [],
            layout: {
                autosize: true,
                dragmode: false,
                barmode: 'group',
                font: {
                    size: 10
                },
                margin: {t: 10, r: 10, b: 30},
                hovermode: 'closest',
                xaxis: {
                    automargin: true
                },
                yaxis: {
                    automargin: true,
                    title: localization.featureImportance
                },
                showlegend: false
            } as any
        };

        const x = sortedIndexVector.map((unused, index) => index);

        this.props.unsortedSeries.forEach((series, seriesIndex) => {
            baseSeries.data.push({
                orientation: 'v',
                type: 'bar',
                name: series.name,
                x,
                y: sortedIndexVector.map(index => series.unsortedAggregateY[index]),
                marker: {
                    color: sortedIndexVector.map(index => (index === this.props.selectedFeatureIndex && seriesIndex === this.props.selectedSeriesIndex) ?
                        FabricStyles.plotlyColorHexPalette[series.index] : FabricStyles.plotlyColorHexPalette[series.index])
                }
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