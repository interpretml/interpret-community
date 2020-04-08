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
import { InteractiveLegend, ILegendItem } from "./InteractiveLegend";

export interface IBarSeries {
    name: string;
    color: string;
    unsortedY: number[];
}

export interface IFeatureBarProps {
    theme: any;
    topK: number;
    unsortedX: string[];
    unsortedSeries: IBarSeries[];
    onSort?: (sortArray: number[]) => void;
    onSetStartingIndex?: (k: number) => void;
    onClick?: (plotlyData: any) => void;
}

export interface IFeatureBarState {
    sortingSeriesIndex: number;
    seriesIsActive: boolean[];
    startingK: number;
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

    private sortArray: number[];

    constructor(props: IFeatureBarProps) {
        super(props);
        this.state = {
            startingK: 0,
            plotlyProps: undefined,
            sortingSeriesIndex: 0,
            seriesIsActive: props.unsortedSeries.map(unused => true)
        };
        this.setSortingArray(this.props.unsortedSeries[this.state.sortingSeriesIndex].unsortedY);
        this.setStartingK = this.setStartingK.bind(this);
        this.setSortIndex = this.setSortIndex.bind(this);
    }

    public componentDidUpdate(prevProps: IFeatureBarProps) {
        if (this.props.unsortedSeries !== prevProps.unsortedSeries) {
            if(this.props.unsortedSeries.length === 0) {
                this.setState({plotlyProps: undefined});
                return;
            }
            this.setSortingArray(this.props.unsortedSeries[0].unsortedY);
            this.setState({plotlyProps: undefined, sortingSeriesIndex: 0});
        }
    }

    public render(): React.ReactNode {
        const minK = Math.min(4, this.props.unsortedX.length);
        const maxK = Math.min(30, this.props.unsortedX.length);
        const maxStartingK = Math.max(0, this.props.unsortedX.length - this.props.topK);
        const relayoutArg = {'xaxis.range': [this.state.startingK - 0.5, this.state.startingK + this.props.topK - 0.5]};
        const plotlyProps = this.state.plotlyProps;
        _.set(plotlyProps, 'layout.xaxis.range', [this.state.startingK - 0.5, this.state.startingK + this.props.topK - 0.5]);

        if (this.props.unsortedSeries.length === 0) {
            return (<div>No Data</div>)
        }
        if (this.state.plotlyProps === undefined) {
            this.loadProps();
            return <LoadingSpinner/>;
        };
        const items: ILegendItem[] = this.props.unsortedSeries.map((series, index) => {
            return {
                name: series.name,
                color: series.color,
                activated: this.state.seriesIsActive[index],
                onClick: () => {},
                onSort: () => {this.setSortIndex.bind(this, index)}
            }
        });
        return (<div className={FeatureImportanceBar.classNames.wrapper}>
            <div className={FeatureImportanceBar.classNames.chartAndControls}>
                <div className={FeatureImportanceBar.classNames.globalChartControls}>
                    {/* <Dropdown
                        label={localization.FeatureBar.sortBy}
                        selectedKey={this.state.sortingSeriesIndex}
                        onChange={this.setSortIndex}
                        styles={{ dropdown: { width: 150 } }}
                        options={nameOptions}
                    /> */}
                    <SpinButton
                        className={FeatureImportanceBar.classNames.topK}
                        styles={{
                            spinButtonWrapper: {maxWidth: "150px"},
                            labelWrapper: { alignSelf: "center"},
                            root: {
                                display: "inline-flex",
                                float: "right",
                                selectors: {
                                    "> div": {
                                        maxWidth: "160px"
                                    }
                                }
                            }
                        }}
                        label={localization.AggregateImportance.topKFeatures}
                        min={minK}
                        max={maxK}
                        value={this.props.topK.toString()}
                        onIncrement={this.setNumericValue.bind(this, 1, maxK, minK)}
                        onDecrement={this.setNumericValue.bind(this, -1, maxK, minK)}
                        onValidate={this.setNumericValue.bind(this, 0, maxK, minK)}
                    />
                    <Slider
                        className={FeatureImportanceBar.classNames.startingK}
                        ariaLabel={localization.AggregateImportance.topKFeatures}
                        max={maxStartingK}
                        min={0}
                        step={1}
                        value={this.state.startingK}
                        onChange={this.setStartingK}
                        showValue={true}
                    />
                </div>
                <div className={FeatureImportanceBar.classNames.globalChart}>
                    <AccessibleChart
                        plotlyProps={plotlyProps}
                        theme={this.props.theme}
                        relayoutArg={relayoutArg as any}
                        onClickHandler={this.props.onClick}
                    />
                </div>
            </div>
            <InteractiveLegend 
                items={items}
            />
        </div>);
    }

    // done outside of normal react state management to reduce recomputing
    private setSortingArray(unsortedY: number[]): void {
        this.sortArray = ModelExplanationUtils.getSortIndices(unsortedY).reverse();
        if (this.props.onSort) {
            this.props.onSort(this.sortArray);
        }
    }

    private loadProps(): void {
        setTimeout(() => {
            const props = this.buildBarPlotlyProps();
            this.setState({plotlyProps: props});
            }, 1);
    }

    private setSortIndex(newIndex: number): void {
        if (newIndex === this.state.sortingSeriesIndex) {
            return;
        }
        this.setSortingArray(this.props.unsortedSeries[newIndex].unsortedY);
        this.setState({sortingSeriesIndex: newIndex, plotlyProps: undefined});
    }

    private setStartingK(newValue: number): void {
        if (this.props.onSetStartingIndex) {
            this.props.onSetStartingIndex(newValue);
        }
        this.setState({startingK: newValue});
    }

    private readonly setNumericValue = (delta: number, max: number, min: number, stringVal: string): string | void => {
        if (delta === 0) {
            const number = +stringVal;
            if (!Number.isInteger(number)
                || number > max || number < min) {
                return this.state.startingK.toString();
            }
            this.setStartingK(number);
        } else {
            const prevVal = this.state.startingK;
            const newVal = prevVal + delta;
            if (newVal > max || newVal < min) {
                return prevVal.toString();
            }
            this.setStartingK(newVal);
        }
    }

    private buildBarPlotlyProps(): IPlotlyProperty {
        const sortedIndexVector = this.sortArray;
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
                showlegend: true
            } as any
        };

        const x = sortedIndexVector.map((unused, index) => index);

        this.state.seriesIsActive.forEach((isActive, index) => {
            if (!isActive) {
                return;
            }
            const series = this.props.unsortedSeries[index];
            baseSeries.data.push({
                orientation: 'v',
                type: 'bar',
                name: series.name,
                x,
                y: sortedIndexVector.map(index => series.unsortedY[index])
            } as any);
        });

        const ticktext = sortedIndexVector.map(i =>this.props.unsortedX[i]);
        const tickvals = sortedIndexVector.map((val, index) => index);

        _.set(baseSeries, 'layout.xaxis.ticktext', ticktext);
        _.set(baseSeries, 'layout.xaxis.tickvals', tickvals);
        return baseSeries;
    }
}