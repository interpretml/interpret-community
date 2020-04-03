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

export interface IFeatureBarProps {
    theme: any;
    topK: number;
    unsortedX: string[];
    unsortedYs: number[][];
    seriesNames: string[];
    onSort?: (sortArray: number[]) => void;
    onSetStartingIndex?: (k: number) => void;
    onClick?: (plotlyData: any) => void;
}

export interface IFeatureBarState {
    sortingSeriesIndex: number;
    startingK: number;
    plotlyProps: IPlotlyProperty;
}

export class FeatureImportanceBar extends React.PureComponent<IFeatureBarProps, IFeatureBarState> {
    private static readonly classNames = mergeStyleSets({
        wrapper: {

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
    private sortingY: number[];

    constructor(props: IFeatureBarProps) {
        super(props);
        this.state = {
            startingK: 0,
            plotlyProps: undefined,
            sortingSeriesIndex: 0
        };
        this.setSortingArray(this.props.unsortedYs[this.state.sortingSeriesIndex]);
        this.setStartingK = this.setStartingK.bind(this);
        this.setSortIndex = this.setSortIndex.bind(this);
    }

    public componentDidUpdate(prevProps: IFeatureBarProps) {
        if (this.props.unsortedYs !== prevProps.unsortedYs) {
            if(this.props.unsortedYs.length === 0) {
                this.setState({plotlyProps: undefined});
                return;
            }
            // if the passed in Y's don't match, try to find the new index of the prev selection
            // if not present, pick the first item.
            if (this.props.unsortedYs[this.state.sortingSeriesIndex] !== this.sortingY) {
                const newIndex = this.props.unsortedYs.findIndex(array => array === this.sortingY);
                if (newIndex !== -1) {
                    this.setState({plotlyProps: undefined, sortingSeriesIndex: newIndex});
                } else {
                    this.setSortingArray(this.props.unsortedYs[0]);
                    this.setState({plotlyProps: undefined, sortingSeriesIndex: 0});
                }
            } else {
                this.setState({plotlyProps: undefined});
            }
        }
    }

    public render(): React.ReactNode {
        const minK = Math.min(4, this.props.unsortedX.length);
        const maxK = Math.min(30, this.props.unsortedX.length);
        const maxStartingK = Math.max(0, this.props.unsortedX.length - this.props.topK);
        const relayoutArg = {'xaxis.range': [this.state.startingK - 0.5, this.state.startingK + this.props.topK - 0.5]};
        const plotlyProps = this.state.plotlyProps;
        _.set(plotlyProps, 'layout.xaxis.range', [this.state.startingK - 0.5, this.state.startingK + this.props.topK - 0.5]);
        const nameOptions: IDropdownOption[] = this.props.seriesNames.map((name, i) => {
            return {key: i, text: name};
        });
        if (this.props.unsortedYs.length === 0) {
            return (<div>No Data</div>)
        }
        if (this.state.plotlyProps === undefined) {
            this.loadProps();
            return <LoadingSpinner/>;
        };
        return (<div className={FeatureImportanceBar.classNames.wrapper}>
            <div className={FeatureImportanceBar.classNames.globalChartControls}>
                <Dropdown
                    label={localization.FeatureBar.sortBy}
                    selectedKey={this.state.sortingSeriesIndex}
                    onChange={this.setSortIndex}
                    styles={{ dropdown: { width: 150 } }}
                    options={nameOptions}
                />
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
        </div>);
    }

    // done outside of normal react state management to reduce recomputing
    private setSortingArray(unsortedY: number[]): void {
        this.sortingY = unsortedY;
        this.sortArray = ModelExplanationUtils.getSortIndices(this.sortingY).reverse();
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

    private setSortIndex(event: React.FormEvent<HTMLDivElement>, item: IDropdownOption): void {
        const newIndex = item.key as number;
        if (newIndex === this.state.sortingSeriesIndex) {
            return;
        }
        this.setSortingArray(this.props.unsortedYs[newIndex]);
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
        // y = sortedIndexVector.map(index => this.props.subsetAverageImportance[index]);

        this.props.unsortedYs.forEach((yVector, seriesIndex) => {
            baseSeries.data.push({
                orientation: 'v',
                type: 'bar',
                name: this.props.seriesNames[seriesIndex],
                x,
                y: sortedIndexVector.map(index => yVector[index])
            } as any);
        })

        const ticktext = sortedIndexVector.map(i =>this.props.unsortedX[i]);
        const tickvals = sortedIndexVector.map((val, index) => index);

        _.set(baseSeries, 'layout.xaxis.ticktext', ticktext);
        _.set(baseSeries, 'layout.xaxis.tickvals', tickvals);
        return baseSeries;
    }
}