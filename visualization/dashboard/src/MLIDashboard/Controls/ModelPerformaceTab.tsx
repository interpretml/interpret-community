import React from "react";
import { IGenericChartProps, ISelectorConfig, ChartTypes } from "../NewExplanationDashboard";
import { JointDataset } from "../JointDataset";
import { IExplanationModelMetadata } from "../IExplanationContext";
import { Cohort } from "../Cohort";
import { mergeStyleSets } from "@uifabric/styling";
import _ from "lodash";

export interface IModelPerformanceTabProps {
    chartProps: IGenericChartProps;
    theme?: string;
    jointDataset: JointDataset;
    metadata: IExplanationModelMetadata;
    cohorts: Cohort[];
    onChange: (props: IGenericChartProps) => void;
}

export class ModelPerformanceTab extends React.PureComponent<IModelPerformanceTabProps> {
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

    constructor(props: IModelPerformanceTabProps) {
        super(props);
        if (props.chartProps === undefined) {
            this.generateDefaultChartAxes();
        }
        this.onXSet = this.onXSet.bind(this);
        this.onYSet = this.onYSet.bind(this);

        this.state = {
            xDialogOpen: false,
            yDialogOpen: false,
            selectedCohortIndex: 0
        };
    }

    public render(): React.ReactNode {
        return (
            <div className={ModelPerformanceTab.classNames.tab}>

            </div>
        );
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
}