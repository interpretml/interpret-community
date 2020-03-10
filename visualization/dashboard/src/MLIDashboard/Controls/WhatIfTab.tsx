import React from "react";
import { JointDataset, ColumnCategories } from "../JointDataset";
import { IExplanationModelMetadata } from "../IExplanationContext";
import { mergeStyleSets } from "@uifabric/styling";
import { IPlotlyProperty, AccessibleChart } from "mlchartlib";
import { Panel, PanelType, IPanelProps } from "office-ui-fabric-react/lib/Panel";
import { localization } from "../../Localization/localization";
import { IRenderFunction } from "@uifabric/utilities";
import { IconButton, DefaultButton } from "office-ui-fabric-react/lib/Button";
import { IconNames } from "@uifabric/icons";
import { FilterControl } from "./FilterControl";
import { IFilterContext } from "../Interfaces/IFilter";
import { ChartTypes, IGenericChartProps, ISelectorConfig } from "../NewExplanationDashboard";
import { AxisConfigDialog } from "./AxisConfigDialog";
import _ from "lodash";

export interface IWhatIfTabProps {
    theme: any;
    jointDataset: JointDataset;
    metadata: IExplanationModelMetadata;
    filterContext: IFilterContext;
    chartProps: IGenericChartProps;
    onChange: (config: IGenericChartProps) => void; 
}

export interface IWhatIfTabState {
    isPanelOpen: boolean;
    xDialogOpen: boolean;
    yDialogOpen: boolean;
}

export class WhatIfTab extends React.PureComponent<IWhatIfTabProps, IWhatIfTabState> {
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
    
    private static readonly classNames = mergeStyleSets({
        dataTab: {
            display: "flex",
            flexDirection: "row",
            height: "100%"
        },
        expandedPanel: {
            width: "250px",
            height: "100%",
            borderRight: "1px solid black"
        },
        collapsedPanel: {
            width: "40px",
            height: "100%",
            borderRight: "1px solid black"
        },
        mainArea: {
            flex: 1
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

    constructor(props: IWhatIfTabProps) {
        super(props);
        this.state = {
            isPanelOpen: false,
            xDialogOpen: false,
            yDialogOpen: false
        };
        this.dismissPanel = this.dismissPanel.bind(this);
        this.openPanel = this.openPanel.bind(this);
    }

    public render(): React.ReactNode {
        if (this.props.chartProps === undefined) {
            return (<div/>);
        }
        const plotlyProps = WhatIfTab.generatePlotlyProps(
            this.props.jointDataset,
            this.props.chartProps
        );
        return (<div className={WhatIfTab.classNames.dataTab}>
            <div className={this.state.isPanelOpen ?
                WhatIfTab.classNames.expandedPanel :
                WhatIfTab.classNames.collapsedPanel}>
                {this.state.isPanelOpen && (<div>
                    <IconButton 
                        iconProps={{iconName: "ChevronLeft"}}
                        onClick={this.dismissPanel}
                    />
                    <div>Test content</div>
                </div>)}
                {!this.state.isPanelOpen && (<IconButton 
                    iconProps={{iconName: "ChevronRight"}}
                    onClick={this.openPanel}
                />)}
            </div>
            <div className={WhatIfTab.classNames.mainArea}>
                <FilterControl 
                    jointDataset={this.props.jointDataset}
                    filterContext={this.props.filterContext}
                />
                <div className={WhatIfTab.classNames.chartWithAxes}>
                    <div className={WhatIfTab.classNames.chartWithVertical}>
                        <div className={WhatIfTab.classNames.verticalAxis}>
                            <div className={WhatIfTab.classNames.rotatedVerticalBox}>
                                {(this.props.chartProps.chartType === ChartTypes.Scatter) && (
                                    <DefaultButton 
                                        onClick={this.setYOpen.bind(this, true)}
                                        id={this._yButtonId}
                                        text={localization.ExplanationScatter.yValue + this.props.jointDataset.metaDict[this.props.chartProps.yAxis.property].abbridgedLabel}
                                        title={localization.ExplanationScatter.yValue + this.props.jointDataset.metaDict[this.props.chartProps.yAxis.property].label}
                                    />
                                )}
                                {(this.props.chartProps.chartType !== ChartTypes.Scatter) && (
                                    <div>{localization.ExplanationScatter.count}</div>
                                )}
                                {(this.state.yDialogOpen) && (
                                    <AxisConfigDialog 
                                        jointDataset={this.props.jointDataset}
                                        orderedGroupTitles={[ColumnCategories.index, ColumnCategories.dataset, ColumnCategories.outcome]}
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
                    <div className={WhatIfTab.classNames.horizontalAxisWithPadding}>
                        <div className={WhatIfTab.classNames.paddingDiv}></div>
                        <div className={WhatIfTab.classNames.horizontalAxis}>
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
                </div >
            </div>
        </div>);
    }

    private dismissPanel(): void {
        this.setState({isPanelOpen: false});
    }

    private openPanel(): void {
        this.setState({isPanelOpen: true});
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
}