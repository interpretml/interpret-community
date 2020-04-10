import React from "react";
import * as memoize from 'memoize-one';
import { JointDataset, ColumnCategories } from "../JointDataset";
import { IExplanationModelMetadata } from "../IExplanationContext";
import { mergeStyleSets } from "@uifabric/styling";
import { IPlotlyProperty, AccessibleChart, PlotlyMode, RangeTypes } from "mlchartlib";
import { localization } from "../../Localization/localization";
import { IRenderFunction } from "@uifabric/utilities";
import { IconButton, DefaultButton } from "office-ui-fabric-react/lib/Button";
import { SearchBox } from "office-ui-fabric-react/lib/SearchBox";
import { IconNames } from "@uifabric/icons";
import { FilterControl } from "./FilterControl";
import { IFilterContext } from "../Interfaces/IFilter";
import { ChartTypes, IGenericChartProps, ISelectorConfig } from "../NewExplanationDashboard";
import { AxisConfigDialog } from "./AxisConfigDialog";
import { Transform } from "plotly.js-dist";
import _ from "lodash";
import { TextField } from "office-ui-fabric-react/lib/TextField";
import { IDropdownOption, Dropdown } from "office-ui-fabric-react/lib/Dropdown";
import { Cohort } from "../Cohort";
import { FeatureImportanceBar } from "./FeatureImportanceBar";
import { Pivot, PivotItem } from "office-ui-fabric-react/lib/Pivot";
import { MultiICEPlot } from "./MultiICEPlot";
import { FabricStyles } from "../FabricStyles";
import { InteractiveLegend, ILegendItem } from "./InteractiveLegend";

export interface IWhatIfTabProps {
    theme: any;
    jointDataset: JointDataset;
    metadata: IExplanationModelMetadata;
    cohorts: Cohort[];
    chartProps: IGenericChartProps;
    onChange: (config: IGenericChartProps) => void; 
    invokeModel: (data: any[], abortSignal: AbortSignal) => Promise<any[]>;
}

export interface IWhatIfTabState {
    isPanelOpen: boolean;
    xDialogOpen: boolean;
    yDialogOpen: boolean;
    selectedWhatIfRootIndex?: number;
    editingDataCustomIndex?: number;
    customPoints: Array<{[key: string]: any}>;
    indexText: string;
    selectedIndexErrorMessage?: string;
    selectedCohortIndex: number;
    filteredFeatureList: Array<{key: string, label: string}>;
    requestList: AbortController[];
    selectedPointsIndexes: number[]; 
}

interface ISelectedRowInfo {
    name: string;
    color: string;
    rowData: any[];
    rowImportances?: number[];
    isCustom: boolean;
    index?: number;
}

export class WhatIfTab extends React.PureComponent<IWhatIfTabProps, IWhatIfTabState> {
    private static readonly MAX_SELECTION = 2;
    private static readonly defaultIndexString = " - ";
    private static readonly colorPath = "Color";
    private static readonly namePath = "Name";
    private readonly colorOptions: IDropdownOption[] = [
        {key: "#FF0000", text: "Red", data: {color: "#FF0000"}}, 
        {key: "#00FF00", text: "Green", data: {color: "#00FF00"}},
        {key: "#0000FF", text: "Blue", data: {color: "#0000FF"}}
    ]; 
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
                b: 20,
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
            borderRight: "1px solid black",
            display: "flex",
            flexDirection: "column"
        },
        parameterList: {
            display: "flex",
            flexGrow: 1,
            flexDirection: "column"
        },
        featureList: {
            display: "flex",
            flexGrow: 1,
            flexDirection: "column",
            maxHeight: "400px",
            overflowY: "auto"
        },
        customPointsList: {
            borderTop: "2px solid black",
            height: "250px",
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
        },
        customPointItem: {
            padding: "2px 5px",
            borderBottom: "1px solid black",
            cursor: "pointer"
        },
        colorBox: {
            width: "10px",
            height: "10px",
            paddingLeft: "3px",
            display: "inline-block"
        },
        secondaryTab: {
            display: "inline-flex",
            flexDirection: "row",
            width: "100%"
        }
    });

    private buildCustomRowSeries(customRows: Array<{[key: string]: any}>): any[] {
        return customRows.map((row, i) => {
            return {
                name: row[WhatIfTab.namePath],
                unsortedFeatureValues: JointDataset.datasetSlice(row, this.props.jointDataset.metaDict, this.props.jointDataset.localExplanationFeatureCount),
                unsortedY: undefined,
                color: row[WhatIfTab.colorPath],
                onEdit: this.setTemporaryPointToCustomPoint.bind(this, i),
                onDelete: this.removeCustomPoint.bind(this, i)
            };
        });
    }

    private buildSelectedRowSeriess(rowIndices: number[]): any[] {
        const result: any[] = [];
        rowIndices.forEach((rowIndex, seriesIndex) => {
            const row = this.props.jointDataset.getRow(rowIndex);
            result.push({
                name: localization.WhatIf.rowLabel + rowIndex,
                unsortedFeatureValues: JointDataset.datasetSlice(row, this.props.jointDataset.metaDict, this.props.jointDataset.localExplanationFeatureCount),
                unsortedY: JointDataset.localExplanationSlice(row, this.props.jointDataset.localExplanationFeatureCount) as number[],
                disabled: false,
                color: FabricStyles.plotlyColorHexPalette[seriesIndex],
                onDelete: this.toggleSelectionOfPoint.bind(this,rowIndex)
            });
        });
        return result;
    }

    private readonly _xButtonId = "x-button-id";
    private readonly _yButtonId = "y-button-id";

    private readonly featureList: Array<{key: string, label: string}> = new Array(this.props.jointDataset.datasetFeatureCount)
        .fill(0).map((unused, colIndex) => {
            const key = JointDataset.DataLabelRoot + colIndex.toString();
            const meta = this.props.jointDataset.metaDict[key];
            return {key, label: meta.label.toLowerCase()};
        });

    private selectedRowSeries: any[];
    private customRowSeries: any[];
    private temporaryPoint: {[key: string]: any};
    private customPoints: Array< {[key: string]: any}> = [];

    constructor(props: IWhatIfTabProps) {
        super(props);
        if (props.chartProps === undefined) {
            this.generateDefaultChartAxes();
        }
        this.state = {
            isPanelOpen: true,
            xDialogOpen: false,
            yDialogOpen: false,
            selectedWhatIfRootIndex: 0,
            indexText: "0",
            editingDataCustomIndex: undefined,
            customPoints: [],
            selectedCohortIndex: 0,
            requestList: [],
            filteredFeatureList: this.featureList,
            selectedPointsIndexes: this.getDefaultSelectedPointIndexes(props.cohorts[0])
        };
        this.temporaryPoint = this.createCopyOfFirstRow();
        //this.seriesOfRows = this.buildJoinedSelectedRows(this.state.selectedPointsIndexes, this.state.customPoints);
        this.dismissPanel = this.dismissPanel.bind(this);
        this.openPanel = this.openPanel.bind(this);
        this.onXSet = this.onXSet.bind(this);
        this.onYSet = this.onYSet.bind(this);
        this.setIndexText = this.setIndexText.bind(this);
        this.setCustomRowProperty = this.setCustomRowProperty.bind(this);
        this.setCustomRowPropertyDropdown = this.setCustomRowPropertyDropdown.bind(this);
        this.savePoint = this.savePoint.bind(this);
        this.saveCopyOfPoint = this.saveCopyOfPoint.bind(this);
        this.selectPointFromChart = this.selectPointFromChart.bind(this);
        this.filterFeatures = this.filterFeatures.bind(this);
        this.setSelectedCohort = this.setSelectedCohort.bind(this);
        this.fetchData = _.debounce(this.fetchData.bind(this), 400);
    }

    public render(): React.ReactNode {
        if (this.props.chartProps === undefined) {
            return (<div/>);
        }
        const plotlyProps = this.generatePlotlyProps(
            this.props.jointDataset,
            this.props.chartProps,
            this.props.cohorts[this.state.selectedCohortIndex]
        );
        //const selectedRows = WhatIfTab.buildJoinedSelectedRows(this.props.jointDataset, this.state.selectedPointsIndexes, this.state.customPoints);
        // const unsortedYsImportance = selectedRows.map(row => {
        //     return row.rowImportances;
        // }).filter(x => !!x);
        // const names = selectedRows.map(row => {
        //     if (row.rowImportances) {
        //         return row.name
        //     }
        //     return undefined;
        // }).filter(x => !!x);
        // const datasets = selectedRows.map(row => {
        //     return row.rowData;
        // }).filter(x => !!x);
        const cohortOptions: IDropdownOption[] = this.props.cohorts.map((cohort, index) => {return {key: index, text: cohort.name};});
        return (<div className={WhatIfTab.classNames.dataTab}>
            <div className={this.state.isPanelOpen ?
                WhatIfTab.classNames.expandedPanel :
                WhatIfTab.classNames.collapsedPanel}>
                {this.state.isPanelOpen && (<div>
                    <IconButton 
                        iconProps={{iconName: "ChevronLeft"}}
                        onClick={this.dismissPanel}
                    />
                    <div className={WhatIfTab.classNames.parameterList}>
                        <TextField
                            label={localization.WhatIf.indexLabel}
                            value={this.state.indexText}
                            onChange={this.setIndexText}
                            styles={{ fieldGroup: { width: 100 } }}
                        />
                        <TextField
                            label={localization.WhatIf.namePropLabel}
                            value={this.temporaryPoint[WhatIfTab.namePath]}
                            onChange={this.setCustomRowProperty.bind(this, WhatIfTab.namePath, true)}
                            styles={{ fieldGroup: { width: 200 } }}
                        />
                        <Dropdown
                            label={localization.WhatIf.colorLabel}
                            selectedKey={this.temporaryPoint[WhatIfTab.colorPath]}
                            onChange={this.setCustomRowPropertyDropdown.bind(this, WhatIfTab.colorPath)}
                            onRenderTitle={this._onRenderTitle}
                            onRenderOption={this._onRenderOption}
                            styles={{ dropdown: { width: 200 } }}
                            options={this.colorOptions}
                        />
                        <div>Features: </div>
                        <SearchBox
                            placeholder={localization.WhatIf.filterFeaturePlaceholder}
                            onChange={this.filterFeatures}
                        />
                        <div className={WhatIfTab.classNames.featureList}>
                            {this.state.filteredFeatureList.map(item => {
                                return <TextField
                                    label={this.props.jointDataset.metaDict[item.key].abbridgedLabel}
                                    value={this.temporaryPoint[item.key].toString()}
                                    onChange={this.setCustomRowProperty.bind(this, item.key, this.props.jointDataset.metaDict[item.key].treatAsCategorical)}
                                    styles={{ fieldGroup: { width: 100 } }}
                                />  
                            })}
                        </div>
                    </div>
                    <DefaultButton
                        disabled={this.temporaryPoint[JointDataset.PredictedYLabel] === undefined}
                        text={"Save Point"}
                        onClick={this.savePoint}
                    />
                    {this.state.editingDataCustomIndex !== undefined && (
                        <DefaultButton
                            disabled={this.temporaryPoint[JointDataset.PredictedYLabel] === undefined}
                            text={"Save copy of point"}
                            onClick={this.saveCopyOfPoint}
                        />
                    )}
                    <div className={WhatIfTab.classNames.customPointsList}>
                        {this.state.customPoints.length !== 0 &&
                            this.state.customPoints.map((row, index) => {
                                return <div className={WhatIfTab.classNames.customPointItem}
                                            onClick={this.setTemporaryPointToCustomPoint.bind(this, index)}>
                                            <div className={WhatIfTab.classNames.colorBox} 
                                                style={{backgroundColor: row[WhatIfTab.colorPath]}} />
                                            <span>{row[WhatIfTab.namePath]}</span>
                                            <span>{row[JointDataset.PredictedYLabel] || "loading"}</span>
                                            <IconButton iconProps={{iconName: "Cancel"}}
                                                onClick={this.removeCustomPoint.bind(this, index)}
                                            />
                                    </div>
                            })
                        }
                        {this.state.customPoints.length === 0 && (
                            <div>{localization.WhatIf.noCustomPoints}</div>
                        )}
                    </div>
                </div>)}
                {!this.state.isPanelOpen && (<IconButton 
                    iconProps={{iconName: "ChevronRight"}}
                    onClick={this.openPanel}
                />)}
            </div>
            <div className={WhatIfTab.classNames.mainArea}>
                {/* <FilterControl 
                    jointDataset={this.props.jointDataset}
                    filterContext={this.props.filterContext}
                /> */}
                {cohortOptions && (<Dropdown 
                    styles={{ dropdown: { width: 150 } }}
                    options={cohortOptions}
                    selectedKey={this.state.selectedCohortIndex}
                    onChange={this.setSelectedCohort}
                />)}
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
                            onClickHandler={this.selectPointFromChart}
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
                <Pivot>
                    <PivotItem headerText={"Feature Importance"} className={WhatIfTab.classNames.secondaryTab}>
                        {/* <FeatureImportanceBar
                            unsortedX={this.props.metadata.featureNamesAbridged}
                            unsortedYs={unsortedYsImportance}
                            theme={this.props.theme}
                            topK={8}
                            seriesNames={names}
                        />
                        <InteractiveLegend
                            onClick={() => {}}
                            items={selectedRows.map(row => {
                                return {
                                    name: row.name,
                                    color: row.color,
                                    editable: row.isCustom
                                }
                            })}
                        /> */}
                    </PivotItem>
                    <PivotItem headerText={"ICE"} className={WhatIfTab.classNames.secondaryTab}>
                        {/* <MultiICEPlot 
                            invokeModel={this.props.invokeModel}
                            datapoints={datasets}
                            jointDataset={this.props.jointDataset}
                            metadata={this.props.metadata}
                            theme={this.props.theme}
                        /> */}
                        {/* <InteractiveLegend
                            onClick={() => {}}
                            items={selectedRows.map(row => {
                                return {
                                    name: row.name,
                                    color: row.color,
                                    editable: row.isCustom
                                }
                            })}
                        /> */}
                    </PivotItem>
                </Pivot>
            </div>
        </div>);
    }

    private getDefaultSelectedPointIndexes(cohort: Cohort): number[] {
        const indexes = cohort.unwrap(JointDataset.IndexLabel);
        if (indexes.length > 0) {
            return [indexes[0]];
        }
        return [];
    }

    private setSelectedCohort(event: React.FormEvent<HTMLDivElement>, item: IDropdownOption): void {
        this.setState({selectedCohortIndex: item.key as number, selectedPointsIndexes: []});
    }

    private _onRenderOption = (option: IDropdownOption): JSX.Element => {
        return (
          <div>
            {option.data && option.data.color && (
              <div style={{ marginRight: '8px', backgroundColor: option.data.color }} className={WhatIfTab.classNames.colorBox} aria-hidden="true" title={option.data.color} />
            )}
            <span>{option.text}</span>
          </div>
        );
      };
    
      private _onRenderTitle = (options: IDropdownOption[]): JSX.Element => {
        const option = options[0];
    
        return (
          <div>
            {option.data && option.data.color && (
              <div style={{ marginRight: '8px', backgroundColor: option.data.color}} className={WhatIfTab.classNames.colorBox} aria-hidden="true" title={option.data.color} />
            )}
            <span>{option.text}</span>
          </div>
        );
      };

    private setIndexText(event: React.FormEvent<HTMLInputElement | HTMLTextAreaElement>, newValue?: string): void {
        const asNumber = +newValue;
        const maxIndex = this.props.jointDataset.metaDict[JointDataset.IndexLabel].featureRange.max;
        if (Number.isInteger(asNumber) && asNumber >= 0 && asNumber <= maxIndex) {
            this.setTemporaryPointToCopyOfDatasetPoint(asNumber);
        } else {
            const error = localization.formatString(localization.WhatIf.indexErrorMessage, maxIndex) as string;
            this.setState({indexText: newValue, selectedWhatIfRootIndex: undefined, selectedIndexErrorMessage: error});
        }
    }

    private setTemporaryPointToCopyOfDatasetPoint(index: number): void {
        this.temporaryPoint = this.props.jointDataset.getRow(index);
        this.temporaryPoint[WhatIfTab.namePath] = localization.formatString(localization.WhatIf.defaultCustomRootName, index) as string;
        this.temporaryPoint[WhatIfTab.colorPath] = this.colorOptions[0].key;

        this.setState({
            selectedWhatIfRootIndex: index,
            indexText: index.toString(),
            selectedIndexErrorMessage: undefined,
            editingDataCustomIndex: undefined
        });
    }

    private setTemporaryPointToCustomPoint(index: number): void {
        this.temporaryPoint = _.cloneDeep(this.state.customPoints[index]);
        this.setState({
            selectedWhatIfRootIndex: this.temporaryPoint[JointDataset.IndexLabel],
            indexText: this.temporaryPoint[JointDataset.IndexLabel].toString(),
            selectedIndexErrorMessage: undefined,
            editingDataCustomIndex: index
        });
    }

    private removeCustomPoint(index: number): void {
        this.setState(prevState => {
            const customPoints = [...prevState.customPoints];
            customPoints.splice(index, 1);
            return {customPoints};
        });
    }

    private setCustomRowProperty(key: string, isString: boolean, event: React.FormEvent<HTMLInputElement | HTMLTextAreaElement>, newValue?: string): void {
        const editingData = this.temporaryPoint;
        if (isString) {
            editingData[key] = newValue;
        } else {
            const asNumber = +newValue;
            if (!Number.isFinite(asNumber)) {
                alert('thats no number')
            }
            editingData[key] = asNumber;
        }
        this.fetchData(editingData);
    }

    private setCustomRowPropertyDropdown(key: string, event: React.FormEvent<HTMLDivElement>, item: IDropdownOption): void {
        const editingData = this.temporaryPoint;
        editingData[key] = item.key;
        this.fetchData(editingData);
    }

    private savePoint(): void {
        let editingDataCustomIndex = this.customPoints.length;
        if (this.state.editingDataCustomIndex !== undefined) {
            this.customPoints[this.state.editingDataCustomIndex] = this.temporaryPoint;
            editingDataCustomIndex = this.state.editingDataCustomIndex
        }
        else {
            this.customPoints.push(this.temporaryPoint);
            
        }
        this.temporaryPoint = _.cloneDeep(this.temporaryPoint);
        this.setState({editingDataCustomIndex});
    }

    private saveCopyOfPoint(): void {
        let editingDataCustomIndex = this.customPoints.length;
        this.customPoints.push(this.temporaryPoint);
        this.temporaryPoint = _.cloneDeep(this.temporaryPoint);
        this.setState({editingDataCustomIndex});
    }

    private createCopyOfFirstRow(): {[key: string]: any} {
        const indexes = this.getDefaultSelectedPointIndexes(this.props.cohorts[this.state.selectedCohortIndex]);
        if (indexes.length === 0) {
            return undefined;
        }
        const customData = this.props.jointDataset.getRow(indexes[0]) as any;
        customData[WhatIfTab.namePath] = localization.formatString(localization.WhatIf.defaultCustomRootName, indexes[0]) as string;
        customData[WhatIfTab.colorPath] = this.colorOptions[0].key;
        return customData;
    }

    private dismissPanel(): void {
        this.setState({isPanelOpen: false});
        window.dispatchEvent(new Event('resize'));
    }

    private openPanel(): void {
        this.setState({isPanelOpen: true});
        window.dispatchEvent(new Event('resize'));
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

    private filterFeatures(event?: React.ChangeEvent<HTMLInputElement>, newValue?: string): void {
        if (newValue === undefined || newValue === null || !/\S/.test(newValue)) {
            this.setState({filteredFeatureList: this.featureList});
        }
        const filteredFeatureList = this.featureList.filter(item => {
            return item.label.includes(newValue.toLowerCase());
        });
        this.setState({filteredFeatureList});
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

    private selectPointFromChart(data: any): void {
        const trace = data.points[0];
        // custom point
        if (trace.curveNumber === 1) {
            this.setTemporaryPointToCustomPoint(trace.pointNumber);
        } else {
            const index = trace.customdata[JointDataset.IndexLabel];
            this.setTemporaryPointToCopyOfDatasetPoint(index);
            this.toggleSelectionOfPoint(index);
        }
    }

    private toggleSelectionOfPoint(index: number): void {
        const indexOf = this.state.selectedPointsIndexes.indexOf(index);
        let newSelections = [];
        if (indexOf === -1) {
            const startingIdex = this.state.selectedPointsIndexes.length > WhatIfTab.MAX_SELECTION ? 1 : 0;
            newSelections = this.state.selectedPointsIndexes.slice(startingIdex);
            newSelections.push(index);
        } else {
            newSelections = [...this.state.selectedPointsIndexes]
            newSelections.splice(indexOf,1);
        }
        this.setState({selectedPointsIndexes: newSelections});
    }

    private fetchData(fetchingReference: {[key: string]: any}): void {
        if (this.state.requestList[this.state.editingDataCustomIndex] !== undefined) {
            this.state.requestList[this.state.editingDataCustomIndex].abort();
        }
        const requestList = [...this.state.requestList];
        const abortController = new AbortController();
        requestList[this.state.editingDataCustomIndex] = abortController;
        const rawData = JointDataset.datasetSlice(fetchingReference, this.props.jointDataset.metaDict, this.props.jointDataset.datasetFeatureCount);
        fetchingReference[JointDataset.PredictedYLabel] = undefined;
        const promise = this.props.invokeModel([rawData], abortController.signal);
        

        this.setState({requestList}, async () => {
            try {
                const fetchedData = await promise;
                if (Array.isArray(fetchedData)) {
                    fetchingReference[JointDataset.PredictedYLabel] = fetchedData[0];
                    delete this.state.requestList[this.state.editingDataCustomIndex];
                    this.forceUpdate();
                }
            } catch(err) {
                if (err.name === 'AbortError') {
                    return;
                }
                if (err.name === 'PythonError') {
                    alert("error");
                }
            }
        });
    }

    private generatePlotlyProps(jointData: JointDataset, chartProps: IGenericChartProps, cohort: Cohort): IPlotlyProperty {
        const plotlyProps = _.cloneDeep(WhatIfTab.basePlotlyProperties);
        plotlyProps.data[0].hoverinfo = "all";
        const indexes = cohort.unwrap(JointDataset.IndexLabel);
        plotlyProps.data[0].type = chartProps.chartType;
        plotlyProps.data[0].mode = PlotlyMode.markers;
        plotlyProps.data[0].marker = {
            symbol: indexes.map(i => this.state.selectedPointsIndexes.includes(i) ? "square" : "circle") as any,
            color: indexes.map((rowIndex, i) => {
                const selectionIndex = this.state.selectedPointsIndexes.indexOf(rowIndex);
                return selectionIndex !== -1 ? FabricStyles.plotlyColorHexPalette[selectionIndex] : "#0c297e"
            }) as any
        }

        plotlyProps.data[1] = {
            type: "scattergl",
            mode: PlotlyMode.markers,
            marker: {
                color: JointDataset.unwrap(this.state.customPoints, WhatIfTab.colorPath)
            }
        }

        if (chartProps.xAxis) {
            if (jointData.metaDict[chartProps.xAxis.property].isCategorical) {
                const xLabels = jointData.metaDict[chartProps.xAxis.property].sortedCategoricalValues;
                const xLabelIndexes = xLabels.map((unused, index) => index);
                _.set(plotlyProps, "layout.xaxis.ticktext", xLabels);
                _.set(plotlyProps, "layout.xaxis.tickvals", xLabelIndexes);
            }
            const rawX = cohort.unwrap(chartProps.xAxis.property);
            const customX = JointDataset.unwrap(this.state.customPoints, chartProps.xAxis.property);
            if (chartProps.xAxis.options.dither) {
                const dithered = cohort.unwrap(JointDataset.DitherLabel);
                const customDithered = JointDataset.unwrap(this.state.customPoints, JointDataset.DitherLabel);
                plotlyProps.data[0].x = dithered.map((dither, index) => { return rawX[index] + dither;});
                plotlyProps.data[1].x = customDithered.map((dither, index) => { return customX[index] + dither;});
            } else {
                plotlyProps.data[0].x = rawX;
                plotlyProps.data[1].x = customX;
            }
        }
        if (chartProps.yAxis) {
            if (jointData.metaDict[chartProps.yAxis.property].isCategorical) {
                const yLabels = jointData.metaDict[chartProps.yAxis.property].sortedCategoricalValues;
                const yLabelIndexes = yLabels.map((unused, index) => index);
                _.set(plotlyProps, "layout.yaxis.ticktext", yLabels);
                _.set(plotlyProps, "layout.yaxis.tickvals", yLabelIndexes);
            }
            const rawY = cohort.unwrap(chartProps.yAxis.property);
            const customY = JointDataset.unwrap(this.state.customPoints, chartProps.yAxis.property);
            if (chartProps.yAxis.options.dither) {
                const dithered = cohort.unwrap(JointDataset.DitherLabel);
                const customDithered = JointDataset.unwrap(this.state.customPoints, JointDataset.DitherLabel);
                plotlyProps.data[0].y = dithered.map((dither, index) => { return rawY[index] + dither;});
                plotlyProps.data[1].y = customDithered.map((dither, index) => { return customY[index] + dither;});
            } else {
                plotlyProps.data[0].y = rawY;
                plotlyProps.data[1].y = customY;
            }
        }

            
        plotlyProps.data[0].customdata = WhatIfTab.buildCustomData(jointData, chartProps, cohort);
        plotlyProps.data[0].hovertemplate = WhatIfTab.buildHoverTemplate(chartProps);
        return plotlyProps;
    }

    private static buildHoverTemplate(chartProps: IGenericChartProps): string {
        let hovertemplate = "";
        if (chartProps.xAxis) {
            if (chartProps.xAxis.options.dither) {
                hovertemplate += "x: %{customdata.X}<br>";
            } else {
                hovertemplate += "x: %{x}<br>";
            }
        }
        if (chartProps.yAxis) {
            if (chartProps.yAxis.options.dither) {
                hovertemplate += "y: %{customdata.Y}<br>";
            } else {
                hovertemplate += "y: %{y}<br>";
            }
        }
        hovertemplate += "<extra></extra>";
        return hovertemplate;
    }

    private static buildCustomData(jointData: JointDataset, chartProps: IGenericChartProps, cohort: Cohort): Array<any> {
        const customdata = cohort.unwrap(JointDataset.IndexLabel).map(val => {
            const dict = {};
            dict[JointDataset.IndexLabel] = val;
            return dict;
        });
        if (chartProps.chartType === ChartTypes.Scatter) {
            const xAxis = chartProps.xAxis;
            if (xAxis && xAxis.property && xAxis.options.dither) {
                const rawX = cohort.unwrap(chartProps.xAxis.property);
                rawX.forEach((val, index) => {
                    // If categorical, show string value in tooltip
                    if (jointData.metaDict[chartProps.xAxis.property].isCategorical) {
                        customdata[index]["X"] = jointData.metaDict[chartProps.xAxis.property]
                            .sortedCategoricalValues[val];
                    } else {
                        customdata[index]["X"] = val;
                    }
                });
            }
            const yAxis = chartProps.yAxis;
            if (yAxis && yAxis.property && yAxis.options.dither) {
                const rawY = cohort.unwrap(chartProps.yAxis.property);
                rawY.forEach((val, index) => {
                    // If categorical, show string value in tooltip
                    if (jointData.metaDict[chartProps.yAxis.property].isCategorical) {
                        customdata[index]["Y"] = jointData.metaDict[chartProps.yAxis.property]
                            .sortedCategoricalValues[val];
                    } else {
                        customdata[index]["Y"] = val;
                    }
                });
            }
        }
        return customdata;
    }

    private generateDefaultChartAxes(): void {
        const yKey = JointDataset.DataLabelRoot + "0";
        const yIsDithered = this.props.jointDataset.metaDict[yKey].isCategorical;
        const chartProps: IGenericChartProps = {
            chartType: ChartTypes.Scatter,
            xAxis: {
                property: JointDataset.IndexLabel,
                options: {}
            },
            yAxis: {
                property: yKey,
                options: {
                    dither: yIsDithered,
                    bin: false
                }
            },
            colorAxis: {
                property: this.props.jointDataset.hasPredictedY ?
                    JointDataset.PredictedYLabel : JointDataset.IndexLabel,
                options: {}
            }
        }
        this.props.onChange(chartProps);
    }
}