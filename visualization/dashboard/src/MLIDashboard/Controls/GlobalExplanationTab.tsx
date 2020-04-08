import React from "react";
import { JointDataset } from "../JointDataset";
import { IExplanationModelMetadata } from "../IExplanationContext";
import { IFilterContext } from "../Interfaces/IFilter";
import { BarChart, LoadingSpinner } from "../SharedComponents";
import { IPlotlyProperty, AccessibleChart } from "mlchartlib";
import { localization } from "../../Localization/localization";
import _ from "lodash";
import { DependencePlot } from "./DependencePlot";
import { IGenericChartProps, ChartTypes } from "../NewExplanationDashboard";
import { mergeStyleSets } from "@uifabric/styling";
import { SpinButton } from "office-ui-fabric-react/lib/SpinButton";
import { Slider } from "office-ui-fabric-react/lib/Slider";
import { ModelExplanationUtils } from "../ModelExplanationUtils";
import { ComboBox, IComboBox, IComboBoxOption } from "office-ui-fabric-react/lib/ComboBox";
import { FabricStyles } from "../FabricStyles";
import { IDropdownOption, Dropdown } from "office-ui-fabric-react/lib/Dropdown";
import { SwarmFeaturePlot } from "./SwarmFeaturePlot";
import { FilterControl } from "./FilterControl";
import { Cohort } from "../Cohort";
import { FeatureImportanceBar, IBarSeries } from "./FeatureImportanceBar";
import { GlobalViolinPlot } from "./GlobalViolinPlot";

export interface IGlobalBarSettings {
    topK: number;
    startingK: number;
    sortOption: string;
    includeOverallGlobal: boolean;
}

export interface IGlobalExplanationTabProps {
    globalBarSettings: IGlobalBarSettings;
    sortVector: number[];
    // selectionContext: SelectionContext;
    theme?: string;
    // messages?: HelpMessageDict;
    jointDataset: JointDataset;
    dependenceProps: IGenericChartProps;
    metadata: IExplanationModelMetadata;
    globalImportance?: number[];
    isGlobalDerivedFromLocal: boolean;
    cohorts: Cohort[];
    cohortIDs: string[];
    onChange: (props: IGlobalBarSettings) => void;
    requestSortVector: () => void;
    onDependenceChange: (props: IGenericChartProps) => void;
}

export interface IGlobalExplanationtabState {
    secondChart: string;
    startingK: number;
    sortArray: number[];
    selectedCohortIndex: number;
}

export class GlobalExplanationTab extends React.PureComponent<IGlobalExplanationTabProps, IGlobalExplanationtabState> {
    private static readonly classNames = mergeStyleSets({
        page: {
            height: "100%",
            display: "flex",
            flexDirection: "column",
            width: "100%"
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
        chartTypeDropdown: {
            margin: "0 5px 0 0"
        },
        globalChart: {
            height: "400px",
            width: "100%"
        }
    });

    private chartOptions: IDropdownOption[] = [];
    private unsortedSeries: IBarSeries[];

    constructor(props: IGlobalExplanationTabProps) {
        super(props);
        if (this.props.globalBarSettings === undefined) {
            this.setDefaultSettings(props);
        }
        if (this.props.jointDataset.localExplanationFeatureCount > 0) {
            this.chartOptions.push({key: 'swarm', text: 'Swarm plot'});
            if (this.props.jointDataset.datasetFeatureCount > 0) {
                this.chartOptions.push({key: 'depPlot', text: 'Dependence plot'});
            }
        }
        this.state = {
            secondChart: this.chartOptions[0].key as string,
            startingK: 0,
            selectedCohortIndex: 0,
            sortArray: ModelExplanationUtils.getSortIndices(
                this.props.cohorts[0].calculateAverageImportance()).reverse()
        };
        this.buildYsandNames();
        this.onSecondaryChartChange = this.onSecondaryChartChange.bind(this);
        this.selectPointFromChart = this.selectPointFromChart.bind(this);
        this.updateStartingK = this.updateStartingK.bind(this);
        this.updateSortArray = this.updateSortArray.bind(this);
        this.setSelectedCohort = this.setSelectedCohort.bind(this);
    }

    public componentDidUpdate(prevProps: IGlobalExplanationTabProps) {
        if (this.props.cohorts !== prevProps.cohorts) {
            this.updateIncludedCohortsOnCohortEdit();
        }
    }

    public render(): React.ReactNode {
        if (this.props.globalBarSettings === undefined) {
            return (<div/>);
        }
        const cohortOptions: IDropdownOption[] = this.props.cohorts.map((cohort, index) => {return {key: index, text: cohort.name};});

        return (
        <div className={GlobalExplanationTab.classNames.page}>
            <FeatureImportanceBar
                unsortedX={this.props.metadata.featureNamesAbridged}
                unsortedSeries={this.unsortedSeries}
                theme={this.props.theme}
                topK={this.props.globalBarSettings.topK}
                onSort={this.updateSortArray}
                onClick={this.selectPointFromChart}
                onSetStartingIndex={this.updateStartingK}
            />
            {cohortOptions && (<Dropdown 
                styles={{ dropdown: { width: 150 } }}
                options={cohortOptions}
                selectedKey={this.state.selectedCohortIndex}
                onChange={this.setSelectedCohort}
            />)}
            <ComboBox
                className={GlobalExplanationTab.classNames.chartTypeDropdown}
                label={localization.GlobalTab.secondaryChart}
                selectedKey={this.state.secondChart}
                onChange={this.onSecondaryChartChange}
                options={this.chartOptions}
                ariaLabel={"chart selector"}
                useComboBoxAsMenuWidth={true}
                styles={FabricStyles.smallDropdownStyle}
            />
            {this.state.secondChart === 'depPlot' && (<DependencePlot 
                chartProps={this.props.dependenceProps}
                cohort={this.props.cohorts[this.state.selectedCohortIndex]}
                jointDataset={this.props.jointDataset}
                metadata={this.props.metadata}
                onChange={this.props.onDependenceChange}
            />)}
            {this.state.secondChart === 'swarm' && (<GlobalViolinPlot
                jointDataset={this.props.jointDataset}
                metadata={this.props.metadata}
                cohort={this.props.cohorts[this.state.selectedCohortIndex]}
                topK={this.props.globalBarSettings.topK}
                startingK={this.state.startingK}
                sortVector={this.state.sortArray}
            />)}
        </div>);
    }

    private setSelectedCohort(event: React.FormEvent<HTMLDivElement>, item: IDropdownOption): void {
        this.setState({selectedCohortIndex: item.key as number});
    }

    private buildYsandNames(): void {
        this.unsortedSeries = this.props.cohorts.map((cohort, i) => {
            return {
                name: cohort.name,
                unsortedY: cohort.calculateAverageImportance(),
                color: FabricStyles.plotlyColorHexPalette[i]
            }
        });
        this.forceUpdate();
    }

    private updateIncludedCohortsOnCohortEdit(): void {
        let selectedCohortIndex = this.state.selectedCohortIndex;
        if (selectedCohortIndex >= this.props.cohorts.length) {
            selectedCohortIndex = 0;
        }
        this.setState({selectedCohortIndex}, () => {
            this.buildYsandNames();
        });
    }

    private updateSortArray(newArray: number[]): void {
        this.setState({sortArray: newArray});
    }

    private updateStartingK(startingK: number): void {
        this.setState({startingK})
    }

    private setDefaultSettings(props: IGlobalExplanationTabProps): void {
        const result: IGlobalBarSettings = {} as IGlobalBarSettings;
        result.topK = Math.min(this.props.jointDataset.localExplanationFeatureCount, 4);
        result.startingK = 0;
        result.sortOption = "global";
        result.includeOverallGlobal = !this.props.isGlobalDerivedFromLocal;
        this.props.onChange(result);
    }

    private onSecondaryChartChange(event: React.FormEvent<IComboBox>, item: IComboBoxOption): void {
        this.setState({secondChart: item.key as string});
    }

    private selectPointFromChart(data: any): void {

        const trace = data.points[0];
        const featureNumber = this.state.sortArray[trace.x];
        // set to dependence plot initially, can be changed if other feature importances available
        const xKey = JointDataset.DataLabelRoot + featureNumber.toString();
        const xIsDithered = this.props.jointDataset.metaDict[xKey].treatAsCategorical;
        const yKey = JointDataset.ReducedLocalImportanceRoot + featureNumber.toString();
        const chartProps: IGenericChartProps = {
            chartType: ChartTypes.Scatter,
            xAxis: {
                property: xKey,
                options: {
                    dither: xIsDithered,
                    bin: false
                }
            },
            yAxis: {
                property: yKey,
                options: {}
            }
        };
        this.props.onDependenceChange(chartProps);
        this.setState({secondChart: "depPlot", selectedCohortIndex: trace.curveNumber});
        // each group will be a different cohort, setting the selected cohort should follow impl.
    }  
}