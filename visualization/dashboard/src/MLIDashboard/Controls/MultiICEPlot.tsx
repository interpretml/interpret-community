import React from "react";
import { JointDataset } from "../JointDataset";
import { IRangeView } from "./ICEPlot";
import { IDropdownOption } from "office-ui-fabric-react/lib/Dropdown";
import _ from "lodash";
import { IPlotlyProperty, RangeTypes, AccessibleChart, PlotlyMode } from "mlchartlib";
import { IComboBox, IComboBoxOption, ComboBox } from "office-ui-fabric-react/lib/ComboBox";
import { localization } from "../../Localization/localization";
import { NoDataMessage } from "../SharedComponents";
import { ModelTypes, IExplanationModelMetadata } from "../IExplanationContext";
import { FabricStyles } from "../FabricStyles";
import { TextField } from "office-ui-fabric-react/lib/TextField";
import { Data } from "plotly.js-dist";
import { ModelExplanationUtils } from "../ModelExplanationUtils";

export interface IMultiICEPlotProps {
    invokeModel?: (data: any[], abortSignal: AbortSignal) => Promise<any[]>;
    datapoints: Array<string | number>[];
    colors: string[];
    jointDataset: JointDataset;
    metadata: IExplanationModelMetadata;
    theme?: string;
} 

export interface IMultiICEPlotState {
    requestFeatureKey: string | undefined;
    yAxes: number[][] | number[][][];
    xAxisArray: string[] | number[];
    abortControllers: AbortController[];
    rangeView: IRangeView | undefined;
    errorMessage?: string;
}

export class MultiICEPlot extends React.PureComponent<IMultiICEPlotProps, IMultiICEPlotState> {
    private static buildYAxis(metadata: IExplanationModelMetadata): string {
        if (metadata.modelType === ModelTypes.regression) {
            return localization.IcePlot.prediction;
        } if (metadata.modelType === ModelTypes.binary) {
            return localization.IcePlot.predictedProbability + ":<br>" + metadata.classNames[0];
        }
    }
    private static buildPlotlyProps(metadata: IExplanationModelMetadata, featureName: string, colors: string[], rangeType: RangeTypes, xData?: Array<number | string>,  yData?: number[][] | number[][][]): IPlotlyProperty | undefined {
        if (yData === undefined || xData === undefined || yData.length === 0 || yData.some(row => row === undefined)) {
            return undefined;
        }
        const data: Data[] = (yData as number[][][]).map((singleRow, rowIndex) => {
            const transposedY: number[][] = Array.isArray(yData[0]) ?
                ModelExplanationUtils.transpose2DArray((singleRow)) :
                [singleRow] as any;
            return {
                mode: rangeType === RangeTypes.categorical ? PlotlyMode.markers : PlotlyMode.linesMarkers,
                type: 'scatter',
                x: xData,
                y: transposedY[0],
                marker: {
                    color: colors[rowIndex]
                },
                name: "row"
            }
        }) as any;
        return {
            config: { displaylogo: false, responsive: true, displayModeBar: false },
            data,
            layout: {
                dragmode: false,
                autosize: true,
                font: {
                    size: 10
                },
                margin: {
                    t: 10,
                    b: 30,
                    r: 10
                },
                hovermode: 'closest',
                showlegend: false,
                yaxis: {
                    automargin: true,
                    title: MultiICEPlot.buildYAxis(metadata)
                },
                xaxis: {
                    title: featureName,
                    automargin: true
                }
            } as any
        }
    }
    
    private featuresOption: IDropdownOption[];
    private debounceFetchData: () => void;
    constructor(props: IMultiICEPlotProps) {
        super(props);
        this.featuresOption =  new Array(this.props.jointDataset.datasetFeatureCount).fill(0)
            .map((unused, index) => {
                const key = JointDataset.DataLabelRoot + index.toString();
                return {key, text: this.props.jointDataset.metaDict[key].abbridgedLabel};
            });
        const requestFeatureKey = this.featuresOption[0].key as string;
        const rangeView = this.buildRangeView(requestFeatureKey);
        const xAxisArray = this.buildRange(rangeView);
        this.state = {
            yAxes:[],
            abortControllers: [],
            rangeView,
            requestFeatureKey,
            xAxisArray
        };
        this.onFeatureSelected = this.onFeatureSelected.bind(this);
        this.onCategoricalRangeChanged = this.onCategoricalRangeChanged.bind(this);
        this.onMinRangeChanged = this.onMinRangeChanged.bind(this);
        this.onMaxRangeChanged = this.onMaxRangeChanged.bind(this);
        this.onStepsRangeChanged = this.onStepsRangeChanged.bind(this);
        this.debounceFetchData = _.debounce(this.fetchData.bind(this), 500);
    }

    public componentDidMount(): void {
        this.fetchData();
    }

    public componentDidUpdate(prevProps: IMultiICEPlotProps): void {
        if (this.props.datapoints !== prevProps.datapoints) {
            this.fetchData();
        }
    }

    public componentWillUnmount(): void {
        this.state.abortControllers.forEach(abortController => {
            if (abortController !== undefined) {
                abortController.abort();
            }
        });
    }

    public render(): React.ReactNode {
        if (this.props.invokeModel === undefined) {
            return <NoDataMessage />;
        } 
        else {
            const hasOutgoingRequest = this.state.abortControllers.some(x => x !== undefined);
            const plotlyProps = MultiICEPlot.buildPlotlyProps(
                this.props.metadata, 
                this.props.jointDataset.metaDict[this.state.requestFeatureKey].label,
                this.props.colors,
                this.state.rangeView.type,
                this.state.xAxisArray, 
                this.state.yAxes);
            const hasError = this.state.rangeView !== undefined && (
                this.state.rangeView.maxErrorMessage !== undefined ||
                this.state.rangeView.minErrorMessage !== undefined || 
                this.state.rangeView.stepsErrorMessage !== undefined);
            return (<div className="ICE-wrapper">
                <div className="feature-pickers">
                    <div className="feature-picker">
                        <div className="path-selector">
                            <ComboBox
                                options={this.featuresOption}
                                onChange={this.onFeatureSelected}
                                label={localization.IcePlot.featurePickerLabel}
                                ariaLabel="feature picker"
                                selectedKey={this.state.requestFeatureKey }
                                useComboBoxAsMenuWidth={true}
                                styles={FabricStyles.defaultDropdownStyle}
                            />
                        </div>
                        {this.state.rangeView !== undefined && 
                        <div className="rangeview">
                            {this.state.rangeView.type === RangeTypes.categorical &&
                                <ComboBox
                                    multiSelect
                                    selectedKey={this.state.rangeView.selectedOptionKeys as string[]}
                                    allowFreeform={true}
                                    autoComplete="on"
                                    options={this.state.rangeView.categoricalOptions}
                                    onChange={this.onCategoricalRangeChanged}
                                    styles={FabricStyles.defaultDropdownStyle}
                                />
                            }
                            {this.state.rangeView.type !== RangeTypes.categorical && 
                                <div className="parameter-set">
                                    <TextField 
                                        label={localization.IcePlot.minimumInputLabel}
                                        styles={FabricStyles.textFieldStyle}
                                        value={this.state.rangeView.min}
                                        onChange={this.onMinRangeChanged}
                                        errorMessage={this.state.rangeView.minErrorMessage}/>
                                    <TextField 
                                        label={localization.IcePlot.maximumInputLabel}
                                        styles={FabricStyles.textFieldStyle}
                                        value={this.state.rangeView.max}
                                        onChange={this.onMaxRangeChanged}
                                        errorMessage={this.state.rangeView.maxErrorMessage}/>
                                    <TextField 
                                        label={localization.IcePlot.stepInputLabel}
                                        styles={FabricStyles.textFieldStyle}
                                        value={this.state.rangeView.steps}
                                        onChange={this.onStepsRangeChanged}
                                        errorMessage={this.state.rangeView.stepsErrorMessage}/>
                                </div>
                            }
                        </div>}
                    </div>
                </div>
                {hasOutgoingRequest &&
                <div className="loading">{localization.IcePlot.loadingMessage}</div>}
                {this.state.errorMessage &&
                <div className="loading">
                    {this.state.errorMessage}
                </div>}
                {(plotlyProps === undefined && !hasOutgoingRequest) && 
                <div className="charting-prompt">{localization.IcePlot.submitPrompt}</div>}
                {hasError && 
                <div className="charting-prompt">{localization.IcePlot.topLevelErrorMessage}</div>}
                {(plotlyProps !== undefined && !hasOutgoingRequest && !hasError) &&
                <div className="second-wrapper">
                    <div className="chart-wrapper">
                        <AccessibleChart
                            plotlyProps={plotlyProps}
                            theme={this.props.theme}
                        />
                    </div>
                </div>}
            </div>);
        }
    }

    private onFeatureSelected(event: React.FormEvent<IComboBox>, item: IDropdownOption): void {
        const rangeView = this.buildRangeView(item.key as string);
        const xAxisArray = this.buildRange(rangeView);
        this.setState({rangeView, xAxisArray, requestFeatureKey: item.key as string }, ()=> {
            this.debounceFetchData();
        });
    }

    private onMinRangeChanged(ev: React.FormEvent<HTMLInputElement>, newValue?: string): void {
        const val = + newValue;
        const rangeView = _.cloneDeep(this.state.rangeView);
        rangeView.min = newValue;
        if (Number.isNaN(val) || (this.state.rangeView.type === RangeTypes.integer && !Number.isInteger(val))) {
            rangeView.minErrorMessage = this.state.rangeView.type === RangeTypes.integer ? localization.IcePlot.integerError : localization.IcePlot.numericError;
            this.setState({rangeView});
        }
        else {
            const xAxisArray = this.buildRange(rangeView);
            rangeView.minErrorMessage = undefined;
            this.setState({rangeView, xAxisArray}, () => {this.debounceFetchData()});
        }
    }

    private onMaxRangeChanged(ev: React.FormEvent<HTMLInputElement>, newValue?: string): void {
        const val = + newValue;
        const rangeView = _.cloneDeep(this.state.rangeView);
        rangeView.max = newValue;
        if (Number.isNaN(val) || (this.state.rangeView.type === RangeTypes.integer && !Number.isInteger(val))) {
            rangeView.maxErrorMessage = this.state.rangeView.type === RangeTypes.integer ? localization.IcePlot.integerError : localization.IcePlot.numericError;
            this.setState({rangeView});
        }
        else {
            const xAxisArray = this.buildRange(rangeView);
            rangeView.maxErrorMessage = undefined;
            this.setState({rangeView, xAxisArray}, () => {this.debounceFetchData()});
        }
    }

    private onStepsRangeChanged(ev: React.FormEvent<HTMLInputElement>, newValue?: string): void {
        const val = + newValue;
        const rangeView = _.cloneDeep(this.state.rangeView);
        rangeView.steps = newValue;
        if (!Number.isInteger(val)) {
            rangeView.stepsErrorMessage = localization.IcePlot.integerError;
            this.setState({rangeView});
        }
        else {
            const xAxisArray = this.buildRange(rangeView);
            rangeView.stepsErrorMessage = undefined;
            this.setState({rangeView, xAxisArray}, () => {this.debounceFetchData()});
        }
    }

    private onCategoricalRangeChanged(event: React.FormEvent<IComboBox>, option?: IComboBoxOption, index?: number, value?: string): void {
        const rangeView = _.cloneDeep(this.state.rangeView);
        const currentSelectedKeys = rangeView.selectedOptionKeys || [];
        if (option) {
            // User selected/de-selected an existing option
            rangeView.selectedOptionKeys = this.updateSelectedOptionKeys(currentSelectedKeys, option);
        } 
        else if (value !== undefined) {
            // User typed a freeform option
            const newOption: IComboBoxOption = { key: value, text: value };
            rangeView.selectedOptionKeys = [...currentSelectedKeys, newOption.key as string];
            rangeView.categoricalOptions.push(newOption);
        }
        const xAxisArray = this.buildRange(rangeView);
        this.setState({rangeView, xAxisArray}, () => {this.debounceFetchData()});
    }

    private updateSelectedOptionKeys = (selectedKeys: Array<string|number>, option: IComboBoxOption): Array<string|number> => {
        selectedKeys = [...selectedKeys]; // modify a copy
        const index = selectedKeys.indexOf(option.key as string);
        if (option.selected && index < 0) {
          selectedKeys.push(option.key as string);
        } else {
          selectedKeys.splice(index, 1);
        }
        return selectedKeys;
    }

    private fetchData(): void {
        this.state.abortControllers.forEach(abortController => {
            if (abortController !== undefined) {
                abortController.abort();
            }
        });
        const promises = this.props.datapoints.map((row, index) => {
            const abortController = new AbortController();
            this.state.abortControllers[index] = abortController;
            const permutations = this.buildDataSpans(row, this.state.xAxisArray);
            return this.props.invokeModel(permutations, abortController.signal);
        });
        const yAxes = this.props.datapoints.map(x => undefined);

        this.setState({yAxes, errorMessage: undefined}, async () => {
            try {
                const fetchedData = await Promise.all(promises);
                if (Array.isArray(fetchedData) && fetchedData.every(prediction => Array.isArray(prediction))) {
                    this.setState({yAxes: fetchedData, abortControllers: this.props.datapoints.map(x => undefined)});
                }
            } catch(err) {
                if (err.name === 'AbortError') {
                    return;
                }
                if (err.name === 'PythonError') {
                    this.setState({errorMessage: localization.formatString(localization.IcePlot.errorPrefix, err.message) as string})
                }
            }
        });
    }

    private buildDataSpans(row: Array<string | number>, range: Array<string | number>): Array<number|string>[] {
        return range.map((val: number | string) => {
            const copy = _.cloneDeep(row);
            copy[this.state.rangeView.featureIndex] = val;
            return copy;
        });
    }

    private buildRangeView(featureKey: string): IRangeView {
        const summary = this.props.jointDataset.metaDict[featureKey];
        if (summary.treatAsCategorical) {
            // Columns that are passed in as categorical strings should be strings when passed to predict
            if (summary.isCategorical) {
                return {
                    featureIndex: summary.index,
                    selectedOptionKeys: summary.sortedCategoricalValues,
                    categoricalOptions: summary.sortedCategoricalValues.map(text => {return {key: text, text}}),
                    type: RangeTypes.categorical
                }; 
            }
            // Columns that were integers that are flagged in the UX as categorical should still be integers when
            // calling predict on the model.
            return {
                featureIndex: summary.index,
                selectedOptionKeys: summary.sortedCategoricalValues.map(x => +x),
                categoricalOptions: summary.sortedCategoricalValues.map(text => {return {key: +text, text: text.toString()}}),
                type: RangeTypes.categorical
            };  
        }
        else {
            return {
                featureIndex: summary.index,
                min: summary.featureRange.min.toString(),
                max: summary.featureRange.max.toString(),
                steps: '20',
                type: summary.featureRange.rangeType
            };
        }
    }

    private buildRange(rangeView: IRangeView): number[] | string[] {
        if (rangeView === undefined ||
            rangeView.minErrorMessage !== undefined ||
            rangeView.maxErrorMessage !== undefined ||
            rangeView.stepsErrorMessage !== undefined) {
            return [];
        }
        const min = +rangeView.min;
        const max = +rangeView.max;
        const steps = +rangeView.steps;

        if (rangeView.type === RangeTypes.categorical && Array.isArray(rangeView.selectedOptionKeys)) {
            return rangeView.selectedOptionKeys as string[];
        } else if (!Number.isNaN(min) && !Number.isNaN(max) && Number.isInteger(steps)) {
            let delta = steps > 0 ? (max - min) / steps :
                max - min;
            return _.uniq(Array.from({length: steps}, (x, i)=> rangeView.type === RangeTypes.integer ? 
                Math.round(min + i * delta) :
                min + i * delta));
        } else {
            return [];
        }
    }

}