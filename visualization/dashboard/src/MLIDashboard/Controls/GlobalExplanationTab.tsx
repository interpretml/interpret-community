import React from "react";
import { JointDataset } from "../JointDataset";
import { IExplanationModelMetadata } from "../IExplanationContext";
import { IFilterContext } from "../Interfaces/IFilter";
import { BarChart } from "../SharedComponents";
import { IPlotlyProperty, AccessibleChart } from "mlchartlib";
import { localization } from "../../Localization/localization";
import _ from "lodash";
import { DependencePlot } from "./DependencePlot";
import { IGenericChartProps } from "../NewExplanationDashboard";

export interface IGlobalBarSettings {
    topK: number;
    sortOption: string;
    includeOverallGlobal: boolean;
    sortIndexVector: number[];
}

export interface IGlobalExplanationTabProps {
    globalBarSettings: IGlobalBarSettings;
    // selectionContext: SelectionContext;
    theme?: string;
    // messages?: HelpMessageDict;
    jointDataset: JointDataset;
    dependenceProps: IGenericChartProps;
    metadata: IExplanationModelMetadata;
    globalImportance: number[];
    subsetAverageImportance: number[];
    isGlobalDerivedFromLocal: boolean;
    filterContext: IFilterContext;
    onChange: (props: IGlobalBarSettings) => void;
    onDependenceChange: (props: IGenericChartProps) => void;
}

export class GlobalExplanationTab extends React.PureComponent<IGlobalExplanationTabProps> {
    constructor(props: IGlobalExplanationTabProps) {
        super(props);
        if (this.props.globalBarSettings === undefined) {
            this.setDefaultSettings(props);
        }
    }
    public render(): React.ReactNode {
        if (this.props.globalBarSettings === undefined) {
            return (<div/>);
        }
        const plotlyProps = this.buildBarPlotlyProps();
        return(<>
            <AccessibleChart
                plotlyProps={plotlyProps}
                sharedSelectionContext={undefined}
                theme={this.props.theme}
                onSelection={undefined}
            />
            <DependencePlot 
                chartProps={this.props.dependenceProps}
                jointDataset={this.props.jointDataset}
                metadata={this.props.metadata}
                onChange={this.props.onDependenceChange}
            />
        </>);
    }

    private buildBarPlotlyProps(): IPlotlyProperty {
        const sortedIndexVector = this.props.globalBarSettings.sortIndexVector.slice(-1 * this.props.globalBarSettings.topK).reverse();
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
                showlegend: this.props.globalBarSettings.includeOverallGlobal
            } as any
        };

        const x = sortedIndexVector.map((unused, index) => index);
        const y = sortedIndexVector.map(index => this.props.subsetAverageImportance[index]);

        baseSeries.data.push({
            orientation: 'v',
            type: 'bar',
            name: 'Absolute Average of Subset',
            x,
            y
        } as any);

        if (this.props.globalBarSettings.includeOverallGlobal) {
            baseSeries.data.push({
                orientation: 'v',
                type: 'bar',
                name: 'Global Importance',
                x,
                y: sortedIndexVector.map(index => this.props.globalImportance[index])
            } as any);
        }

        const ticktext = sortedIndexVector.map(i =>this.props.metadata.featureNamesAbridged[i]);
        const tickvals = sortedIndexVector.map((val, index) => index);

        _.set(baseSeries, 'layout.xaxis.ticktext', ticktext);
        _.set(baseSeries, 'layout.xaxis.tickvals', tickvals);
        return baseSeries;
    }

    private setDefaultSettings(props: IGlobalExplanationTabProps): void {
        const result: IGlobalBarSettings = {} as IGlobalBarSettings;
        result.topK = Math.min(this.props.jointDataset.localExplanationFeatureCount, 4);
        result.sortOption = "global";
        result.includeOverallGlobal = this.props.filterContext.filters.length > 0 || !this.props.isGlobalDerivedFromLocal;
        result.sortIndexVector = [0,1,2,3,4];
        this.props.onChange(result);
    }
}