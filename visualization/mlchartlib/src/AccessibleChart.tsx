import * as _ from 'lodash';
import * as Plotly from 'plotly.js-dist';
import { PlotlyHTMLElement, Layout } from 'plotly.js-dist';
import * as React from 'react';
import uuidv4 from 'uuid/v4';
import { formatValue } from './DisplayFormatters';
import { PlotlyThemes, IPlotlyTheme } from './PlotlyThemes';

import { IPlotlyProperty } from './IPlotlyProperty';
import {SelectionContext} from './SelectionContext';

type SelectableChartType = 'scatter' | 'multi-line' | 'non-selectable';

const s = require('./AccessibleChart.css');
export interface AccessibleChartProps {
    plotlyProps: IPlotlyProperty;
    theme: string;
    themeOverride?: Partial<IPlotlyTheme>;
    sharedSelectionContext: SelectionContext;
    relayoutArg?: Partial<Layout>;
    localizedStrings?: any;
    onSelection?: (chartID: string, selectionIds: string[], plotlyProps: IPlotlyProperty) => void;
}

export class AccessibleChart extends React.Component<AccessibleChartProps> {
    public guid: string = uuidv4();
    private timer: number;
    private subscriptionId: string;
    private plotlyRef: PlotlyHTMLElement;
    private isClickHandled: boolean = false;

    constructor(props: AccessibleChartProps) {
        super(props);
        this.onChartClick = this.onChartClick.bind(this);
    }

    public componentDidMount(): void {
        if (this.hasData()) {
            this.resetRenderTimer();
            this.subscribeToSelections();
        }
    }

    public componentDidUpdate(prevProps: AccessibleChartProps): void {
        if (
            (!_.isEqual(prevProps.plotlyProps, this.props.plotlyProps) || this.props.theme !== prevProps.theme) &&
            this.hasData()
        ) {
            this.resetRenderTimer();
            if (this.plotSelectionType(prevProps.plotlyProps) !== this.plotSelectionType(this.props.plotlyProps)) {
                // The callback differs based on chart type, if the chart is now a different type, un and re subscribe.
                if (this.subscriptionId && this.props.sharedSelectionContext) {
                    this.props.sharedSelectionContext.unsubscribe(this.subscriptionId);
                }
                this.subscribeToSelections();
            }
        } else if (!_.isEqual(this.props.relayoutArg, prevProps.relayoutArg) && this.guid) {
            Plotly.relayout(this.guid, this.props.relayoutArg);
        }
    }

    public componentWillUnmount(): void {
        if (this.subscriptionId && this.props.sharedSelectionContext) {
            this.props.sharedSelectionContext.unsubscribe(this.subscriptionId);
        }
        if (this.timer) {
            window.clearTimeout(this.timer);
        }
    }

    public render(): React.ReactNode {
        if (this.hasData()) {
            return (
                <>
                    <div
                        className="GridChart"
                        id={this.guid}
                    />
                    {this.createTableWithPlotlyData(this.props.plotlyProps.data)}
                </>
            );
        }
        return <div className="centered">{this.props.localizedStrings ? this.props.localizedStrings['noData'] : 'No Data'}</div>;
    }

    private hasData(): boolean {
        return (
            this.props.plotlyProps &&
            this.props.plotlyProps.data.length > 0 &&
            _.some(this.props.plotlyProps.data, datum => !_.isEmpty(datum.y) || !_.isEmpty(datum.x))
        );
    }

    private subscribeToSelections(): void {
        if (this.props.sharedSelectionContext && this.props.onSelection) {
            this.subscriptionId = this.props.sharedSelectionContext.subscribe({
                selectionCallback: selections => {
                    this.props.onSelection(this.guid, selections, this.props.plotlyProps);
                }
            });
        }
    }

    private resetRenderTimer(): void {
        if (this.timer) {
            window.clearTimeout(this.timer);
        }
        const themedProps = this.props.theme
            ? PlotlyThemes.applyTheme(this.props.plotlyProps, this.props.theme, this.props.themeOverride)
            : _.cloneDeep(this.props.plotlyProps);
        this.timer = window.setTimeout(async () => {
            this.plotlyRef = await Plotly.react(this.guid, themedProps.data, themedProps.layout, themedProps.config);
            if (this.props.sharedSelectionContext && this.props.onSelection) {
                this.props.onSelection(this.guid, this.props.sharedSelectionContext.selectedIds, this.props.plotlyProps);
            }

            if (!this.isClickHandled) {
                this.isClickHandled = true;
                this.plotlyRef.on('plotly_click', this.onChartClick);
            }
            this.setState({ loading: false });
        }, 0);
    }

    private onChartClick(data: any): void {
        const selectionType = this.plotSelectionType(this.props.plotlyProps);
        if (selectionType !== 'non-selectable' && this.props.sharedSelectionContext) {
            if (this.props.sharedSelectionContext === undefined) {
                return;
            }
            const clickedId =
                selectionType === 'multi-line'
                    ? (data.points[0].data as any).customdata[0]
                    : (data.points[0] as any).customdata;
            const selections: string[] = this.props.sharedSelectionContext.selectedIds.slice();
            const existingIndex = selections.indexOf(clickedId);
            if (existingIndex !== -1) {
                selections.splice(existingIndex, 1);
            } else {
                selections.push(clickedId);
            }
            this.props.sharedSelectionContext.onSelect(selections);
        }
    }

    private plotSelectionType(plotlyProps: IPlotlyProperty): SelectableChartType {
        if (plotlyProps.data.length > 0 && plotlyProps.data[0] && (((plotlyProps.data[0].type as any) === 'scatter') || (plotlyProps.data[0].type as any) === 'scattergl')) {
            if (
                plotlyProps.data.length > 1 &&
                plotlyProps.data.every(trace => {
                    const customdata = (trace as any).customdata;
                    return customdata && customdata.length === 1;
                })
            ) {
                return 'multi-line';
            }
            if (
                (plotlyProps.data[0].mode as string).includes('markers') &&
                (plotlyProps.data[0] as any).customdata !== undefined
            ) {
                return 'scatter';
            }
        }
        return 'non-selectable';
    }

    private createTableWithPlotlyData(data: Plotly.Data[]): React.ReactNode {
        return (
            <table className="plotly-table hidden">
                <tbody>
                    {data.map((datum, index) => {
                        const xDataLength = datum.x ? datum.x.length : 0;
                        const yDataLength = datum.y ? datum.y.length : 0;
                        const tableWidth = Math.max(xDataLength, yDataLength);
                        // Building this table is slow, need better accesibility for large charts than an unreadable table
                        if (tableWidth > 500) {
                            return;
                        }

                        const xRowCells = [];
                        const yRowCells = [];
                        for (let i = 0; i < tableWidth; i++) {
                            // Add String() because sometimes data may be Nan
                            xRowCells.push(<td key={i + '.x'}>{datum.x ? formatValue(datum.x[i]) : ''}</td>);
                            yRowCells.push(<td key={i + '.y'}>{datum.y ? formatValue(datum.y[i]) : ''}</td>);
                        }
                        return [<tr key={index + '.x'}>{xRowCells}</tr>, <tr key={index + '.y'}>{yRowCells}</tr>];
                    })}
                </tbody>
            </table>
        );
    }
}
