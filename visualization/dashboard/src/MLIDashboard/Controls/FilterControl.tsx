import React from "react";
import { IJointMeta, JointDataset } from "../JointDataset";
import { Button, IconButton } from "office-ui-fabric-react/lib/Button";
import { IFilterContext, IFilter, FilterMethods } from "../Interfaces/IFilter";
import { FilterEditor } from "./FilterEditor";
import _ from "lodash";

export interface IFilterControlProps {
    jointDataset: JointDataset
    filterContext: IFilterContext;
}

export interface IFilterControlState {
    openedFilter?: IFilter;
    filterIndex?: number;
}

export class FilterControl extends React.PureComponent<IFilterControlProps, IFilterControlState> {
    constructor(props: IFilterControlProps) {
        super(props);
        this.openFilter = this.openFilter.bind(this);
        this.editNewFilter = this.editNewFilter.bind(this);
        this.updateFilter = this.updateFilter.bind(this);
        this.cancelFilter = this.cancelFilter.bind(this);
        this.state = {openedFilter: undefined};
    }

    private readonly initialFilter: IFilter = {
        column: JointDataset.IndexLabel,
        method: this.props.jointDataset.metaDict[JointDataset.IndexLabel].treatAsCategorical ?
            FilterMethods.includes : FilterMethods.greaterThan,
        arg: this.props.jointDataset.metaDict[JointDataset.IndexLabel].treatAsCategorical ?
            [0] : 0
    };

    public render(): React.ReactNode {
        const filterList = this.props.filterContext.filters.map((filter, index) => {
            return (<div>
                <div onClick={this.openFilter.bind(this, index)}>{this.props.jointDataset.metaDict[filter.column].abbridgedLabel}</div>
                <IconButton
                    iconProps={{iconName:"Clear"}}
                    onClick={this.removeFilter.bind(this, index)}
                />
            </div>);
        });
        return(
            <div>
                <Button
                    onClick={this.editNewFilter}
                    text="Add Filter"
                />
                {this.state.openedFilter !== undefined && (<FilterEditor
                    jointDataset={this.props.jointDataset}
                    onAccept={this.updateFilter}
                    onCancel={this.cancelFilter}
                    initialFilter={this.state.openedFilter}
                />)}
                {filterList}
            </div>
        );
    }

    private updateFilter(filter: IFilter): void {
        const index = this.state.filterIndex;
        this.setState({openedFilter: undefined, filterIndex: undefined});
        this.props.filterContext.onUpdate(filter, index);
    }

    private cancelFilter(): void {
        this.setState({openedFilter: undefined, filterIndex: undefined});
    }

    private editNewFilter(): void {
        this.setState({openedFilter: this.initialFilter, filterIndex: this.props.filterContext.filters.length});
    }

    private openFilter(index: number): void {
        this.setState({openedFilter: this.props.filterContext.filters[index], filterIndex: index});
    }

    private removeFilter(index: number): void {
        this.props.filterContext.onDelete(index);
    }
}