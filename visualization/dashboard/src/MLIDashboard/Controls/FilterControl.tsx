import React from "react";
import { IJointMeta, JointDataset } from "../JointDataset";
import { Button, IconButton } from "office-ui-fabric-react/lib/Button";
import { IFilterContext, IFilter, FilterMethods } from "../Interfaces/IFilter";
import { FilterEditor } from "./FilterEditor";

export interface IFilterControlProps {
    jointDataset: JointDataset
    filterContext: IFilterContext;
}

export interface IFilterControlState {
    dialogOpen: boolean;
}

export class FilterControl extends React.PureComponent<IFilterControlProps, IFilterControlState> {
    constructor(props: IFilterControlProps) {
        super(props)
        this.openFilter = this.openFilter.bind(this);
        this.addFilter = this.addFilter.bind(this);
        this.cancelFilter = this.cancelFilter.bind(this);
        this.state = {dialogOpen: false};
    }
    public render(): React.ReactNode {
        const filterList = this.props.filterContext.filters.map((filter, index) => {
            return (<div>
                <div>{filter.column}</div>
                <IconButton 
                    iconProps={{iconName:"Clear"}}
                    onClick={this.removeFilter.bind(this, index)}
                />
            </div>)
        });
        const initialFilter: IFilter = {
            column: JointDataset.IndexLabel,
            method: this.props.jointDataset.metaDict[JointDataset.IndexLabel].treatAsCategorical ?
                FilterMethods.includes : FilterMethods.greaterThan,
            arg: this.props.jointDataset.metaDict[JointDataset.IndexLabel].treatAsCategorical ?
                [0] : 0
        };
        return(
            <div>
                <Button
                    onClick={this.openFilter}
                    text="Add Filter"
                />
                {this.state.dialogOpen && (<FilterEditor
                    jointDataset={this.props.jointDataset}
                    onAccept={this.addFilter}
                    onCancel={this.cancelFilter}
                    initialFilter={initialFilter}
                />)}
                {filterList}
            </div>
        );
    }

    private addFilter(filter: IFilter): void {
        this.setState({dialogOpen: false});
        this.props.filterContext.onAdd(filter);
    }

    private cancelFilter(): void {
        this.setState({dialogOpen: false});
    }

    private openFilter(): void {
        this.setState({dialogOpen: true});
    }

    private removeFilter(index: number): void {
        this.props.filterContext.onDelete(index);
    }
}