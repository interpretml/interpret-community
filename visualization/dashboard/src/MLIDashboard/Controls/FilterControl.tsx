import React from "react";
import { IJointMeta } from "../JointDataset";
import { Button, IconButton } from "office-ui-fabric-react/lib/Button";
import { IFilterContext, IFilter } from "../Interfaces/IFilter";
import NascentFilter from "./NascentFilter";

export interface IFilterControlProps {
    metaDict: {[key: string]: IJointMeta};
    filterContext: IFilterContext;
}

export interface IFilterControlState {
    dialogOpen: boolean;
}

export default class FilterControl extends React.PureComponent<IFilterControlProps, IFilterControlState> {
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
        return(
            <div>
                <Button
                    onClick={this.openFilter}
                    text="Add Filter"
                />
                {this.state.dialogOpen && (<NascentFilter
                    metaDict={this.props.metaDict}
                    addFilter={this.addFilter}
                    cancel={this.cancelFilter}
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