import React from "react";
import { IJointMeta, JointDataset } from "../JointDataset";
import { Button, IconButton } from "office-ui-fabric-react/lib/Button";
import { IFilterContext, IFilter, FilterMethods } from "../Interfaces/IFilter";
import { FilterEditor } from "./FilterEditor";
import _ from "lodash";
import { mergeStyleSets } from "@uifabric/styling";
import { Stack, Label, FontWeights, FontSizes } from "office-ui-fabric-react";
import { Text } from "office-ui-fabric-react";


export interface IFilterControlProps {
    jointDataset: JointDataset
    filterContext: IFilterContext;
}

export interface IFilterControlState {
    openedFilter?: IFilter;
    filterIndex?: number;
    filters?:IFilter[];
}

export class FilterControl extends React.PureComponent<IFilterControlProps, IFilterControlState> {
    private static readonly classNames = mergeStyleSets({
        existingFilter: {
            border: '1px solid #0078D4',
            boxSizing: 'border-box',
            borderRadius: '3px',
            display: 'inline-flex',
            flexDirection:"row",
            height:"25px"
        },
        filterLabel: {
            padding: "1px 9px 6px 11px",
            minWidth:"75px",
            maxWidth: "90px",
            color: "#0078D4",
            height:"25px",
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis"
        },
        defaultFilterList: {
            color: "#979797",
            marginLeft:"10px"
        },
        wrapper: {
            display: "flex",
            flexDirection:"column"
        },
        addedFilter:
        {
            fontWeight: FontWeights.semibold,
            color: "#000000",
            marginLeft:"45px",
            height:"30px",
            width:"178px"
        },
        addedFilterDiv:{
            marginRight:"40px",
            marginLeft:"45px",
            marginBottom:"18px",
            maxHeight:"97px",
            overflowY:"auto"
        },
        filterIcon:{
            height:"25px",
            width:"25px",
        }
        
    });
    constructor(props: IFilterControlProps) {
        super(props);
        this.editFilter = this.editFilter.bind(this);
        this.updateFilter = this.updateFilter.bind(this);
        this.cancelFilter = this.cancelFilter.bind(this);
        this.state = {openedFilter: undefined, filterIndex: this.props.filterContext.filters.length};
    }

    public render(): React.ReactNode {
        const filterList = this.props.filterContext.filters.map((filter, index) => {
            return (<div key={index} className={FilterControl.classNames.existingFilter}>
                    <Text variant={"small"} className={FilterControl.classNames.filterLabel}>{this.props.jointDataset.metaDict[filter.column].abbridgedLabel 
                    + " " + filter.method + " " + filter.arg}</Text>
                <IconButton
                    className={FilterControl.classNames.filterIcon}
                    iconProps={{iconName:"Edit"}}
                    onClick={this.editFilter.bind(this, filter, index)}
                />
                <IconButton
                    className={FilterControl.classNames.filterIcon}
                    iconProps={{iconName:"Clear"}}
                    onClick={this.removeFilter.bind(this, index)}
                />
            </div>);
        });

        return(<div className={FilterControl.classNames.wrapper}>
            <FilterEditor
                jointDataset={this.props.jointDataset}
                onAccept={this.updateFilter}
                onCancel={this.cancelFilter}
                initialFilter={this.state.openedFilter}
            />

        <Text variant={"medium"} className={FilterControl.classNames.addedFilter} >Added Filters</Text>
        <div className={FilterControl.classNames.addedFilterDiv}>
        {filterList.length>0
            ? <div>{filterList}</div>
            : <div>
                <Text variant={"smallPlus"} className={FilterControl.classNames.defaultFilterList}>No filters added yet</Text>
            </div>
        }
        </div>     
        </div>
);
    }

    private updateFilter(filter: IFilter): void {
        const index = this.state.filterIndex;
        this.setState({openedFilter: undefined, filterIndex: undefined});
        this.props.filterContext.onUpdate(filter, index);
        this.forceUpdate();
    }

    private cancelFilter(): void {
        this.setState({openedFilter: undefined, filterIndex: undefined});
    }

    private removeFilter(index: number): void {
        this.props.filterContext.onDelete(index);
        this.setState({filterIndex:this.props.filterContext.filters.length});
        this.forceUpdate();        
    }

    private editFilter(filter:IFilter, index: number): void {      
        this.setState({openedFilter: this.props.filterContext.filters[index], filterIndex: index});
        this.forceUpdate();
    }
}