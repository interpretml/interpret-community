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
    key?:string;
}

let id = 1;
export class FilterControl extends React.PureComponent<IFilterControlProps, IFilterControlState> {
    private static readonly classNames = mergeStyleSets({
        existingFilter: {
            border: '1px solid #0078D4',
            borderRadius: '5px',
            display: 'inline-flex',
            marginRight: "4px",
            minWidth:"115px"
        },
        filterLabel: {
            padding: "3px 8px 2px 4px",
            minWidth: "90px",
            color: "#0078D4"
        },
        filterList: {
            fontWeight: FontWeights.regular,
            fontSize: FontSizes.smallPlus,
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
            fontSize: FontSizes.medium,
            color: "#000000",
            marginLeft:"39px",
            height:"30px",
            width:"178px"
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
                <div
                    className={FilterControl.classNames.filterLabel}
                    >
                    {this.props.jointDataset.metaDict[filter.column].abbridgedLabel 
                    + " " + filter.method + " " + filter.arg}
                </div>
                <div>
                <IconButton
                    iconProps={{iconName:"Edit"}}
                    onClick={this.editFilter.bind(this, filter, index)}
                />
                <IconButton
                    iconProps={{iconName:"Clear"}}
                    onClick={this.removeFilter.bind(this, index)}
                />
                </div>
            </div>);
        });

        return(<div className={FilterControl.classNames.wrapper}>
                    <FilterEditor
                        jointDataset={this.props.jointDataset}
                        onAccept={this.updateFilter}
                        onCancel={this.cancelFilter}
                        initialFilter={this.state.openedFilter}
                        key={this.state.filterIndex}
                    />

                <Text variant={"medium"} block className={FilterControl.classNames.addedFilter} >Added Filters</Text>
                
                {filterList.length>0
                    ? <Stack horizontalAlign="start">{filterList}</Stack>
                    : <Stack className={FilterControl.classNames.filterList}> no filters added</Stack>
                }
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
        //this.setState({filterIndex:undefined});
        this.forceUpdate()
        
    }

    private editFilter(filter:IFilter, index: number): void {      
        this.setState({openedFilter: this.props.filterContext.filters[index], filterIndex: index});
        this.forceUpdate()
    }