import React from "react";
import { interactiveLegendStyles } from "./InteractiveLegend.styles";
import { IconButton } from "office-ui-fabric-react";

export enum SortingState {
    ascending = "ascending",
    descending = "descending",
    none = "none"
}

export interface ILegendItem {
    disabled?: boolean;
    disabledMessage?: string;
    activated: boolean;
    sortingState?: SortingState;
    color: string;
    name: string; 
    onClick: () => void;
    onDelete?: () => void;
    onSort?: () => void;
    onEdit?: () => void;
}

export interface IInteractiveLegendProps {
    items: ILegendItem[];
}

export class InteractiveLegend extends React.PureComponent<IInteractiveLegendProps> {
    private readonly classes = interactiveLegendStyles();

    public render(): React.ReactNode {
        return (<div className={this.classes.root}>
            {this.props.items.map((item, index) => {
                return this.buildRowElement(item);
            })}
        </div>);
    }

    private buildRowElement(item: ILegendItem): React.ReactNode {
        let sortIcon: string = "Remove";
        if (sortIcon === SortingState.ascending) {
            sortIcon = "Up";
        }
        if (sortIcon === SortingState.descending) {
            sortIcon = "Down";
        }
        if (item.disabled) {
            return(<div className={this.classes.disabledItem} title={item.disabledMessage || ""}>
                <div className={this.classes.disabledColorBox}/>
                <div className={this.classes.label}>{item.name}</div>
                {item.onEdit !== undefined && (
                    <IconButton 
                        className={this.classes.editButton}
                        iconProps={{iconName:"Edit"}}
                        onClick={item.onEdit} />
                )}
                {item.onDelete !== undefined && (
                    <IconButton
                        className={this.classes.deleteButton}
                        iconProps={{iconName:"Clear"}}
                        onClick={item.onDelete}
                    />
                )}
            </div>);
        }
        const rootClass = item.activated === false ? this.classes.inactiveItem : this.classes.item;
        return(
        <div className={rootClass}>
            <div className={this.classes.colorBox} style={{backgroundColor: item.color}}/>
            {item.onSort !== undefined && (
                <IconButton 
                    className={this.classes.editButton}
                    iconProps={{iconName: sortIcon}}
                    onClick={item.onSort} />
            )}
            <div className={this.classes.label}>{item.name}</div>
            {item.onEdit !== undefined && (
                <IconButton 
                    className={this.classes.editButton}
                    iconProps={{iconName:"Edit"}}
                    onClick={item.onEdit} />
            )}
            {item.onDelete !== undefined && (
                <IconButton
                    className={this.classes.deleteButton}
                    iconProps={{iconName:"Clear"}}
                    onClick={item.onDelete}
                />
            )}
        </div>);
    }
}