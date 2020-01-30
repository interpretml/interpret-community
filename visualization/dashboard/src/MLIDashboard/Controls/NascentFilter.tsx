import React from "react";
import { IJointMeta } from "../JointDataset";
import { Button, PrimaryButton, DefaultButton } from "office-ui-fabric-react/lib/Button";
import { Dropdown } from "office-ui-fabric-react/lib/Dropdown";
import { FilterMethods, IFilter } from "../Interfaces/IFilter";
import { IComboBoxOption, IComboBox, ComboBox } from "office-ui-fabric-react/lib/ComboBox";
import { Dialog, DialogType, DialogFooter } from "office-ui-fabric-react/lib/Dialog";
import { FabricStyles } from "../FabricStyles";

export interface INascentFilterProps {
    metaDict: {[key: string]: IJointMeta};
    addFilter: (filter: IFilter) => void;
    cancel: () => void;
}

export default class NascentFilter extends React.PureComponent<INascentFilterProps, IFilter> {
    private columnOptions: IComboBoxOption[];
    private categoricalOptions: IComboBoxOption[];
    constructor(props: INascentFilterProps) {
        super(props);
        this.columnOptions =  Object.keys(props.metaDict).map((key, index) => {
            return {
                key: key,
                text: props.metaDict[key].label
            };
        });
        this.state = {} as any;
    }
    public render(): React.ReactNode {
        return (
            <Dialog
                hidden={false}
                onDismiss={this.props.cancel}
                dialogContentProps={{
                    type: DialogType.largeHeader,
                    title: 'All emails together',
                    subText: 'Your Inbox has changed. No longer does it include favorites, it is a singular destination for your emails.'
                }}
                modalProps={{
                    isBlocking: false,
                    styles: { main: { maxWidth: 650 } }
                }}
            >
                <ComboBox
                    label="Column"
                    className="path-selector"
                    selectedKey={this.state.column}
                    onChange={this.setColumn}
                    options={this.columnOptions}
                    ariaLabel={"chart type picker"}
                    useComboBoxAsMenuWidth={true}
                    styles={FabricStyles.smallDropdownStyle}
                />
                {this.state.column &&
                this.props.metaDict[this.state.column] &&
                this.props.metaDict[this.state.column].isCategorical && (
                    <ComboBox
                        multiSelect
                        label="Include values"
                        className="path-selector"
                        selectedKey={this.state.arg}
                        onChange={this.setCategoricalValues}
                        options={this.categoricalOptions}
                        ariaLabel={"chart type picker"}
                        useComboBoxAsMenuWidth={true}
                        styles={FabricStyles.smallDropdownStyle}
                    />
                )}
                <DialogFooter>
                    <PrimaryButton onClick={this.onClick} text="Save" />
                    <DefaultButton onClick={this.props.cancel} text="Cancel" />
                </DialogFooter>
            </Dialog>
        );
    }

    private readonly setColumn = (event: React.FormEvent<IComboBox>, item: IComboBoxOption): void => {
        const key = item.key as string;
        if (key === this.state.column) {
            return;
        }
        const column = this.buildDefaultFilter(key);
        this.setState(column);
    }

    private readonly onClick = (): void => {
        this.props.addFilter(this.state);
    }

    private readonly setCategoricalValues = (event: React.FormEvent<IComboBox>, item: IComboBoxOption): void => {
        const selectedVals = [...(this.state.arg as number[])];
        const index = selectedVals.indexOf(item.key as number);
        if (item.selected && index !== -1) {
            selectedVals.push(index);
        } else {
            selectedVals.splice(index, 1);
        }
        this.setState({arg: selectedVals});
    }

    private buildDefaultFilter(key: string): IFilter {
        const filter: IFilter = {column: key} as IFilter;
        const meta = this.props.metaDict[key];
        if (meta.isCategorical) {
            filter.method = FilterMethods.includes;
            filter.arg = Array.from(Array(meta.sortedCategoricalValues.length).keys());
            this.categoricalOptions = Object.keys(meta.sortedCategoricalValues).map((label, index) => {
                return {
                    key: index,
                    text: label
                };
            });
        } else {
            filter.method = FilterMethods.greaterThan;
            filter.arg = meta.featureRange.min;
        }
        return filter;
    }
}