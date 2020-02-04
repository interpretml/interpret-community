import React from "react";
import { IJointMeta } from "../JointDataset";
import { Button, PrimaryButton, DefaultButton } from "office-ui-fabric-react/lib/Button";
import { SpinButton } from 'office-ui-fabric-react/lib/SpinButton';
import { FilterMethods, IFilter } from "../Interfaces/IFilter";
import { IComboBoxOption, IComboBox, ComboBox } from "office-ui-fabric-react/lib/ComboBox";
import { Dialog, DialogType, DialogFooter } from "office-ui-fabric-react/lib/Dialog";
import { FabricStyles } from "../FabricStyles";
import { Slider } from "office-ui-fabric-react/lib/Slider";
import { localization } from "../../Localization/localization";
import { RangeTypes } from "mlchartlib";

export interface INascentFilterProps {
    metaDict: {[key: string]: IJointMeta};
    addFilter: (filter: IFilter) => void;
    cancel: () => void;
}

export default class NascentFilter extends React.PureComponent<INascentFilterProps, IFilter> {
    private columnOptions: IComboBoxOption[];
    private comparisonOptions: IComboBoxOption[];
    private categoricalOptions: IComboBoxOption[];
    constructor(props: INascentFilterProps) {
        super(props);
        this.columnOptions =  Object.keys(props.metaDict).map((key, index) => {
            return {
                key: key,
                text: props.metaDict[key].label
            };
        });
        this.comparisonOptions = [
            {
                key: FilterMethods.equal,
                text: localization.Filters.equalComparison
            },
            {
                key: FilterMethods.greaterThan,
                text: localization.Filters.greaterThanComparison
            },
            {
                key: FilterMethods.lessThan,
                text: localization.Filters.lessThanComparison
            }
        ];
        this.state = {} as any;
    }
    public render(): React.ReactNode {
        const selectedColumn = this.state.column !== undefined ?
            this.props.metaDict[this.state.column] :
            undefined;
        const numericDelta = selectedColumn === undefined || selectedColumn.isCategorical || selectedColumn.featureRange.rangeType === RangeTypes.integer ?
            1 : (selectedColumn.featureRange.max - selectedColumn.featureRange.min)/10;
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
                {selectedColumn &&
                selectedColumn.isCategorical && (
                    <div>
                        <ComboBox
                            multiSelect
                            label={localization.Filters.categoricalIncludeValues}
                            className="path-selector"
                            selectedKey={this.state.arg}
                            onChange={this.setCategoricalValues}
                            options={this.categoricalOptions}
                            useComboBoxAsMenuWidth={true}
                            styles={FabricStyles.smallDropdownStyle}
                        />
                    </div>
                )}
                {selectedColumn &&
                !selectedColumn.isCategorical && (
                    <div>
                        <ComboBox
                            label={localization.Filters.numericalComparison}
                            className="path-selector"
                            selectedKey={this.state.method}
                            onChange={this.setComparison}
                            options={this.comparisonOptions}
                            useComboBoxAsMenuWidth={true}
                            styles={FabricStyles.smallDropdownStyle}
                        />
                        <SpinButton
                            styles={{
                                spinButtonWrapper: {maxWidth: "98px"},
                                labelWrapper: { alignSelf: "center"},
                                root: {
                                    display: "inline-flex",
                                    float: "right",
                                    selectors: {
                                        "> div": {
                                            maxWidth: "108px"
                                        }
                                    }
                                }
                            }}
                            label={localization.Filters.numericValue}
                            min={selectedColumn.featureRange.min}
                            max={selectedColumn.featureRange.max}
                            value={this.state.arg.toString()}
                            onIncrement={this.setNumericValue.bind(this, numericDelta, selectedColumn)}
                            onDecrement={this.setNumericValue.bind(this, -numericDelta, selectedColumn)}
                            onValidate={this.setNumericValue.bind(this, 0, selectedColumn)}
                        />
                    </div>
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

    private readonly setComparison = (event: React.FormEvent<IComboBox>, item: IComboBoxOption): void => {
        this.setState({method: item.key as FilterMethods});
    }

    private readonly setNumericValue = (delta: number, column: IJointMeta, stringVal: string): string | void => {
        if (delta === 0) {
            const number = +stringVal;
            if (!Number.isInteger(number) || number > column.featureRange.max || number < column.featureRange.min) {
                return this.state.arg.toString();
            }
            this.setState({arg: number});
        } else {
            const prevVal = this.state.arg as number;
            const newVal = prevVal + delta;
            if (newVal > column.featureRange.max || newVal < column.featureRange.min) {
                return prevVal.toString();
            }
            this.setState({arg: newVal});
        }
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
            filter.method = FilterMethods.lessThan;
            filter.arg = meta.featureRange.max;
        }
        return filter;
    }
}