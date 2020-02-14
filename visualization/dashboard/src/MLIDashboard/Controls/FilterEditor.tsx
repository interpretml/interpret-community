import React from "react";
import { IJointMeta, JointDataset } from "../JointDataset";
import { Button, PrimaryButton, DefaultButton } from "office-ui-fabric-react/lib/Button";
import { SpinButton } from 'office-ui-fabric-react/lib/SpinButton';
import { FilterMethods, IFilter } from "../Interfaces/IFilter";
import { IComboBoxOption, IComboBox, ComboBox } from "office-ui-fabric-react/lib/ComboBox";
import { Dialog, DialogType, DialogFooter } from "office-ui-fabric-react/lib/Dialog";
import { FabricStyles } from "../FabricStyles";
import { localization } from "../../Localization/localization";
import { RangeTypes } from "mlchartlib";
import { Target, Callout } from "office-ui-fabric-react/lib/Callout";
import _ from "lodash";
import { getTheme, mergeStyleSets } from "@uifabric/styling";
import { DetailsList, Selection, SelectionMode } from "office-ui-fabric-react/lib/DetailsList";
import { Checkbox } from "office-ui-fabric-react/lib/Checkbox";

export interface IFilterEditorProps {
    jointDataset: JointDataset;
    initialFilter: IFilter;
    target?: Target;
    onAccept: (filter: IFilter) => void;
    onCancel: () => void;
}

const theme = getTheme();
const styles = mergeStyleSets({
    wrapper: {
        minHeight: "300px",
        width: "400px",
        display: "flex"
    },
    leftHalf: {
        display: "inline-flex",
        width: "50%",
        height: "100%",
        borderRight: "2px solid #CCC"
    },
    rightHalf: {
        display: "inline-flex",
        width: "50%",
        flexDirection: "column"
    }
});

export default class FilterEditor extends React.PureComponent<IFilterEditorProps, IFilter> {
    private _leftSelection: Selection;
    private readonly dataArray: IComboBoxOption[] = new Array(this.props.jointDataset.datasetFeatureCount).fill(0)
        .map((unused, index) => {
            const key = JointDataset.DataLabelRoot + index.toString();
            return {key, text: this.props.jointDataset.metaDict[key].abbridgedLabel}
        });

    private readonly leftItems = [
        JointDataset.IndexLabel,
        JointDataset.DataLabelRoot,
        JointDataset.PredictedYLabel,
        JointDataset.TrueYLabel,
        JointDataset.ClassificationError,
        JointDataset.RegressionError
    ].map(key => {
        const metaVal = this.props.jointDataset.metaDict[key];
        if (key === JointDataset.DataLabelRoot) {
            return {key, title: "Dataset"};
        }
        if  (metaVal === undefined) {
            return undefined;
        }
        return {key, title: metaVal.abbridgedLabel};
    }).filter(obj => obj !== undefined);

    private comparisonOptions: IComboBoxOption[] = [
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
    private _isInitialized = false;

    constructor(props: IFilterEditorProps) {
        super(props);
        this.state = _.cloneDeep(this.props.initialFilter);
        this._leftSelection = new Selection({
            selectionMode: SelectionMode.single,
            onSelectionChanged: this._setSelection
          });
        this._leftSelection.setItems(this.leftItems);
        this._leftSelection.setKeySelected(this.extractSelectionKey(this.props.initialFilter.column), true, false);
        this._isInitialized = true;
    }

    public render(): React.ReactNode {
        const selectedMeta = this.props.jointDataset.metaDict[this.state.column];
        const numericDelta = selectedMeta.treatAsCategorical || selectedMeta.featureRange.rangeType === RangeTypes.integer ?
            1 : (selectedMeta.featureRange.max - selectedMeta.featureRange.min)/10;
        const isDataColumn = this.state.column.indexOf(JointDataset.DataLabelRoot) !== -1;
        const categoricalOptions: IComboBoxOption[] = selectedMeta.treatAsCategorical ?
            selectedMeta.sortedCategoricalValues.map((label, index) => {return {key: index, text: label}}) : [];
        return (
            <Callout
                target={this.props.target ? '#' + this.props.target : undefined}
                onDismiss={this.props.onCancel}
                setInitialFocus={true}
                hidden={false}
            >
                <div className={styles.wrapper}>
                    <div className={styles.leftHalf}>
                        <DetailsList
                            items={this.leftItems}
                            ariaLabelForSelectionColumn="Toggle selection"
                            ariaLabelForSelectAllCheckbox="Toggle selection for all items"
                            checkButtonAriaLabel="Row checkbox"
                            onRenderDetailsHeader={this._onRenderDetailsHeader}
                            selection={this._leftSelection}
                            selectionPreservedOnEmptyClick={true}
                            setKey={"set"}
                            columns={[{key: 'col1', name: 'name', minWidth: 200, fieldName: 'title'}]}
                        />
                    </div>
                    <div className={styles.rightHalf}>
                        {isDataColumn && (
                            <ComboBox
                                options={this.dataArray}
                                onChange={this.setSelectedProperty}
                                label={"Feature: "}
                                ariaLabel="feature picker"
                                selectedKey={this.state.column}
                                useComboBoxAsMenuWidth={true}
                                styles={FabricStyles.defaultDropdownStyle} />
                        )}
                        {selectedMeta.featureRange && selectedMeta.featureRange.rangeType === RangeTypes.integer && (
                            <Checkbox label="Treat as categorical" checked={selectedMeta.treatAsCategorical} onChange={this.setAsCategorical} />
                        )}
                        <div>Data summary</div>
                        {selectedMeta.treatAsCategorical && (
                            <div>
                                <div>{`# of unique values: ${selectedMeta.sortedCategoricalValues.length}`}</div>
                                <ComboBox
                                    multiSelect
                                    label={localization.Filters.categoricalIncludeValues}
                                    className="path-selector"
                                    selectedKey={this.state.arg}
                                    onChange={this.setCategoricalValues}
                                    options={categoricalOptions}
                                    useComboBoxAsMenuWidth={true}
                                    styles={FabricStyles.smallDropdownStyle}
                                />
                            </div>
                        )}
                        {!selectedMeta.treatAsCategorical && (
                            <div>
                                <div>{`min: ${selectedMeta.featureRange.min}`}</div>
                                <div>{`max: ${selectedMeta.featureRange.max}`}</div>
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
                                    min={selectedMeta.featureRange.min}
                                    max={selectedMeta.featureRange.max}
                                    value={this.state.arg.toString()}
                                    onIncrement={this.setNumericValue.bind(this, numericDelta, selectedMeta)}
                                    onDecrement={this.setNumericValue.bind(this, -numericDelta, selectedMeta)}
                                    onValidate={this.setNumericValue.bind(this, 0, selectedMeta)}
                                />
                            </div>
                        )}
                    </div>
                </div>
                <DefaultButton 
                    text={"Accept"}
                    onClick={this.saveState}
                />
            </Callout>
        );
    }

    private extractSelectionKey(key: string): string {
        let index = key.indexOf(JointDataset.DataLabelRoot);
        if (index !== -1) {
            return JointDataset.DataLabelRoot;
        }
        return key;
    }

    private readonly setAsCategorical = (ev: React.FormEvent<HTMLElement>, checked: boolean): void => {
        this.props.jointDataset.metaDict[this.state.column].treatAsCategorical = checked;
        if (checked) {
            this.props.jointDataset.addBin(this.state.column,
                this.props.jointDataset.metaDict[this.state.column].featureRange.max + 1);
        }
        this.forceUpdate();
    }

    private readonly _setSelection = (): void => {
        if (!this._isInitialized) {
            return;
        }
        let property = this._leftSelection.getSelection()[0].key as string;
        if (property === JointDataset.DataLabelRoot) {
            property += "0";
        }
        this.setDefaultStateForKey(property);
    }

    private readonly setSelectedProperty = (event: React.FormEvent<IComboBox>, item: IComboBoxOption): void => {
        const property = item.key as string;
        this.setDefaultStateForKey(property);
    }

    private readonly saveState = (): void => {
        this.props.onAccept(this.state);
    }

    private readonly _onRenderDetailsHeader = () => {
        return <div></div>
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
            if ((!Number.isInteger(number) && column.featureRange.rangeType === RangeTypes.integer)
                || number > column.featureRange.max || number < column.featureRange.min) {
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

    private setDefaultStateForKey(key: string): void {
        const filter: IFilter = {column: key} as IFilter;
        const meta = this.props.jointDataset.metaDict[key];
        if (meta.isCategorical) {
            filter.method = FilterMethods.includes;
            filter.arg = Array.from(Array(meta.sortedCategoricalValues.length).keys());
        } else {
            filter.method = FilterMethods.lessThan;
            filter.arg = meta.featureRange.max;
        }
        this.setState(filter);
    }
}