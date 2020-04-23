import _ from "lodash";
import { RangeTypes } from "mlchartlib";
import { IStyle, Text, TextField } from "office-ui-fabric-react";
import { DefaultButton, IconButton, PrimaryButton } from "office-ui-fabric-react/lib/Button";
import { Callout, ICalloutContentStyles } from "office-ui-fabric-react/lib/Callout";
import { Checkbox } from "office-ui-fabric-react/lib/Checkbox";
import { ComboBox, IComboBox, IComboBoxOption } from "office-ui-fabric-react/lib/ComboBox";
import { CheckboxVisibility, DetailsList, Selection, SelectionMode } from "office-ui-fabric-react/lib/DetailsList";
import { SpinButton } from 'office-ui-fabric-react/lib/SpinButton';
import { Position } from "office-ui-fabric-react/lib/utilities/positioning";
import React from "react";
import { localization } from "../../../Localization/localization";
import { Cohort } from "../../Cohort";
import { FilterMethods, IFilter } from "../../Interfaces/IFilter";
import { IJointMeta, JointDataset } from "../../JointDataset";
import { cohortEditorStyles } from "./CohortEditor.styles";

export interface ICohortEditorProps {
    jointDataset: JointDataset;
    filterList: IFilter[];
    cohortName: string;
    onSave: (newCohort: Cohort) => void;
    onCancel: () => void;
}

export interface ICohortEditorState {
    openedFilter?: IFilter;
    filterIndex?: number;
    filters?: IFilter[];
    isEditingFilter?: boolean;
    cohortName: string;
    cohortErrorName: string;
}

//TODO: move this to style file
let cohortEditor: IStyle = {
    position: 'absolute',
    overflowY: 'visible',
    width: '560px',
    height: '624px',
    //alignSelf:"center",
    left: '500px',
    top: '100px',
    //TODO: fix rgba here
    boxShadow: '0px 0.6px 1.8px rgba(0, 0, 0, 0.108), 0px 3.2px 7.2px rgba(0, 0, 0, 0.132)',
    borderRadius: '2px'
}

let calloutMain: ICalloutContentStyles = {
    container: {},
    root: {},
    beak: {},
    beakCurtain: {},
    calloutMain: cohortEditor
}


const styles = cohortEditorStyles();


export class CohortEditor extends React.PureComponent<ICohortEditorProps, ICohortEditorState> {
    private _leftSelection: Selection;
    private readonly dataArray: IComboBoxOption[] = new Array(this.props.jointDataset.datasetFeatureCount).fill(0)
        .map((unused, index) => {
            const key = JointDataset.DataLabelRoot + index.toString();
            return { key, text: this.props.jointDataset.metaDict[key].abbridgedLabel }
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
            return { key, title: "Dataset" };
        }
        if (metaVal === undefined) {
            return undefined;
        }
        return { key, title: metaVal.abbridgedLabel };
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
        },
        {
            key: FilterMethods.inTheRangeOf,
            text: localization.Filters.inTheRangeOf
        }
    ];
    private _isInitialized = false;

    constructor(props: ICohortEditorProps) {
        super(props);
        this.state = {
            openedFilter: undefined,
            filterIndex: this.props.filterList.length,
            filters: this.props.filterList,
            isEditingFilter: false,
            cohortName: this.props.cohortName,
            cohortErrorName: ""
        };
        this._leftSelection = new Selection({
            selectionMode: SelectionMode.single,
            onSelectionChanged: this._setSelection
        });
        this._leftSelection.setItems(this.leftItems);
        this._isInitialized = true;

        this.saveCohort = this.saveCohort.bind(this);
        this.setCohortName = this.setCohortName.bind(this);
    }

    public render(): React.ReactNode {
        const openedFilter = this.state.openedFilter;

        const filterList = this.state.filters.map((filter, index) => {
            return (<div key={index} className={styles.existingFilter}>
                {this.setFilterLabel(filter)}
                <IconButton
                    className={styles.filterIcon}
                    iconProps={{ iconName: "Edit" }}
                    onClick={this.editFilter.bind(this, index)}
                />
                <IconButton
                    className={styles.filterIcon}
                    iconProps={{ iconName: "Clear" }}
                    onClick={this.removeFilter.bind(this, index)}
                />
            </div>);
        });
        return (
            <Callout
                onDismiss={this.props.onCancel}
                setInitialFocus={true}
                hidden={false}
                styles={calloutMain}
            >
                <div className={styles.container}>
                    <div className={styles.cohortName}>
                        <TextField
                            value={this.state.cohortName}
                            label={localization.CohortEditor.cohortNameLabel}
                            placeholder={localization.CohortEditor.cohortNamePlaceholder}
                            onGetErrorMessage= {this._getErrorMessage}
                            validateOnLoad={false}
                            onChange={this.setCohortName} />
                    </div>

                    <div className={styles.wrapper}>
                        <div className={styles.leftHalf}>
                            <DetailsList
                                className={styles.detailedList}
                                items={this.leftItems}
                                ariaLabelForSelectionColumn="Toggle selection"
                                ariaLabelForSelectAllCheckbox="Toggle selection for all items"
                                checkButtonAriaLabel="Row checkbox"
                                checkboxVisibility={CheckboxVisibility.hidden}
                                onRenderDetailsHeader={this._onRenderDetailsHeader}
                                selection={this._leftSelection}
                                selectionPreservedOnEmptyClick={false}
                                setKey={"set"}
                                columns={[{ key: 'col1', name: 'name', minWidth: 150, fieldName: 'title' }]}
                            />
                        </div>
                        {openedFilter == undefined ?
                            <div className={styles.rightHalf}>
                                <Text className={styles.defaultText} variant={"medium"}>
                                    {localization.CohortEditor.defaultFilterState}
                                </Text>
                            </div>
                            :
                            this.buildRightPanel(openedFilter)
                        }
                    </div>
                    <div>
                        <Text variant={"medium"} className={styles.addedFilter} >{localization.CohortEditor.addedFilters}</Text>
                        <div className={styles.addedFilterDiv}>
                            {filterList.length > 0
                                ? <div>{filterList}</div>
                                : <div>
                                    <Text variant={"smallPlus"} className={styles.defaultFilterList}>{localization.CohortEditor.noAddedFilters}</Text>
                                </div>
                            }
                        </div>
                    </div>
                    <PrimaryButton onClick={this.saveCohort} className={styles.saveCohort}>Save</PrimaryButton>
                </div>
            </Callout>
        );
    }

    private readonly setAsCategorical = (ev: React.FormEvent<HTMLElement>, checked: boolean): void => {
        const openedFilter = this.state.openedFilter;
        this.props.jointDataset.setTreatAsCategorical(openedFilter.column, checked);
        if (checked) {
            this.setState(
                {
                    openedFilter: {
                        arg: [],
                        method: FilterMethods.includes,
                        column: openedFilter.column
                    }
                }
            )
        } else {
            this.setState(
                {
                    openedFilter: {
                        arg: this.props.jointDataset.metaDict[openedFilter.column].featureRange.max,
                        method: FilterMethods.lessThan,
                        column: openedFilter.column
                    }
                }
            )
        }
    }

    private _getErrorMessage = (): string => {
       if (this.state.cohortName.length<=0){
           return localization.CohortEditor.cohortNameError;
       }
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

    private saveState = (): void => {
        this.updateFilter(this.state.openedFilter);
    }

    private readonly _onRenderDetailsHeader = () => {
        return <div className={styles.filterHeader}>{localization.CohortEditor.selectFilter}</div>
    }

    private readonly setCategoricalValues = (event: React.FormEvent<IComboBox>, item: IComboBoxOption): void => {
        const openedFilter = this.state.openedFilter;
        const selectedVals = [...(openedFilter.arg as number[])];

        const index = selectedVals.indexOf(item.key as number);
        if (item.selected && index === -1) {
            selectedVals.push(item.key as number);
        } else {
            selectedVals.splice(index, 1);
        }
        this.setState(
            {
                openedFilter: {
                    arg: selectedVals,
                    method: openedFilter.method,
                    column: openedFilter.column
                }
            }
        )
    }

    private readonly setComparison = (event: React.FormEvent<IComboBox>, item: IComboBoxOption): void => {
        const openedFilter = this.state.openedFilter;
        this.setState(
            {
                openedFilter: {
                    arg: openedFilter.arg,
                    method: item.key as FilterMethods,
                    column: openedFilter.column
                }
            }
        )
    }

    private readonly setNumericValue = (delta: number, column: IJointMeta, stringVal: string): string | void => {
        const openedFilter = this.state.openedFilter;
        if (delta === 0) {
            const number = +stringVal;
            if ((!Number.isInteger(number) && column.featureRange.rangeType === RangeTypes.integer)
                || number > column.featureRange.max || number < column.featureRange.min) {
                return this.state.openedFilter.arg.toString();
            }

            this.setState(
                {
                    openedFilter: {
                        arg: number,
                        method: openedFilter.method,
                        column: openedFilter.column
                    }
                }
            )
        } else {
            const prevVal = openedFilter.arg as number;
            const newVal = prevVal + delta;
            if (newVal > column.featureRange.max || newVal < column.featureRange.min) {
                return prevVal.toString();
            }
            this.setState(
                {
                    openedFilter: {
                        arg: newVal,
                        method: openedFilter.method,
                        column: openedFilter.column
                    }
                }
            )
        }
    }

    private setDefaultStateForKey(key: string): void {
        let filter: IFilter = { column: key } as IFilter;
        const meta = this.props.jointDataset.metaDict[key];
        if (meta.isCategorical) {
            filter.method = FilterMethods.includes;
            filter.arg = Array.from(Array(meta.sortedCategoricalValues.length).keys());
        } else if (meta.treatAsCategorical) {
            filter.method = FilterMethods.includes;
            filter.arg = meta.sortedCategoricalValues as any[];
        } else {
            filter.method = FilterMethods.lessThan;
            filter.arg = meta.featureRange.max;
        }
        this.setState(
            {
                openedFilter: filter
            }
        )
    }

    private updateFilter(filter: IFilter): void {
        const isEditingFilter = this.state.isEditingFilter;
        let filters = [...this.state.filters];
        if (!isEditingFilter) {
            filters.push(filter);
        }
        else {
            filters[this.state.filterIndex] = filter;
        }

        this.setState({ filters });
        this.setState({ openedFilter: undefined, filterIndex: undefined, isEditingFilter: false });
    }

    private cancelFilter = (): void => {
        this.setState({ openedFilter: undefined, isEditingFilter: false });
    }

    private removeFilter(index: number): void {
        let filters = [...this.state.filters];
        filters.splice(index, 1);
        this.setState({ filters });
    }

    private editFilter(index: number): void {
        const editFilter = this.state.filters[index];
        this.setState({ isEditingFilter: true, filterIndex: index });
        this.setState({ openedFilter: _.cloneDeep(editFilter) });
    }

    private saveCohort(): void {
        if (this.state.cohortName.length > 0){
            let newCohort = new Cohort(this.state.cohortName, this.props.jointDataset, this.state.filters);
            this.props.onSave(newCohort);
        }
        else {
            this._getErrorMessage
        }
    }

    private setCohortName(event): void {
        this.setState({ cohortName: event.target.value });   
    }

    private setFilterLabel(filter: IFilter): React.ReactNode {
        //TODO: change the function unwrap neatly 
        let label = "";
        const space = " ";
        label = this.props.jointDataset.metaDict[filter.column].abbridgedLabel + space
        label += filter.method + space
        label += filter.arg

        return (<Text variant={"small"} className={styles.filterLabel}>{label}</Text>);
    }

    private buildRightPanel(openedFilter): React.ReactNode {
        const selectedMeta = this.props.jointDataset.metaDict[openedFilter.column];
        const numericDelta = selectedMeta.treatAsCategorical || selectedMeta.featureRange.rangeType === RangeTypes.integer ?
            1 : (selectedMeta.featureRange.max - selectedMeta.featureRange.min) / 10;
        const isDataColumn = openedFilter.column.indexOf(JointDataset.DataLabelRoot) !== -1;
        let categoricalOptions: IComboBoxOption[] = [];
        if (selectedMeta.treatAsCategorical) {
            // Numerical values treated as categorical are stored with the values in the column,
            // true categorical values store indexes to the string values
            categoricalOptions = selectedMeta.isCategorical ?
                selectedMeta.sortedCategoricalValues.map((label, index) => { return { key: index, text: label } }) :
                selectedMeta.sortedCategoricalValues.map((label) => { return { key: label, text: label.toString() } })
        }
        return (
            <div className={styles.rightHalf}>
                {isDataColumn && (
                    <ComboBox
                        className={styles.featureComboBox}
                        options={this.dataArray}
                        onChange={this.setSelectedProperty}
                        label={localization.CohortEditor.selectFilter}
                        selectedKey={openedFilter.column} />
                )}
                {selectedMeta.featureRange && selectedMeta.featureRange.rangeType === RangeTypes.integer && (
                    <Checkbox className={styles.treatCategorical}
                        label={localization.CohortEditor.TreatAsCategorical}
                        checked={selectedMeta.treatAsCategorical}
                        onChange={this.setAsCategorical} />
                )}
                {selectedMeta.treatAsCategorical && (
                    <div className={styles.featureTextDiv}>
                        <Text variant={"small"} className={styles.featureText}>
                            {`# of unique values: ${selectedMeta.sortedCategoricalValues.length}`}
                        </Text>
                        <ComboBox
                            multiSelect
                            label={localization.Filters.categoricalIncludeValues}
                            className={styles.operationComboBox}
                            selectedKey={openedFilter.arg}
                            onChange={this.setCategoricalValues}
                            options={categoricalOptions}
                            useComboBoxAsMenuWidth={true}
                        />
                    </div>
                )}
                {!selectedMeta.treatAsCategorical && (
                    <div className={styles.featureTextDiv}>
                        <Text variant={"small"} className={styles.featureText}>
                            {`Min: ${selectedMeta.featureRange.min}`} {`Max: ${selectedMeta.featureRange.max}`}
                        </Text>
                        <ComboBox
                            label={localization.Filters.numericalComparison}
                            className={styles.operationComboBox}
                            selectedKey={openedFilter.method}
                            onChange={this.setComparison}
                            options={this.comparisonOptions}
                            useComboBoxAsMenuWidth={true}
                        />
                        {openedFilter.method == FilterMethods.inTheRangeOf ?
                            <div className={styles.valueSpinButtonDiv}>
                                <SpinButton
                                    labelPosition={Position.top}
                                    className={styles.minSpinBox}
                                    label={localization.Filters.minimum}
                                    min={selectedMeta.featureRange.min}
                                    max={selectedMeta.featureRange.max}
                                />
                                <SpinButton
                                    labelPosition={Position.top}
                                    className={styles.maxSpinBox}
                                    label={localization.Filters.maximum}
                                    min={selectedMeta.featureRange.min}
                                    max={selectedMeta.featureRange.max}
                                />
                            </div>
                            :
                            <div className={styles.valueSpinButtonDiv}>
                                <SpinButton
                                    labelPosition={Position.top}
                                    className={styles.valueSpinButton}
                                    label={localization.Filters.numericValue}
                                    min={selectedMeta.featureRange.min}
                                    max={selectedMeta.featureRange.max}
                                    value={openedFilter.arg.toString()}
                                    onIncrement={this.setNumericValue.bind(this, numericDelta, selectedMeta)}
                                    onDecrement={this.setNumericValue.bind(this, -numericDelta, selectedMeta)}
                                    onValidate={this.setNumericValue.bind(this, 0, selectedMeta)}
                                />
                            </div>
                        }
                    </div>
                )}
                {this.state.isEditingFilter ?
                    <div className={styles.saveAndCancelDiv}>
                        <DefaultButton
                            className={styles.saveFilterButton}
                            text={localization.CohortEditor.save}
                            onClick={this.saveState}
                        />
                        <DefaultButton
                            className={styles.cancelFilterButton}
                            text={localization.CohortEditor.cancel}
                            onClick={this.cancelFilter.bind(this)}
                        />
                    </div>
                    :
                    <DefaultButton
                        className={styles.addFilterButton}
                        text={localization.CohortEditor.addFilter}
                        onClick={this.saveState} />
                }
            </div>
        );
    }
}